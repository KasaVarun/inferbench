"""Local causal-LM inference on Apple MPS (with CPU fallback), Phase 4.

Uses PyTorch + Hugging Face `transformers` to load a small causal language
model and run deterministic, greedy generation locally -- no CUDA, no
remote GPU, no network access anywhere except inside `load()` itself
(downloading model/tokenizer weights, which is explicit and only happens
when a caller actually calls it).

Device selection is intentionally simple:

- ``"cpu"``: always available, used as the safe fallback.
- ``"mps"``: Apple's Metal Performance Shaders backend. Available only if
  both ``torch.backends.mps.is_built()`` (PyTorch was compiled with MPS
  support) and ``torch.backends.mps.is_available()`` (an MPS device is
  actually present/usable in this process) are true.
- ``"auto"``: prefers MPS, falls back to CPU.

There is no CUDA branch here at all -- Apple Silicon has no NVIDIA GPU,
and pretending otherwise would misrepresent the hardware.
"""

from __future__ import annotations

import time
from typing import Any

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark_core.inference_backend import (
    BackendInfo,
    GenerationRequest,
    GenerationResponse,
    InferenceBackend,
)

__all__ = ["LocalTransformersBackend", "is_mps_available", "resolve_device"]

_SUPPORTED_DEVICES = ("auto", "mps", "cpu")


def is_mps_available() -> bool:
    """Return whether Apple MPS is both compiled-in and usable right now.

    A thin, directly-mockable wrapper around the two checks PyTorch
    recommends combining: a PyTorch build without MPS support can still
    report varying things from `is_available()`, and a machine without an
    MPS-capable GPU will report `is_available() == False` even on an
    MPS-built PyTorch. Both must be true to actually use MPS.
    """
    return bool(torch.backends.mps.is_built()) and bool(torch.backends.mps.is_available())


def resolve_device(requested: str) -> str:
    """Resolve a requested device (``auto``/``mps``/``cpu``) to an actual one.

    Raises:
        ValueError: If `requested` is not one of the supported values, or
            `requested == "mps"` but MPS is not available on this machine.
            The latter is a deliberate hard failure -- explicitly asking
            for MPS should never silently fall back to CPU.
    """
    if requested not in _SUPPORTED_DEVICES:
        raise ValueError(f"unknown device '{requested}'; expected one of {_SUPPORTED_DEVICES}")

    if requested == "cpu":
        return "cpu"
    if requested == "mps":
        if not is_mps_available():
            raise ValueError("device 'mps' was requested but MPS is not available on this machine")
        return "mps"
    return "mps" if is_mps_available() else "cpu"


def _resolve_dtype(device: str) -> torch.dtype:
    """Pick a dtype that is reliably supported on `device`.

    CPU: float32 -- the universally reliable default; float16 matmul
    support on CPU is inconsistent across PyTorch builds.
    MPS: float16 -- generally stable for small causal LMs on Apple's MPS
    backend and halves memory/bandwidth versus float32.
    """
    return torch.float16 if device == "mps" else torch.float32


def _dtype_name(dtype: torch.dtype) -> str:
    # `str(torch.float16)` is `"torch.float16"`; strip the module prefix so
    # this matches the plain dtype names used elsewhere (e.g. "float32").
    return str(dtype).removeprefix("torch.")


def _synchronize(device: str) -> None:
    """Synchronize `device` before/after timed generation, if it needs it.

    CPU execution is already synchronous from Python's perspective, so
    there is nothing to synchronize. MPS dispatches work asynchronously,
    so an un-synchronized `time.perf_counter()` window would measure
    dispatch time, not actual completion time. `torch.mps.synchronize()`
    is the API PyTorch documents for this (checked via `hasattr` rather
    than assumed, since it was only added in newer PyTorch releases).
    """
    if device == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


class LocalTransformersBackend(InferenceBackend):
    """Runs a small causal LM locally via `transformers`, on MPS or CPU.

    No model or tokenizer is loaded until `load()` is called explicitly --
    constructing this class does no I/O and touches no network.
    """

    def __init__(self, model_name: str, device: str = "auto") -> None:
        self._model_name = model_name
        self._requested_device = device
        self._resolved_device: str | None = None
        self._dtype: torch.dtype | None = None
        # `transformers`'s `Auto*` factory classes dynamically select a
        # concrete model/tokenizer subclass at runtime based on the
        # model's config -- there is no single static type to annotate
        # these as (see module docstring / docs/local-inference.md for
        # why `Any` is the honest annotation here, not a shortcut).
        self._model: Any = None
        self._tokenizer: Any = None

    def load(self) -> None:
        """Resolve the device/dtype and load the tokenizer and model.

        Raises whatever `resolve_device`, `AutoTokenizer.from_pretrained`,
        or `AutoModelForCausalLM.from_pretrained` raise -- callers (e.g.
        the CLI) are responsible for catching and reporting these clearly;
        this method does not swallow or reinterpret them.
        """
        device = resolve_device(self._requested_device)
        dtype = _resolve_dtype(device)

        # `Any`-typed locals here (not just the attributes): `transformers`'s
        # dynamically-dispatched `Auto*` return types confuse mypy's
        # overload resolution for otherwise-unrelated calls like
        # `Module.to()` below -- see the module-level override comment in
        # `pyproject.toml` for why this integration boundary is `Any`.
        tokenizer: Any = AutoTokenizer.from_pretrained(self._model_name)
        if tokenizer.pad_token_id is None:
            # Many small causal LMs (e.g. GPT-2-family) ship with no pad
            # token at all. Reusing eos as pad is the standard, documented
            # workaround for single-sequence generation like ours.
            tokenizer.pad_token = tokenizer.eos_token

        model: Any = AutoModelForCausalLM.from_pretrained(self._model_name, dtype=dtype)
        model.to(device)
        model.eval()

        self._resolved_device = device
        self._dtype = dtype
        self._tokenizer = tokenizer
        self._model = model

    def health(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        self._resolved_device = None
        self._dtype = None

    def info(self) -> BackendInfo:
        if not self.health():
            raise RuntimeError("backend is not loaded; call load() first")
        assert self._resolved_device is not None
        assert self._dtype is not None
        return BackendInfo(
            model_name=self._model_name,
            device=self._resolved_device,
            dtype=_dtype_name(self._dtype),
            framework_version=f"torch {torch.__version__}; transformers {transformers.__version__}",
        )

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if not self.health():
            raise RuntimeError("backend is not loaded; call load() first")
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._resolved_device is not None

        tokenizer = self._tokenizer
        model = self._model
        device = self._resolved_device

        encoded = tokenizer(request.prompt, return_tensors="pt").to(device)
        prompt_tokens = int(encoded["input_ids"].shape[1])

        _synchronize(device)
        start = time.perf_counter()
        with torch.no_grad():
            generated_ids = model.generate(
                **encoded,
                max_new_tokens=request.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        _synchronize(device)
        latency_ms = (time.perf_counter() - start) * 1000.0

        completion_ids = generated_ids[0][prompt_tokens:]
        completion_tokens = int(completion_ids.shape[0])
        text = tokenizer.decode(completion_ids, skip_special_tokens=True)

        return GenerationResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )
