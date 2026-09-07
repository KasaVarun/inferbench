"""A fake `InferenceBackend` for exercising `workload_runner` without a real model.

No `torch`, no `transformers`, no network, no MPS -- this is a plain
Python stand-in used only by tests, so `workload_runner`'s orchestration
(warmup, per-request success/failure recording, aggregation, result
construction) can be tested in isolation from `LocalTransformersBackend`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from benchmark_core.inference_backend import (
    BackendInfo,
    GenerationRequest,
    GenerationResponse,
    InferenceBackend,
)


@dataclass
class FakeBackend(InferenceBackend):
    model_name: str = "fake-model"
    device: str = "cpu"
    dtype: str = "float32"
    latency_ms: float = 1.0
    fail_prompts: frozenset[str] = field(default_factory=frozenset)
    fail_all_with: str | None = None
    load_should_fail: bool = False

    _loaded: bool = field(default=False, init=False, repr=False)
    calls: list[str] = field(default_factory=list, init=False, repr=False)

    def load(self) -> None:
        if self.load_should_fail:
            raise RuntimeError("simulated model load failure")
        self._loaded = True

    def health(self) -> bool:
        return self._loaded

    def close(self) -> None:
        # Idempotent: closing an already-closed (or never-loaded) backend
        # is a no-op, not an error.
        self._loaded = False

    def info(self) -> BackendInfo:
        if not self._loaded:
            raise RuntimeError("backend is not loaded; call load() first")
        return BackendInfo(
            model_name=self.model_name,
            device=self.device,
            dtype=self.dtype,
            framework_version="fake-framework 0.0",
        )

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if not self._loaded:
            raise RuntimeError("backend is not loaded; call load() first")
        self.calls.append(request.prompt)
        if self.fail_all_with is not None:
            raise RuntimeError(self.fail_all_with)
        if request.prompt in self.fail_prompts:
            raise RuntimeError(f"simulated generation failure for prompt: {request.prompt!r}")

        # Deterministic, network-free stand-in for a real tokenizer: a
        # simple, fixed character-based count distinct from Phase 3's
        # estimator, so tests can tell the two apart.
        prompt_tokens = max(1, len(request.prompt) // 4)
        completion_tokens = request.max_new_tokens
        text = " ".join(["fake"] * completion_tokens)
        return GenerationResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=self.latency_ms,
        )


def make_fake_backend(
    *, fail_prompts: Iterable[str] = (), load_should_fail: bool = False, latency_ms: float = 1.0
) -> FakeBackend:
    return FakeBackend(
        latency_ms=latency_ms,
        fail_prompts=frozenset(fail_prompts),
        load_should_fail=load_should_fail,
    )
