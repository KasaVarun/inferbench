"""Tests for device selection and `LocalTransformersBackend` (Phase 4).

No test here downloads a model, reaches huggingface.co, or requires real
MPS hardware: `torch.backends.mps.is_built`/`is_available` are monkeypatched,
and `transformers.AutoTokenizer`/`AutoModelForCausalLM` are monkeypatched
with in-process fakes for the load/generate tests.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from benchmark_core.inference_backend import GenerationRequest
from benchmark_core.local_transformers_backend import (
    LocalTransformersBackend,
    is_mps_available,
    resolve_device,
)

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------


def _mock_mps(monkeypatch: pytest.MonkeyPatch, *, built: bool, available: bool) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: built)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: available)


def test_is_mps_available_requires_both_built_and_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_mps(monkeypatch, built=True, available=True)
    assert is_mps_available() is True

    _mock_mps(monkeypatch, built=True, available=False)
    assert is_mps_available() is False

    _mock_mps(monkeypatch, built=False, available=True)
    assert is_mps_available() is False


def test_auto_selects_mps_when_mocked_available(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_mps(monkeypatch, built=True, available=True)
    assert resolve_device("auto") == "mps"


def test_auto_falls_back_to_cpu_when_mps_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_mps(monkeypatch, built=True, available=False)
    assert resolve_device("auto") == "cpu"


def test_explicit_cpu_selection_always_returns_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_mps(monkeypatch, built=True, available=True)
    assert resolve_device("cpu") == "cpu"


def test_explicit_mps_selection_succeeds_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_mps(monkeypatch, built=True, available=True)
    assert resolve_device("mps") == "mps"


def test_explicit_mps_selection_fails_clearly_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_mps(monkeypatch, built=True, available=False)
    with pytest.raises(ValueError, match="MPS is not available"):
        resolve_device("mps")


def test_unknown_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown device"):
        resolve_device("cuda")


# ---------------------------------------------------------------------------
# No model load at import / construction time
# ---------------------------------------------------------------------------


def test_construction_does_not_load_a_model() -> None:
    backend = LocalTransformersBackend(model_name="not-a-real-model", device="cpu")
    assert backend.health() is False


def test_health_before_load_is_false() -> None:
    backend = LocalTransformersBackend(model_name="not-a-real-model", device="cpu")
    assert backend.health() is False


def test_info_before_load_raises() -> None:
    backend = LocalTransformersBackend(model_name="not-a-real-model", device="cpu")
    with pytest.raises(RuntimeError, match="not loaded"):
        backend.info()


def test_generate_before_load_raises() -> None:
    backend = LocalTransformersBackend(model_name="not-a-real-model", device="cpu")
    with pytest.raises(RuntimeError, match="not loaded"):
        backend.generate(GenerationRequest(prompt="hello", max_new_tokens=4))


def test_close_before_load_is_a_safe_no_op() -> None:
    backend = LocalTransformersBackend(model_name="not-a-real-model", device="cpu")
    backend.close()
    backend.close()  # idempotent
    assert backend.health() is False


# ---------------------------------------------------------------------------
# Fake tokenizer/model doubles for load()/generate() tests (no network,
# no real model -- these stand in for what AutoTokenizer/AutoModelForCausalLM
# would return).
# ---------------------------------------------------------------------------


class _FakeEncoding(dict[str, Any]):
    def to(self, _device: str) -> _FakeEncoding:
        return self


class _FakeTokenizer:
    def __init__(self, pad_token_id: int | None = None) -> None:
        self.pad_token_id = pad_token_id
        self.pad_token: str | None = None
        self.eos_token = "<eos>"

    def __call__(self, prompt: str, return_tensors: str = "pt") -> _FakeEncoding:
        # One fake token per character -- deterministic, no real tokenizer.
        length = max(1, len(prompt))
        return _FakeEncoding({"input_ids": torch.zeros((1, length), dtype=torch.long)})

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        return " ".join(["gen"] * int(token_ids.shape[0]))


class _FakeModel:
    def __init__(self) -> None:
        self.eval_called = False
        self.device: str | None = None

    def to(self, device: str) -> _FakeModel:
        self.device = device
        return self

    def eval(self) -> None:
        self.eval_called = True

    def generate(
        self, *, input_ids: torch.Tensor, max_new_tokens: int, do_sample: bool, pad_token_id: int
    ) -> torch.Tensor:
        assert do_sample is False
        prompt_len = input_ids.shape[1]
        new_tokens = torch.ones((1, max_new_tokens), dtype=torch.long)
        return torch.cat([input_ids, new_tokens], dim=1)[:, : prompt_len + max_new_tokens]


def _patch_transformers(
    monkeypatch: pytest.MonkeyPatch, *, tokenizer: _FakeTokenizer, model: _FakeModel
) -> None:
    monkeypatch.setattr(
        "benchmark_core.local_transformers_backend.AutoTokenizer.from_pretrained",
        lambda *_a, **_k: tokenizer,
    )
    monkeypatch.setattr(
        "benchmark_core.local_transformers_backend.AutoModelForCausalLM.from_pretrained",
        lambda *_a, **_k: model,
    )


# ---------------------------------------------------------------------------
# load() / generate() behavior via fakes
# ---------------------------------------------------------------------------


def test_load_resolves_device_dtype_and_marks_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_mps(monkeypatch, built=True, available=False)
    tokenizer = _FakeTokenizer(pad_token_id=0)
    model = _FakeModel()
    _patch_transformers(monkeypatch, tokenizer=tokenizer, model=model)

    backend = LocalTransformersBackend(model_name="fake/model", device="auto")
    assert backend.health() is False

    backend.load()

    assert backend.health() is True
    info = backend.info()
    assert info.device == "cpu"
    assert info.dtype == "float32"
    assert info.model_name == "fake/model"
    assert "torch" in info.framework_version
    assert "transformers" in info.framework_version
    assert model.eval_called is True
    assert model.device == "cpu"


def test_load_on_mps_uses_float16(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_mps(monkeypatch, built=True, available=True)
    tokenizer = _FakeTokenizer(pad_token_id=0)
    model = _FakeModel()
    _patch_transformers(monkeypatch, tokenizer=tokenizer, model=model)

    backend = LocalTransformersBackend(model_name="fake/model", device="mps")
    backend.load()

    assert backend.info().device == "mps"
    assert backend.info().dtype == "float16"


def test_load_sets_pad_token_when_tokenizer_has_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_mps(monkeypatch, built=False, available=False)
    tokenizer = _FakeTokenizer(pad_token_id=None)
    model = _FakeModel()
    _patch_transformers(monkeypatch, tokenizer=tokenizer, model=model)

    backend = LocalTransformersBackend(model_name="fake/model", device="cpu")
    backend.load()

    assert tokenizer.pad_token == tokenizer.eos_token


def test_load_does_not_touch_pad_token_when_already_set(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_mps(monkeypatch, built=False, available=False)
    tokenizer = _FakeTokenizer(pad_token_id=7)
    model = _FakeModel()
    _patch_transformers(monkeypatch, tokenizer=tokenizer, model=model)

    backend = LocalTransformersBackend(model_name="fake/model", device="cpu")
    backend.load()

    assert tokenizer.pad_token is None  # never assigned, since pad_token_id was already set


def test_generate_uses_deterministic_do_sample_false(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_mps(monkeypatch, built=False, available=False)
    tokenizer = _FakeTokenizer(pad_token_id=0)
    model = _FakeModel()
    _patch_transformers(monkeypatch, tokenizer=tokenizer, model=model)

    backend = LocalTransformersBackend(model_name="fake/model", device="cpu")
    backend.load()

    response = backend.generate(GenerationRequest(prompt="hello world", max_new_tokens=5))

    assert response.completion_tokens == 5
    assert response.prompt_tokens == len("hello world")
    assert response.text == "gen gen gen gen gen"
    assert response.latency_ms >= 0.0


def test_generate_excludes_prompt_tokens_from_completion_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_mps(monkeypatch, built=False, available=False)
    tokenizer = _FakeTokenizer(pad_token_id=0)
    model = _FakeModel()
    _patch_transformers(monkeypatch, tokenizer=tokenizer, model=model)

    backend = LocalTransformersBackend(model_name="fake/model", device="cpu")
    backend.load()

    long_prompt = "x" * 50
    response = backend.generate(GenerationRequest(prompt=long_prompt, max_new_tokens=3))

    assert response.prompt_tokens == 50
    assert response.completion_tokens == 3


def test_generate_reports_device_appropriate_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_mps(monkeypatch, built=True, available=True)
    # `is_available()` is faked as True, but there is no real MPS device in
    # this test process -- the real `torch.mps.synchronize()` would crash,
    # so it must be mocked too whenever a "mps" generate() call is exercised.
    monkeypatch.setattr(torch.mps, "synchronize", lambda: None)
    tokenizer = _FakeTokenizer(pad_token_id=0)
    model = _FakeModel()
    _patch_transformers(monkeypatch, tokenizer=tokenizer, model=model)

    backend = LocalTransformersBackend(model_name="fake/model", device="mps")
    backend.load()
    backend.generate(GenerationRequest(prompt="hi", max_new_tokens=2))

    info = backend.info()
    assert info.device == "mps"
    assert info.dtype == "float16"


def test_close_after_load_resets_health_and_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_mps(monkeypatch, built=False, available=False)
    tokenizer = _FakeTokenizer(pad_token_id=0)
    model = _FakeModel()
    _patch_transformers(monkeypatch, tokenizer=tokenizer, model=model)

    backend = LocalTransformersBackend(model_name="fake/model", device="cpu")
    backend.load()
    assert backend.health() is True

    backend.close()
    assert backend.health() is False
    backend.close()  # idempotent
    assert backend.health() is False

    with pytest.raises(RuntimeError, match="not loaded"):
        backend.generate(GenerationRequest(prompt="hi", max_new_tokens=1))


def test_generation_request_rejects_non_positive_max_new_tokens() -> None:
    with pytest.raises(ValueError, match="max_new_tokens must be > 0"):
        GenerationRequest(prompt="hi", max_new_tokens=0)


def test_mps_synchronize_is_called_on_mps_device(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_mps(monkeypatch, built=True, available=True)
    tokenizer = _FakeTokenizer(pad_token_id=0)
    model = _FakeModel()
    _patch_transformers(monkeypatch, tokenizer=tokenizer, model=model)

    sync_calls = []
    monkeypatch.setattr(torch.mps, "synchronize", lambda: sync_calls.append(1))

    backend = LocalTransformersBackend(model_name="fake/model", device="mps")
    backend.load()
    backend.generate(GenerationRequest(prompt="hi", max_new_tokens=2))

    assert len(sync_calls) == 2  # once before, once after timed generation


def test_mps_synchronize_is_not_called_on_cpu_device(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_mps(monkeypatch, built=False, available=False)
    tokenizer = _FakeTokenizer(pad_token_id=0)
    model = _FakeModel()
    _patch_transformers(monkeypatch, tokenizer=tokenizer, model=model)

    sync_calls = []
    monkeypatch.setattr(torch.mps, "synchronize", lambda: sync_calls.append(1))

    backend = LocalTransformersBackend(model_name="fake/model", device="cpu")
    backend.load()
    backend.generate(GenerationRequest(prompt="hi", max_new_tokens=2))

    assert sync_calls == []
