"""Tests for the `run-local` CLI command (Phase 4).

No test here downloads a model or reaches huggingface.co: most tests
monkeypatch `benchmark_core.cli.LocalTransformersBackend` with the fake
backend from `tests/fake_backend.py`; the one MPS-unavailability test uses
the real class but mocks `torch.backends.mps` so no real hardware or
network is ever touched.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch

from benchmark_core import BenchmarkResult, generate_workload, write_workload
from benchmark_core.cli import main
from fake_backend import FakeBackend


def _run_local_args(model: str, workload: Path, output: Path, **overrides: str) -> list[str]:
    args = {
        "--model": model,
        "--workload": str(workload),
        "--warmup": "0",
        "--output": str(output),
    }
    args.update(overrides)
    result: list[str] = ["run-local"]
    for key, value in args.items():
        result.extend([key, value])
    return result


def _patch_fake_backend(
    monkeypatch: pytest.MonkeyPatch, backend: FakeBackend
) -> Callable[..., FakeBackend]:
    def _factory(*_args: Any, **_kwargs: Any) -> FakeBackend:
        return backend

    monkeypatch.setattr("benchmark_core.cli.LocalTransformersBackend", _factory)
    return _factory


@pytest.fixture
def json_workload_path(tmp_path: Path) -> Path:
    workload = generate_workload("short_prompt_short_output", request_count=3, seed=42)
    path = tmp_path / "workload.json"
    write_workload(workload, path)
    return path


@pytest.fixture
def yaml_workload_path(tmp_path: Path) -> Path:
    workload = generate_workload("short_prompt_short_output", request_count=3, seed=42)
    path = tmp_path / "workload.yaml"
    write_workload(workload, path)
    return path


# ---------------------------------------------------------------------------
# Successful runs (JSON and YAML workload input)
# ---------------------------------------------------------------------------


def test_run_local_with_json_workload_succeeds(
    tmp_path: Path, json_workload_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_fake_backend(monkeypatch, FakeBackend())
    output = tmp_path / "result.json"

    exit_code = main(_run_local_args("fake/tiny-model", json_workload_path, output))

    assert exit_code == 0
    result = BenchmarkResult.model_validate_json(output.read_text(encoding="utf-8"))
    assert result.success is True
    assert result.request.backend == "local-transformers"


def test_run_local_with_yaml_workload_succeeds(
    tmp_path: Path, yaml_workload_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_fake_backend(monkeypatch, FakeBackend())
    output = tmp_path / "result.json"

    exit_code = main(_run_local_args("fake/tiny-model", yaml_workload_path, output))

    assert exit_code == 0
    result = BenchmarkResult.model_validate_json(output.read_text(encoding="utf-8"))
    assert result.success is True


# ---------------------------------------------------------------------------
# Workload loading failures
# ---------------------------------------------------------------------------


def test_run_local_missing_workload_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_fake_backend(monkeypatch, FakeBackend())
    output = tmp_path / "result.json"
    missing = tmp_path / "does_not_exist.json"

    exit_code = main(_run_local_args("fake/tiny-model", missing, output))

    assert exit_code == 1
    assert not output.exists()
    captured = capsys.readouterr()
    assert "failed to load workload" in captured.err


def test_run_local_malformed_workload_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_fake_backend(monkeypatch, FakeBackend())
    output = tmp_path / "result.json"
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{ not valid json", encoding="utf-8")

    exit_code = main(_run_local_args("fake/tiny-model", malformed, output))

    assert exit_code == 1
    assert not output.exists()
    captured = capsys.readouterr()
    assert "failed to load workload" in captured.err


def test_run_local_unsupported_workload_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_fake_backend(monkeypatch, FakeBackend())
    output = tmp_path / "result.json"
    bad_extension = tmp_path / "workload.txt"
    bad_extension.write_text("irrelevant", encoding="utf-8")

    exit_code = main(_run_local_args("fake/tiny-model", bad_extension, output))

    assert exit_code == 1
    assert not output.exists()


# ---------------------------------------------------------------------------
# CLI argument validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("warmup", ["-1", "-5"])
def test_run_local_rejects_negative_warmup(
    tmp_path: Path, json_workload_path: Path, monkeypatch: pytest.MonkeyPatch, warmup: str
) -> None:
    _patch_fake_backend(monkeypatch, FakeBackend())
    output = tmp_path / "result.json"

    with pytest.raises(SystemExit):
        main(_run_local_args("fake/tiny-model", json_workload_path, output, **{"--warmup": warmup}))
    assert not output.exists()


def test_run_local_rejects_invalid_device_choice(
    tmp_path: Path, json_workload_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_fake_backend(monkeypatch, FakeBackend())
    output = tmp_path / "result.json"

    with pytest.raises(SystemExit):
        main(_run_local_args("fake/tiny-model", json_workload_path, output, **{"--device": "cuda"}))
    assert not output.exists()


# ---------------------------------------------------------------------------
# Explicit MPS unavailability (real LocalTransformersBackend, mocked torch)
# ---------------------------------------------------------------------------


def test_run_local_explicit_mps_fails_clearly_when_unavailable(
    tmp_path: Path,
    json_workload_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    output = tmp_path / "result.json"

    exit_code = main(
        _run_local_args("fake/tiny-model", json_workload_path, output, **{"--device": "mps"})
    )

    assert exit_code == 1
    assert not output.exists()
    captured = capsys.readouterr()
    assert "model/tokenizer load failed" in captured.err
    assert "MPS is not available" in captured.err


# ---------------------------------------------------------------------------
# Model load / warmup failure surfacing
# ---------------------------------------------------------------------------


def test_run_local_reports_model_load_failure(
    tmp_path: Path,
    json_workload_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_fake_backend(monkeypatch, FakeBackend(load_should_fail=True))
    output = tmp_path / "result.json"

    exit_code = main(_run_local_args("fake/tiny-model", json_workload_path, output))

    assert exit_code == 1
    assert not output.exists()
    captured = capsys.readouterr()
    assert "model/tokenizer load failed" in captured.err


def test_run_local_reports_warmup_failure_and_writes_nothing(
    tmp_path: Path,
    json_workload_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workload = generate_workload("short_prompt_short_output", request_count=3, seed=42)
    backend = FakeBackend(fail_prompts=frozenset([workload.requests[0].prompt]))
    _patch_fake_backend(monkeypatch, backend)
    output = tmp_path / "result.json"

    exit_code = main(
        _run_local_args("fake/tiny-model", json_workload_path, output, **{"--warmup": "1"})
    )

    assert exit_code == 1
    assert not output.exists()
    captured = capsys.readouterr()
    assert "warmup failed, no output written" in captured.err


def test_run_local_backend_is_closed_after_model_load_failure(
    tmp_path: Path, json_workload_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = FakeBackend(load_should_fail=True)
    _patch_fake_backend(monkeypatch, backend)
    output = tmp_path / "result.json"

    main(_run_local_args("fake/tiny-model", json_workload_path, output))

    assert backend.health() is False


def test_run_local_backend_is_closed_after_success(
    tmp_path: Path, json_workload_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = FakeBackend()
    _patch_fake_backend(monkeypatch, backend)
    output = tmp_path / "result.json"

    main(_run_local_args("fake/tiny-model", json_workload_path, output))

    assert backend.health() is False


# ---------------------------------------------------------------------------
# Output directory creation + atomic writing + failure safety
# ---------------------------------------------------------------------------


def test_run_local_creates_missing_output_parent_directories(
    tmp_path: Path, json_workload_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_fake_backend(monkeypatch, FakeBackend())
    output = tmp_path / "nested" / "dirs" / "result.json"
    assert not output.parent.exists()

    exit_code = main(_run_local_args("fake/tiny-model", json_workload_path, output))

    assert exit_code == 0
    assert output.exists()


def test_run_local_all_requests_failing_reports_nonzero_exit_but_writes_result(
    tmp_path: Path, json_workload_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workload = generate_workload("short_prompt_short_output", request_count=3, seed=42)
    backend = FakeBackend(fail_prompts=frozenset(r.prompt for r in workload.requests))
    _patch_fake_backend(monkeypatch, backend)
    output = tmp_path / "result.json"

    exit_code = main(_run_local_args("fake/tiny-model", json_workload_path, output))

    assert exit_code == 1
    assert output.exists()
    result = BenchmarkResult.model_validate_json(output.read_text(encoding="utf-8"))
    assert result.success is False


def test_run_local_existing_output_preserved_on_workload_load_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_fake_backend(monkeypatch, FakeBackend())
    output = tmp_path / "result.json"
    original_content = '{"already": "here"}'
    output.write_text(original_content, encoding="utf-8")
    missing = tmp_path / "does_not_exist.json"

    exit_code = main(_run_local_args("fake/tiny-model", missing, output))

    assert exit_code == 1
    assert output.read_text(encoding="utf-8") == original_content
