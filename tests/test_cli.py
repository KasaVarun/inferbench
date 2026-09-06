"""Tests for the `benchmark_core` CLI (Phase 2).

The CLI smoke test invokes `benchmark_core.cli.main` in-process (not via a
subprocess) so it stays fast and hermetic -- no network and no GPU are
touched anywhere in this module.
"""

from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from benchmark_core import BenchmarkResult
from benchmark_core.cli import main


def _base_args(output: Path, **overrides: str) -> list[str]:
    args = {
        "--iterations": "20",
        "--warmup": "3",
        "--matrix-size": "16",
        "--seed": "42",
        "--output": str(output),
    }
    args.update(overrides)
    result: list[str] = ["benchmark-cpu"]
    for key, value in args.items():
        result.extend([key, value])
    return result


# ---------------------------------------------------------------------------
# CLI validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("iterations", ["0", "-1"])
def test_cli_rejects_non_positive_iterations(tmp_path: Path, iterations: str) -> None:
    output = tmp_path / "result.json"
    with pytest.raises(SystemExit):
        main(_base_args(output, **{"--iterations": iterations}))
    assert not output.exists()


@pytest.mark.parametrize("warmup", ["-1", "-5"])
def test_cli_rejects_negative_warmup(tmp_path: Path, warmup: str) -> None:
    output = tmp_path / "result.json"
    with pytest.raises(SystemExit):
        main(_base_args(output, **{"--warmup": warmup}))
    assert not output.exists()


@pytest.mark.parametrize("matrix_size", ["0", "-4"])
def test_cli_rejects_non_positive_matrix_size(tmp_path: Path, matrix_size: str) -> None:
    output = tmp_path / "result.json"
    with pytest.raises(SystemExit):
        main(_base_args(output, **{"--matrix-size": matrix_size}))
    assert not output.exists()


def test_cli_accepts_zero_warmup() -> None:
    # Zero warmup is explicitly valid (>= 0), unlike iterations/matrix-size (> 0).
    from benchmark_core.cli import _non_negative_int

    assert _non_negative_int("0") == 0


# ---------------------------------------------------------------------------
# Output directory creation + atomic write + JSON validity
# ---------------------------------------------------------------------------


def test_cli_creates_missing_output_parent_directories(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "dirs" / "result.json"
    assert not output.parent.exists()

    exit_code = main(_base_args(output))

    assert exit_code == 0
    assert output.exists()


def test_cli_smoke_run_produces_valid_benchmark_result(tmp_path: Path) -> None:
    output = tmp_path / "cpu_smoke.json"

    exit_code = main(_base_args(output))

    assert exit_code == 0
    assert output.exists()

    payload = output.read_text(encoding="utf-8")
    result = BenchmarkResult.model_validate_json(payload)

    assert result.success is True
    assert result.request.backend == "numpy"
    assert result.execution is not None
    assert result.execution.measured_iterations == 20
    assert result.execution.warmup_iterations == 3
    assert result.latency.p50_ms <= result.latency.p95_ms <= result.latency.p99_ms

    # Double-check it's also valid, well-formed JSON on disk (not just via pydantic).
    parsed = json.loads(payload)
    assert parsed["success"] is True


def test_cli_write_leaves_no_leftover_temp_files(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    main(_base_args(output))

    leftovers = list(tmp_path.glob(".*.tmp"))
    assert leftovers == []


def test_cli_existing_output_not_corrupted_on_simulated_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "result.json"
    original_content = '{"already": "here"}'
    output.write_text(original_content, encoding="utf-8")

    def _failing_fsync(_fd: int) -> None:
        raise OSError("simulated disk failure during fsync")

    monkeypatch.setattr(os, "fsync", _failing_fsync)

    with pytest.raises(OSError, match="simulated disk failure"):
        main(_base_args(output))

    # The pre-existing file must be completely untouched.
    assert output.read_text(encoding="utf-8") == original_content
    # No stray temp file should be left behind either.
    leftovers = list(tmp_path.glob(".*.tmp"))
    assert leftovers == []


def test_cli_reports_warmup_failure_and_writes_no_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "result.json"

    def _always_raise(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("warmup boom")

    monkeypatch.setattr("benchmark_core.cli.make_matmul_workload", lambda **_kwargs: _always_raise)

    exit_code = main(_base_args(output, **{"--warmup": "2"}))

    assert exit_code == 1
    assert not output.exists()
    captured = capsys.readouterr()
    assert "warmup failed, no output written" in captured.err


def test_main_module_entrypoint_runs_via_runpy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Executes `benchmark_core/__main__.py`'s guard in-process (coverage-visible)."""
    output = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["benchmark_core", *_base_args(output)])

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("benchmark_core", run_name="__main__")

    assert exc_info.value.code == 0
    assert output.exists()


def test_cli_real_subprocess_invocation_matches_documented_command(tmp_path: Path) -> None:
    """Exercises the literal documented command, including `python -m benchmark_core`."""
    output = tmp_path / "cpu_smoke.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmark_core",
            "benchmark-cpu",
            "--iterations",
            "5",
            "--warmup",
            "1",
            "--matrix-size",
            "8",
            "--seed",
            "42",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    assert output.exists()
    BenchmarkResult.model_validate_json(output.read_text(encoding="utf-8"))


def test_write_json_atomic_directly(tmp_path: Path) -> None:
    from benchmark_core.cli import _write_json_atomic

    target = tmp_path / "sub" / "out.json"
    _write_json_atomic(target, '{"a": 1}')

    assert target.read_text(encoding="utf-8") == '{"a": 1}'
    assert list(tmp_path.glob("**/.*.tmp")) == []
