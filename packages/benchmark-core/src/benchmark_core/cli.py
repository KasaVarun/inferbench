"""Command-line interface for benchmark_core (Phase 2).

Exact command shape::

    python -m benchmark_core benchmark-cpu \\
        --iterations 100 \\
        --warmup 10 \\
        --matrix-size 256 \\
        --seed 42 \\
        --output benchmarks/results/cpu_baseline.json

This runs the deterministic NumPy matmul workload through
`benchmark_core.runner.BenchmarkRunner` and atomically writes the
resulting `BenchmarkResult` as JSON to `--output`.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from benchmark_core.models import BenchmarkConfiguration, BenchmarkRequest, WorkloadConfiguration
from benchmark_core.runner import BenchmarkExecutionError, BenchmarkRunner, WarmupConfig
from benchmark_core.workloads import make_matmul_workload

__all__ = ["build_parser", "main"]

# WorkloadConfiguration (Phase 1) was designed for LLM-shaped workloads: it
# requires `output_tokens > 0` and treats `prompt_tokens`/`output_tokens` as
# token counts. The CPU matmul workload here has no tokens at all, so these
# fields are populated with documented placeholders rather than fabricated
# token counts. `name` and `seed` carry the real, meaningful values for this
# workload (matrix dimensions and RNG seed); the token fields exist only to
# satisfy the shared schema and must not be interpreted as LLM token counts.
_CPU_WORKLOAD_PROMPT_TOKENS_PLACEHOLDER = 0
_CPU_WORKLOAD_OUTPUT_TOKENS_PLACEHOLDER = 1


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {value}")
    return value


def _non_negative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark_core",
        description="InferBench Phase 2: local CPU benchmark harness.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    cpu_parser = subparsers.add_parser(
        "benchmark-cpu",
        help="Run the deterministic CPU NumPy matmul benchmark.",
    )
    cpu_parser.add_argument(
        "--iterations", type=_positive_int, required=True, help="Measured iterations (> 0)."
    )
    cpu_parser.add_argument(
        "--warmup", type=_non_negative_int, required=True, help="Warmup iterations (>= 0)."
    )
    cpu_parser.add_argument(
        "--matrix-size", type=_positive_int, required=True, help="Square matrix side length (> 0)."
    )
    cpu_parser.add_argument("--seed", type=int, required=True, help="Deterministic RNG seed.")
    cpu_parser.add_argument(
        "--output", type=Path, required=True, help="Path to write the BenchmarkResult JSON to."
    )

    return parser


def _build_cpu_benchmark_request(
    *, matrix_size: int, seed: int, iterations: int
) -> BenchmarkRequest:
    workload = WorkloadConfiguration(
        name=f"cpu-matmul-{matrix_size}x{matrix_size}",
        prompt_tokens=_CPU_WORKLOAD_PROMPT_TOKENS_PLACEHOLDER,
        output_tokens=_CPU_WORKLOAD_OUTPUT_TOKENS_PLACEHOLDER,
        request_count=iterations,
        shared_prefix_ratio=0.0,
        seed=seed,
    )
    configuration = BenchmarkConfiguration(
        dtype="float32",
        batch_size=1,
        concurrency=1,
        prefix_caching=False,
        quantization=None,
        gpu_type=None,
        max_model_length=None,
    )
    return BenchmarkRequest(
        model_name="cpu-matmul-workload",
        backend="numpy",
        workload=workload,
        configuration=configuration,
    )


def _write_json_atomic(path: Path, payload: str) -> None:
    """Write `payload` to `path` atomically, never corrupting an existing file.

    A temporary file in the same directory is written and fsynced fully
    before an atomic `os.replace` swaps it into place. If anything fails
    before the replace, the temporary file is removed and `path` is left
    completely untouched -- so a failed write can never leave a partial or
    corrupted result behind, nor overwrite a previously valid one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            tmp_file.write(payload)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(tmp_name, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def _run_benchmark_cpu(args: argparse.Namespace) -> int:
    request = _build_cpu_benchmark_request(
        matrix_size=args.matrix_size, seed=args.seed, iterations=args.iterations
    )
    workload = make_matmul_workload(matrix_size=args.matrix_size, seed=args.seed)
    runner = BenchmarkRunner(
        warmup=WarmupConfig(iterations=args.warmup), measured_iterations=args.iterations
    )

    try:
        result = runner.run(request, workload)
    except BenchmarkExecutionError as exc:
        print(f"benchmark-cpu: warmup failed, no output written: {exc}", file=sys.stderr)
        return 1

    output_path: Path = args.output
    _write_json_atomic(output_path, result.model_dump_json(indent=2))

    status = "success" if result.success else "failure"
    print(f"benchmark-cpu: {status}, wrote result to {output_path}")
    return 0 if result.success else 1


def main(argv: Sequence[str] | None = None) -> int:
    # `benchmark-cpu` is the only registered subcommand and `dest="command"`
    # is `required=True`, so argparse itself rejects any invocation that
    # doesn't resolve to it before `parse_args` returns.
    parser = build_parser()
    args = parser.parse_args(argv)
    return _run_benchmark_cpu(args)
