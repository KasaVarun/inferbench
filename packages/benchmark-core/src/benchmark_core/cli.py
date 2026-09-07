"""Command-line interface for benchmark_core (Phase 2 + Phase 3).

Phase 2 command shape::

    python -m benchmark_core benchmark-cpu \\
        --iterations 100 \\
        --warmup 10 \\
        --matrix-size 256 \\
        --seed 42 \\
        --output benchmarks/results/cpu_baseline.json

Phase 3 command shape::

    python -m benchmark_core generate-workload \\
        --profile shared_prefix \\
        --requests 100 \\
        --seed 42 \\
        --output benchmarks/workloads/shared_prefix.json

`generate-workload` writes a deterministic `GeneratedWorkload` as JSON or
YAML (selected by `--output`'s extension) using
`benchmark_core.workload_generation.generate_workload`.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from benchmark_core.atomic_io import write_atomic
from benchmark_core.models import BenchmarkConfiguration, BenchmarkRequest, WorkloadConfiguration
from benchmark_core.runner import BenchmarkExecutionError, BenchmarkRunner, WarmupConfig
from benchmark_core.serialization import SUPPORTED_EXTENSIONS, workload_to_bytes
from benchmark_core.workload_generation import SUPPORTED_PROFILES, generate_workload
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


def _unit_interval_float(raw: str) -> float:
    value = float(raw)
    if not (0.0 <= value <= 1.0):
        raise argparse.ArgumentTypeError(f"must be within [0.0, 1.0], got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark_core",
        description="InferBench local benchmark harness and workload generator.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_benchmark_cpu_parser(subparsers)
    _add_generate_workload_parser(subparsers)

    return parser


def _add_benchmark_cpu_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
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


def _add_generate_workload_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    workload_parser = subparsers.add_parser(
        "generate-workload",
        help="Generate a deterministic synthetic LLM workload artifact (JSON or YAML).",
    )
    workload_parser.add_argument(
        "--profile", choices=SUPPORTED_PROFILES, required=True, help="Workload profile to generate."
    )
    workload_parser.add_argument(
        "--requests", type=_positive_int, required=True, help="Number of requests (> 0)."
    )
    workload_parser.add_argument("--seed", type=int, required=True, help="Deterministic RNG seed.")
    workload_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=f"Output path; extension selects format ({', '.join(SUPPORTED_EXTENSIONS)}).",
    )
    workload_parser.add_argument(
        "--input-tokens",
        type=_positive_int,
        default=None,
        help="Override the target input tokens (> 0). Not meaningful for mixed_workload.",
    )
    workload_parser.add_argument(
        "--output-tokens",
        type=_positive_int,
        default=None,
        help="Override the requested output tokens (> 0). Not meaningful for mixed_workload.",
    )
    workload_parser.add_argument(
        "--shared-prefix-ratio",
        type=_unit_interval_float,
        default=None,
        help="Override the fraction of requests sharing an exact prefix ([0.0, 1.0]). "
        "Only meaningful for the shared_prefix profile.",
    )


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
    payload = (result.model_dump_json(indent=2) + "\n").encode("utf-8")
    write_atomic(output_path, payload)

    status = "success" if result.success else "failure"
    print(f"benchmark-cpu: {status}, wrote result to {output_path}")
    return 0 if result.success else 1


def _run_generate_workload(args: argparse.Namespace) -> int:
    output_path: Path = args.output
    if output_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print(
            f"generate-workload: unsupported output extension '{output_path.suffix}'; "
            f"expected one of {SUPPORTED_EXTENSIONS}",
            file=sys.stderr,
        )
        return 1

    try:
        workload = generate_workload(
            profile=args.profile,
            request_count=args.requests,
            seed=args.seed,
            target_input_tokens=args.input_tokens,
            requested_output_tokens=args.output_tokens,
            shared_prefix_ratio=args.shared_prefix_ratio,
        )
    except ValueError as exc:
        print(f"generate-workload: {exc}", file=sys.stderr)
        return 1

    payload = workload_to_bytes(workload, output_path.suffix)
    write_atomic(output_path, payload)

    print(f"generate-workload: wrote {len(workload.requests)} requests to {output_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "benchmark-cpu":
        return _run_benchmark_cpu(args)
    return _run_generate_workload(args)
