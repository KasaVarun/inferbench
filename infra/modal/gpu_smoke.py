"""One-shot Modal CUDA matrix-multiplication benchmark for InferBench Phase 5.

Run with:

    modal run infra/modal/gpu_smoke.py

The app exists only for the duration of ``modal run``. It creates no web
endpoint, schedule, volume, secret, or persistent deployment.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import modal

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_CORE_SOURCE = REPOSITORY_ROOT / "packages" / "benchmark-core" / "src"

app = modal.App("inferbench-cuda-smoke")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy>=1.26,<3",
        "pydantic>=2.7,<3",
        "pyyaml>=6.0,<7",
        "torch>=2.2,<3",
    )
    .add_local_dir(BENCHMARK_CORE_SOURCE, "/root/inferbench", copy=True)
    .env({"PYTHONPATH": "/root/inferbench"})
)

_AUTH_HELP = (
    "Modal is not authenticated. Run `.venv/bin/modal setup` "
    "(or `.venv/bin/modal token new`) and retry. Do not commit credentials."
)


def modal_is_authenticated() -> bool:
    """Return whether the local Modal client has credentials configured.

    Reads presence of token or OAuth settings only. Never prints token IDs,
    secrets, or any other credential material.
    """
    from modal.config import config

    token_id = config.get("token_id")
    token_secret = config.get("token_secret")
    if token_id and token_secret:
        return True
    return bool(config.get("oauth_refresh_token"))


@app.function(image=image, gpu="A10G", timeout=600, single_use_containers=True)
def run_cuda_benchmark(config_data: dict[str, Any]) -> dict[str, Any]:
    """Run one correctly timed CUDA matmul benchmark on the requested GPU."""
    import platform

    import torch

    from benchmark_core.cuda_benchmark import (
        CudaBenchmarkConfiguration,
        CudaEnvironment,
        build_cuda_benchmark_report,
        cuda_correctness_tolerances,
    )

    config = CudaBenchmarkConfiguration.model_validate(config_data)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the Modal GPU container")

    device_count = torch.cuda.device_count()
    if device_count < 1:
        raise RuntimeError("CUDA reported available but no CUDA devices were found")

    torch.cuda.set_device(0)
    current_device = torch.cuda.current_device()
    cuda_version = torch.version.cuda
    if cuda_version is None:
        raise RuntimeError("PyTorch reports CUDA available but torch.version.cuda is None")

    properties = torch.cuda.get_device_properties(current_device)
    environment = CudaEnvironment(
        torch_version=torch.__version__,
        cuda_version=cuda_version,
        cuda_available=True,
        device_count=device_count,
        gpu_name=torch.cuda.get_device_name(current_device),
        compute_capability=torch.cuda.get_device_capability(current_device),
        total_gpu_memory_bytes=properties.total_memory,
        current_device=current_device,
        python_version=platform.python_version(),
    )

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[config.dtype.value]
    if config.dtype.value == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"bfloat16 was requested but is unsupported by {environment.gpu_name}")

    # Generate deterministic inputs on CPU, then transfer/cast before any
    # timing. Neither allocation nor host-to-device transfer contaminates
    # the measured matmul latency.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    left_cpu = torch.randn(
        (config.matrix_size, config.matrix_size),
        dtype=torch.float32,
        generator=generator,
    )
    right_cpu = torch.randn(
        (config.matrix_size, config.matrix_size),
        dtype=torch.float32,
        generator=generator,
    )
    reference = torch.matmul(left_cpu, right_cpu)
    left = left_cpu.to(device="cuda", dtype=torch_dtype)
    right = right_cpu.to(device="cuda", dtype=torch_dtype)

    # Correctness runs before performance measurement. This also creates
    # the CUDA context and initializes library kernels, keeping that startup
    # work out of steady-state timings.
    candidate = torch.matmul(left, right)
    torch.cuda.synchronize()
    candidate_cpu = candidate.float().cpu()
    rtol, atol = cuda_correctness_tolerances(config.dtype)
    correctness_passed = bool(torch.allclose(candidate_cpu, reference, rtol=rtol, atol=atol))
    del candidate, candidate_cpu, reference, left_cpu, right_cpu
    if not correctness_passed:
        del left, right
        torch.cuda.empty_cache()
        raise RuntimeError(
            f"CUDA matmul correctness check failed for dtype={config.dtype.value} "
            f"with rtol={rtol} and atol={atol}"
        )

    # Warmup is never measured. It absorbs one-time context, allocator,
    # cuBLAS, and kernel-selection effects before the measured sequence.
    for _ in range(config.warmup):
        warmup_output = torch.matmul(left, right)
    if config.warmup:
        del warmup_output
    torch.cuda.synchronize()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cuda_latencies_ms: list[float] = []
    host_latencies_ms: list[float] = []

    torch.cuda.synchronize()
    measured_wall_start = time.perf_counter()
    for _ in range(config.iterations):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        host_start = time.perf_counter()
        start_event.record()
        output = torch.matmul(left, right)
        end_event.record()
        end_event.synchronize()
        host_end = time.perf_counter()

        cuda_latencies_ms.append(float(start_event.elapsed_time(end_event)))
        host_latencies_ms.append((host_end - host_start) * 1000.0)
        del output
    torch.cuda.synchronize()
    measured_wall_seconds = time.perf_counter() - measured_wall_start

    peak_allocated_bytes = torch.cuda.max_memory_allocated()
    peak_reserved_bytes = torch.cuda.max_memory_reserved()

    del left, right
    torch.cuda.empty_cache()

    report = build_cuda_benchmark_report(
        config=config,
        environment=environment,
        cuda_latencies_ms=cuda_latencies_ms,
        host_latencies_ms=host_latencies_ms,
        measured_wall_seconds=measured_wall_seconds,
        peak_allocated_bytes=peak_allocated_bytes,
        peak_reserved_bytes=peak_reserved_bytes,
    )
    return report.model_dump(mode="json")


@app.local_entrypoint()
def main(
    gpu: str = "A10G",
    matrix_size: int = 2048,
    iterations: int = 50,
    warmup: int = 10,
    dtype: str = "float16",
    seed: int = 42,
    output: str | None = None,
) -> None:
    """Validate arguments locally, invoke one GPU job, and print its result."""
    from modal.exception import AuthError

    from benchmark_core.atomic_io import write_atomic
    from benchmark_core.cuda_benchmark import (
        CudaBenchmarkConfiguration,
        CudaBenchmarkReport,
        parse_cuda_dtype,
        parse_modal_gpu_type,
    )

    if not modal_is_authenticated():
        raise RuntimeError(_AUTH_HELP)

    config = CudaBenchmarkConfiguration(
        matrix_size=matrix_size,
        iterations=iterations,
        warmup=warmup,
        dtype=parse_cuda_dtype(dtype),
        seed=seed,
        gpu_type=parse_modal_gpu_type(gpu),
    )
    try:
        report_data = run_cuda_benchmark.with_options(gpu=config.gpu_type.value).remote(
            config.model_dump(mode="json")
        )
    except AuthError as exc:
        raise RuntimeError(_AUTH_HELP) from exc
    report = CudaBenchmarkReport.model_validate(report_data)

    payload = (report.model_dump_json(indent=2) + "\n").encode()
    if output is not None:
        output_path = Path(output)
        write_atomic(output_path, payload)
        print(f"Wrote validated CUDA benchmark report to {output_path}")

    result = report.result
    print(
        f"CUDA smoke succeeded on {result.gpu.gpu_name} "
        f"({result.gpu.gpu_type}, CUDA {result.gpu.cuda_version}, "
        f"PyTorch {result.gpu.framework_version})"
    )
    print(
        f"CUDA latency p50/p95/p99: {result.latency.p50_ms:.3f}/"
        f"{result.latency.p95_ms:.3f}/{result.latency.p99_ms:.3f} ms"
    )
    print(
        f"Peak allocated/reserved: {result.memory.peak_allocated_mb:.2f}/"
        f"{result.memory.peak_reserved_mb:.2f} MiB"
    )
    if output is None:
        print(json.dumps(report.model_dump(mode="json"), indent=2))
