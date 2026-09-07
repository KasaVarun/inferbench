"""Typed, CUDA-runtime-independent helpers for the Phase 5 Modal benchmark.

This module deliberately does not import Modal or touch ``torch.cuda``.
The remote Modal function owns CUDA execution; these helpers validate its
configuration and environment metadata, aggregate measured CUDA Event
latencies, and map the measurements into the shared ``BenchmarkResult``.
Consequently, all helpers remain testable on Apple Silicon without Modal
authentication or NVIDIA hardware.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Self

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

from benchmark_core.models import (
    BenchmarkConfiguration,
    BenchmarkRequest,
    BenchmarkResult,
    CostMetrics,
    ExecutionMetadata,
    GPUInfo,
    LatencyMetrics,
    MemoryMetrics,
    ThroughputMetrics,
    WorkloadConfiguration,
)

__all__ = [
    "CudaBenchmarkConfiguration",
    "CudaBenchmarkReport",
    "CudaDtype",
    "CudaEnvironment",
    "ModalGPUType",
    "build_cuda_benchmark_report",
    "build_cuda_failure_result",
    "bytes_to_mib",
    "cuda_correctness_tolerances",
    "latency_percentiles",
    "parse_cuda_dtype",
    "parse_modal_gpu_type",
]

_MATMUL_PROMPT_TOKENS_PLACEHOLDER = 0
_MATMUL_OUTPUT_TOKENS_PLACEHOLDER = 1


class CudaDtype(StrEnum):
    """Numeric dtypes supported by the Phase 5 CUDA matmul benchmark."""

    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"


class ModalGPUType(StrEnum):
    """Modal GPU resource types intentionally supported in Phase 5."""

    A10G = "A10G"
    A100 = "A100"


def parse_cuda_dtype(value: str) -> CudaDtype:
    """Parse a CUDA dtype or raise a concise error listing supported values."""
    try:
        return CudaDtype(value)
    except ValueError as exc:
        supported = ", ".join(dtype.value for dtype in CudaDtype)
        raise ValueError(f"unsupported dtype '{value}'; expected one of: {supported}") from exc


def parse_modal_gpu_type(value: str) -> ModalGPUType:
    """Parse a supported Modal GPU type without silently substituting hardware."""
    try:
        return ModalGPUType(value)
    except ValueError as exc:
        supported = ", ".join(gpu.value for gpu in ModalGPUType)
        raise ValueError(f"unsupported GPU '{value}'; expected one of: {supported}") from exc


def bytes_to_mib(value: int) -> float:
    """Convert bytes to binary mebibytes (MiB), using exactly 1024² bytes."""
    if value < 0:
        raise ValueError(f"byte count must be >= 0, got {value}")
    return value / (1024**2)


def cuda_correctness_tolerances(dtype: CudaDtype) -> tuple[float, float]:
    """Return explicit ``(rtol, atol)`` values for CUDA reference comparison."""
    if dtype is CudaDtype.FLOAT32:
        return 1e-4, 1e-3
    if dtype is CudaDtype.FLOAT16:
        return 1e-2, 5e-1
    return 5e-2, 2.0


def latency_percentiles(latencies_ms: list[float]) -> tuple[float, float, float]:
    """Return p50/p95/p99 for non-empty, non-negative latency measurements."""
    if not latencies_ms:
        raise ValueError("at least one latency measurement is required")
    if any(latency < 0 for latency in latencies_ms):
        raise ValueError("latency measurements must be >= 0")
    p50, p95, p99 = (float(value) for value in np.percentile(latencies_ms, [50, 95, 99]))
    return p50, p95, p99


class CudaBenchmarkConfiguration(BaseModel):
    """Validated configuration passed from the local CLI to the Modal worker."""

    matrix_size: int = Field(default=2048, gt=0)
    iterations: int = Field(default=50, gt=0)
    warmup: int = Field(default=10, ge=0)
    dtype: CudaDtype = CudaDtype.FLOAT16
    seed: int = 42
    gpu_type: ModalGPUType = ModalGPUType.A10G


class CudaEnvironment(BaseModel):
    """CUDA/PyTorch/Python metadata captured from the actual remote container."""

    torch_version: str = Field(min_length=1)
    cuda_version: str = Field(min_length=1)
    cuda_available: bool
    device_count: int = Field(gt=0)
    gpu_name: str = Field(min_length=1)
    compute_capability: tuple[int, int]
    total_gpu_memory_bytes: int = Field(gt=0)
    current_device: int = Field(ge=0)
    python_version: str = Field(min_length=1)

    @field_validator("torch_version", "cuda_version", "gpu_name", "python_version")
    @classmethod
    def _strip_non_empty_strings(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("environment metadata strings must not be empty")
        return stripped

    @model_validator(mode="after")
    def _validate_cuda_device(self) -> Self:
        if not self.cuda_available:
            raise ValueError("a successful CUDA environment must report cuda_available=True")
        if self.current_device >= self.device_count:
            raise ValueError(
                "current_device must be less than device_count "
                f"(got {self.current_device} >= {self.device_count})"
            )
        major, minor = self.compute_capability
        if major < 0 or minor < 0:
            raise ValueError("compute capability components must be >= 0")
        return self


class CudaBenchmarkReport(BaseModel):
    """Typed return value from the one-shot Modal CUDA function.

    ``BenchmarkResult.latency`` contains CUDA Event/device timings. Raw
    per-iteration CUDA and host wall-clock timings remain alongside it so
    callers can compare device execution with host-observed latency without
    conflating the two concepts.
    """

    environment: CudaEnvironment
    correctness_passed: bool
    cuda_latencies_ms: list[float]
    host_latencies_ms: list[float]
    result: BenchmarkResult

    @model_validator(mode="after")
    def _validate_successful_report(self) -> Self:
        iterations = self.result.request.workload.request_count
        if not self.correctness_passed:
            raise ValueError("a successful CUDA report requires correctness_passed=True")
        if not self.result.success:
            raise ValueError("CudaBenchmarkReport requires a successful BenchmarkResult")
        if len(self.cuda_latencies_ms) != iterations:
            raise ValueError("CUDA latency count must equal measured iteration count")
        if len(self.host_latencies_ms) != iterations:
            raise ValueError("host latency count must equal measured iteration count")
        if any(value < 0 for value in self.cuda_latencies_ms + self.host_latencies_ms):
            raise ValueError("all latency measurements must be >= 0")
        return self


def _build_request(config: CudaBenchmarkConfiguration) -> BenchmarkRequest:
    return BenchmarkRequest(
        model_name=f"cuda-matmul-{config.matrix_size}x{config.matrix_size}",
        backend="pytorch-cuda",
        workload=WorkloadConfiguration(
            name=f"cuda-matmul-{config.matrix_size}x{config.matrix_size}",
            prompt_tokens=_MATMUL_PROMPT_TOKENS_PLACEHOLDER,
            output_tokens=_MATMUL_OUTPUT_TOKENS_PLACEHOLDER,
            request_count=config.iterations,
            shared_prefix_ratio=0.0,
            seed=config.seed,
        ),
        configuration=BenchmarkConfiguration(
            dtype=config.dtype.value,
            batch_size=1,
            concurrency=1,
            prefix_caching=False,
            quantization=None,
            gpu_type=config.gpu_type.value,
            max_model_length=None,
        ),
    )


def _gpu_info(config: CudaBenchmarkConfiguration, environment: CudaEnvironment | None) -> GPUInfo:
    if environment is None:
        return GPUInfo(gpu_type=config.gpu_type.value)
    return GPUInfo(
        gpu_name=environment.gpu_name,
        gpu_type=config.gpu_type.value,
        cuda_version=environment.cuda_version,
        framework_version=environment.torch_version,
    )


def build_cuda_benchmark_report(
    *,
    config: CudaBenchmarkConfiguration,
    environment: CudaEnvironment,
    cuda_latencies_ms: list[float],
    host_latencies_ms: list[float],
    measured_wall_seconds: float,
    peak_allocated_bytes: int,
    peak_reserved_bytes: int,
) -> CudaBenchmarkReport:
    """Build a successful report from actual remote CUDA measurements."""
    if len(cuda_latencies_ms) != config.iterations:
        raise ValueError("CUDA latency count must equal configured iterations")
    if len(host_latencies_ms) != config.iterations:
        raise ValueError("host latency count must equal configured iterations")
    if measured_wall_seconds < 0:
        raise ValueError("measured_wall_seconds must be >= 0")
    if peak_reserved_bytes < peak_allocated_bytes:
        raise ValueError("peak reserved CUDA memory must be >= peak allocated CUDA memory")

    p50_ms, p95_ms, p99_ms = latency_percentiles(cuda_latencies_ms)
    total_device_seconds = sum(cuda_latencies_ms) / 1000.0
    operations_per_second = (
        config.iterations / total_device_seconds if total_device_seconds > 0 else 0.0
    )

    result = BenchmarkResult(
        created_at=datetime.now(UTC),
        request=_build_request(config),
        latency=LatencyMetrics(
            p50_ms=p50_ms,
            p95_ms=p95_ms,
            p99_ms=p99_ms,
            ttft_ms=None,
            tpot_ms=None,
        ),
        throughput=ThroughputMetrics(requests_per_second=operations_per_second),
        memory=MemoryMetrics(
            peak_allocated_mb=bytes_to_mib(peak_allocated_bytes),
            peak_reserved_mb=bytes_to_mib(peak_reserved_bytes),
            gpu_utilization_percent=None,
        ),
        cost=CostMetrics(),
        gpu=_gpu_info(config, environment),
        execution=ExecutionMetadata(
            warmup_iterations=config.warmup,
            measured_iterations=config.iterations,
            successful_iterations=config.iterations,
            failed_iterations=0,
            total_elapsed_seconds=measured_wall_seconds,
        ),
        success=True,
        error=None,
    )
    return CudaBenchmarkReport(
        environment=environment,
        correctness_passed=True,
        cuda_latencies_ms=cuda_latencies_ms,
        host_latencies_ms=host_latencies_ms,
        result=result,
    )


def build_cuda_failure_result(
    *,
    config: CudaBenchmarkConfiguration,
    error: str,
    environment: CudaEnvironment | None = None,
) -> BenchmarkResult:
    """Represent a pre-measurement CUDA failure without fabricating performance."""
    if not error.strip():
        raise ValueError("error must not be empty")
    return BenchmarkResult(
        created_at=datetime.now(UTC),
        request=_build_request(config),
        latency=LatencyMetrics(p50_ms=0.0, p95_ms=0.0, p99_ms=0.0),
        throughput=ThroughputMetrics(requests_per_second=0.0),
        memory=MemoryMetrics(),
        cost=CostMetrics(),
        gpu=_gpu_info(config, environment),
        execution=None,
        success=False,
        error=error,
    )
