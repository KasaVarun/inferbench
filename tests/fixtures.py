"""Fixture builders for benchmark_core domain model tests.

These builders construct valid, realistic model instances with sensible
defaults so individual tests only need to override the fields relevant to
what they're checking. Nothing here executes a benchmark -- these are
plain data builders.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from benchmark_core import (
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


def make_workload_configuration(**overrides: Any) -> WorkloadConfiguration:
    """A standard, non-shared-prefix workload."""
    defaults: dict[str, Any] = {
        "name": "standard-chat",
        "prompt_tokens": 256,
        "output_tokens": 128,
        "request_count": 100,
        "shared_prefix_ratio": 0.0,
        "seed": 42,
    }
    defaults.update(overrides)
    return WorkloadConfiguration(**defaults)


def make_shared_prefix_workload(**overrides: Any) -> WorkloadConfiguration:
    """A workload where most requests share a common system-prompt-style prefix."""
    defaults: dict[str, Any] = {
        "name": "shared-prefix-chat",
        "prompt_tokens": 1024,
        "output_tokens": 128,
        "request_count": 200,
        "shared_prefix_ratio": 0.8,
        "seed": 7,
    }
    defaults.update(overrides)
    return WorkloadConfiguration(**defaults)


def make_fp16_configuration(**overrides: Any) -> BenchmarkConfiguration:
    """A standard FP16 configuration, no prefix caching or quantization."""
    defaults: dict[str, Any] = {
        "dtype": "fp16",
        "batch_size": 8,
        "concurrency": 4,
        "prefix_caching": False,
        "quantization": None,
        "gpu_type": "A100",
        "max_model_length": 8192,
    }
    defaults.update(overrides)
    return BenchmarkConfiguration(**defaults)


def make_bf16_configuration(**overrides: Any) -> BenchmarkConfiguration:
    """A BF16 configuration."""
    defaults: dict[str, Any] = {
        "dtype": "bf16",
        "batch_size": 16,
        "concurrency": 8,
        "prefix_caching": False,
        "quantization": None,
        "gpu_type": "H100",
        "max_model_length": 8192,
    }
    defaults.update(overrides)
    return BenchmarkConfiguration(**defaults)


def make_prefix_caching_configuration(**overrides: Any) -> BenchmarkConfiguration:
    """An FP16 configuration with prefix/KV-cache reuse enabled."""
    defaults: dict[str, Any] = {
        "dtype": "fp16",
        "batch_size": 8,
        "concurrency": 4,
        "prefix_caching": True,
        "quantization": None,
        "gpu_type": "H100",
        "max_model_length": 8192,
    }
    defaults.update(overrides)
    return BenchmarkConfiguration(**defaults)


def make_quantized_configuration(**overrides: Any) -> BenchmarkConfiguration:
    """An INT8-quantized configuration."""
    defaults: dict[str, Any] = {
        "dtype": "int8",
        "batch_size": 32,
        "concurrency": 16,
        "prefix_caching": False,
        "quantization": "int8",
        "gpu_type": "L4",
        "max_model_length": 4096,
    }
    defaults.update(overrides)
    return BenchmarkConfiguration(**defaults)


def make_benchmark_request(**overrides: Any) -> BenchmarkRequest:
    """A complete, valid benchmark request using the standard FP16 configuration."""
    defaults: dict[str, Any] = {
        "model_name": "meta-llama/Llama-3-8B-Instruct",
        "backend": "vllm",
        "workload": make_workload_configuration(),
        "configuration": make_fp16_configuration(),
    }
    defaults.update(overrides)
    return BenchmarkRequest(**defaults)


def make_latency_metrics(**overrides: Any) -> LatencyMetrics:
    defaults: dict[str, Any] = {
        "p50_ms": 120.0,
        "p95_ms": 250.0,
        "p99_ms": 400.0,
        "ttft_ms": 45.0,
        "tpot_ms": 12.5,
    }
    defaults.update(overrides)
    return LatencyMetrics(**defaults)


def make_throughput_metrics(**overrides: Any) -> ThroughputMetrics:
    defaults: dict[str, Any] = {
        "requests_per_second": 10.5,
        "input_tokens_per_second": 2048.0,
        "output_tokens_per_second": 1024.0,
        "total_tokens_per_second": 3072.0,
    }
    defaults.update(overrides)
    return ThroughputMetrics(**defaults)


def make_memory_metrics(**overrides: Any) -> MemoryMetrics:
    defaults: dict[str, Any] = {
        "peak_allocated_mb": 40960.0,
        "peak_reserved_mb": 45056.0,
        "gpu_utilization_percent": 87.5,
    }
    defaults.update(overrides)
    return MemoryMetrics(**defaults)


def make_cost_metrics(**overrides: Any) -> CostMetrics:
    defaults: dict[str, Any] = {
        "estimated_benchmark_cost_usd": 4.20,
        "estimated_cost_per_request_usd": 0.042,
        "estimated_cost_per_million_output_tokens_usd": 32.50,
        "tokens_per_dollar": 30_769.0,
    }
    defaults.update(overrides)
    return CostMetrics(**defaults)


def make_gpu_info(**overrides: Any) -> GPUInfo:
    defaults: dict[str, Any] = {
        "gpu_name": "NVIDIA A100-SXM4-80GB",
        "gpu_type": "A100",
        "cuda_version": "12.4",
        "framework_version": "vllm-0.6.3",
    }
    defaults.update(overrides)
    return GPUInfo(**defaults)


def make_execution_metadata(**overrides: Any) -> ExecutionMetadata:
    defaults: dict[str, Any] = {
        "warmup_iterations": 3,
        "measured_iterations": 10,
        "successful_iterations": 10,
        "failed_iterations": 0,
        "total_elapsed_seconds": 0.5,
    }
    defaults.update(overrides)
    return ExecutionMetadata(**defaults)


def make_successful_benchmark_result(**overrides: Any) -> BenchmarkResult:
    """A complete, valid, successful benchmark result."""
    defaults: dict[str, Any] = {
        "created_at": datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
        "request": make_benchmark_request(),
        "latency": make_latency_metrics(),
        "throughput": make_throughput_metrics(),
        "memory": make_memory_metrics(),
        "cost": make_cost_metrics(),
        "gpu": make_gpu_info(),
        "success": True,
        "error": None,
    }
    defaults.update(overrides)
    return BenchmarkResult(**defaults)


def make_failed_benchmark_result(**overrides: Any) -> BenchmarkResult:
    """A complete, valid, failed benchmark result (zeroed-out metrics)."""
    defaults: dict[str, Any] = {
        "created_at": datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
        "request": make_benchmark_request(),
        "latency": make_latency_metrics(
            p50_ms=0.0, p95_ms=0.0, p99_ms=0.0, ttft_ms=None, tpot_ms=None
        ),
        "throughput": make_throughput_metrics(
            requests_per_second=0.0,
            input_tokens_per_second=None,
            output_tokens_per_second=None,
            total_tokens_per_second=None,
        ),
        "memory": MemoryMetrics(),
        "cost": CostMetrics(),
        "gpu": make_gpu_info(),
        "success": False,
        "error": "CUDA out of memory",
    }
    defaults.update(overrides)
    return BenchmarkResult(**defaults)
