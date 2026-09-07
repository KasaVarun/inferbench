"""Local-only tests for Phase 5 CUDA benchmark models and aggregation.

These tests use synthetic metadata and measured values. They never import
Modal, call ``torch.cuda``, contact a network, or require NVIDIA hardware.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from pydantic import ValidationError

from benchmark_core.cuda_benchmark import (
    CudaBenchmarkConfiguration,
    CudaBenchmarkReport,
    CudaDtype,
    CudaEnvironment,
    ModalGPUType,
    build_cuda_benchmark_report,
    build_cuda_failure_result,
    bytes_to_mib,
    cuda_correctness_tolerances,
    latency_percentiles,
    parse_cuda_dtype,
    parse_modal_gpu_type,
)


@pytest.fixture
def config() -> CudaBenchmarkConfiguration:
    return CudaBenchmarkConfiguration(
        matrix_size=1024,
        iterations=3,
        warmup=2,
        dtype=CudaDtype.FLOAT16,
        seed=42,
        gpu_type=ModalGPUType.A10G,
    )


@pytest.fixture
def environment() -> CudaEnvironment:
    return CudaEnvironment(
        torch_version="2.8.0+cu128",
        cuda_version="12.8",
        cuda_available=True,
        device_count=1,
        gpu_name="NVIDIA A10G",
        compute_capability=(8, 6),
        total_gpu_memory_bytes=24 * 1024**3,
        current_device=0,
        python_version="3.12.11",
    )


def test_bytes_to_mib_uses_binary_mebibytes() -> None:
    assert bytes_to_mib(0) == 0.0
    assert bytes_to_mib(1024**2) == 1.0
    assert bytes_to_mib(5 * 1024**2 + 512 * 1024) == 5.5


def test_bytes_to_mib_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="must be >= 0"):
        bytes_to_mib(-1)


@pytest.mark.parametrize("value", ["float32", "float16", "bfloat16"])
def test_parse_cuda_dtype_accepts_supported_values(value: str) -> None:
    assert parse_cuda_dtype(value).value == value


def test_parse_cuda_dtype_rejects_unsupported_value() -> None:
    with pytest.raises(ValueError, match="unsupported dtype 'float64'"):
        parse_cuda_dtype("float64")


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (CudaDtype.FLOAT32, (1e-4, 1e-3)),
        (CudaDtype.FLOAT16, (1e-2, 5e-1)),
        (CudaDtype.BFLOAT16, (5e-2, 2.0)),
    ],
)
def test_cuda_correctness_tolerances_are_explicit(
    dtype: CudaDtype, expected: tuple[float, float]
) -> None:
    tolerances = cuda_correctness_tolerances(dtype)
    assert tolerances == expected
    assert tolerances[0] > 0
    assert tolerances[1] > 0


@pytest.mark.parametrize("value", ["A10G", "A100"])
def test_parse_modal_gpu_type_accepts_supported_values(value: str) -> None:
    assert parse_modal_gpu_type(value).value == value


def test_parse_modal_gpu_type_rejects_unsupported_value() -> None:
    with pytest.raises(ValueError, match="unsupported GPU 'H100'"):
        parse_modal_gpu_type("H100")


def test_cuda_benchmark_configuration_defaults() -> None:
    parsed = CudaBenchmarkConfiguration()
    assert parsed.matrix_size == 2048
    assert parsed.iterations == 50
    assert parsed.warmup == 10
    assert parsed.dtype is CudaDtype.FLOAT16
    assert parsed.gpu_type is ModalGPUType.A10G


@pytest.mark.parametrize("matrix_size", [0, -1])
def test_configuration_rejects_invalid_matrix_size(matrix_size: int) -> None:
    with pytest.raises(ValidationError):
        CudaBenchmarkConfiguration(matrix_size=matrix_size)


@pytest.mark.parametrize("iterations", [0, -1])
def test_configuration_rejects_invalid_iteration_count(iterations: int) -> None:
    with pytest.raises(ValidationError):
        CudaBenchmarkConfiguration(iterations=iterations)


def test_configuration_rejects_invalid_warmup_count() -> None:
    with pytest.raises(ValidationError):
        CudaBenchmarkConfiguration(warmup=-1)


def test_configuration_rejects_unsupported_dtype() -> None:
    with pytest.raises(ValidationError):
        CudaBenchmarkConfiguration(dtype="float64")


def test_latency_percentiles_are_ordered() -> None:
    p50, p95, p99 = latency_percentiles([3.0, 1.0, 5.0, 2.0, 4.0])
    assert p50 <= p95 <= p99
    assert p50 == 3.0


def test_latency_percentiles_reject_empty_measurements() -> None:
    with pytest.raises(ValueError, match="at least one"):
        latency_percentiles([])


def test_latency_percentiles_reject_negative_measurement() -> None:
    with pytest.raises(ValueError, match="must be >= 0"):
        latency_percentiles([1.0, -1.0])


def test_environment_rejects_unavailable_cuda() -> None:
    with pytest.raises(ValidationError, match="cuda_available=True"):
        CudaEnvironment(
            torch_version="2.8.0",
            cuda_version="12.8",
            cuda_available=False,
            device_count=1,
            gpu_name="NVIDIA A10G",
            compute_capability=(8, 6),
            total_gpu_memory_bytes=24 * 1024**3,
            current_device=0,
            python_version="3.12",
        )


def test_environment_rejects_whitespace_only_metadata(
    environment: CudaEnvironment,
) -> None:
    payload = environment.model_dump()
    payload["gpu_name"] = " "
    with pytest.raises(ValidationError, match="must not be empty"):
        CudaEnvironment.model_validate(payload)


def test_environment_rejects_current_device_outside_device_count(
    environment: CudaEnvironment,
) -> None:
    payload = environment.model_dump()
    payload["current_device"] = 1
    with pytest.raises(ValidationError, match="less than device_count"):
        CudaEnvironment.model_validate(payload)


def test_environment_rejects_negative_compute_capability(
    environment: CudaEnvironment,
) -> None:
    payload = environment.model_dump()
    payload["compute_capability"] = (-1, 0)
    with pytest.raises(ValidationError, match="compute capability"):
        CudaEnvironment.model_validate(payload)


def test_successful_report_maps_benchmark_result_fields(
    config: CudaBenchmarkConfiguration, environment: CudaEnvironment
) -> None:
    report = build_cuda_benchmark_report(
        config=config,
        environment=environment,
        cuda_latencies_ms=[3.0, 1.0, 2.0],
        host_latencies_ms=[3.5, 1.5, 2.5],
        measured_wall_seconds=0.008,
        peak_allocated_bytes=100 * 1024**2,
        peak_reserved_bytes=128 * 1024**2,
    )

    result = report.result
    assert result.success is True
    assert result.request.backend == "pytorch-cuda"
    assert result.request.model_name == "cuda-matmul-1024x1024"
    assert result.request.configuration.dtype == "float16"
    assert result.request.configuration.gpu_type == "A10G"
    assert result.request.configuration.batch_size == 1
    assert result.request.configuration.concurrency == 1
    assert result.request.configuration.prefix_caching is False
    assert result.request.configuration.quantization is None
    assert result.request.workload.prompt_tokens == 0
    assert result.request.workload.output_tokens == 1


def test_successful_report_maps_gpu_metadata(
    config: CudaBenchmarkConfiguration, environment: CudaEnvironment
) -> None:
    report = build_cuda_benchmark_report(
        config=config,
        environment=environment,
        cuda_latencies_ms=[1.0, 2.0, 3.0],
        host_latencies_ms=[1.5, 2.5, 3.5],
        measured_wall_seconds=0.01,
        peak_allocated_bytes=1,
        peak_reserved_bytes=2,
    )
    assert report.result.gpu.gpu_name == "NVIDIA A10G"
    assert report.result.gpu.gpu_type == "A10G"
    assert report.result.gpu.cuda_version == "12.8"
    assert report.result.gpu.framework_version == "2.8.0+cu128"


def test_successful_report_maps_memory_metrics_in_mib(
    config: CudaBenchmarkConfiguration, environment: CudaEnvironment
) -> None:
    report = build_cuda_benchmark_report(
        config=config,
        environment=environment,
        cuda_latencies_ms=[1.0, 2.0, 3.0],
        host_latencies_ms=[1.5, 2.5, 3.5],
        measured_wall_seconds=0.01,
        peak_allocated_bytes=100 * 1024**2,
        peak_reserved_bytes=128 * 1024**2,
    )
    assert report.result.memory.peak_allocated_mb == 100.0
    assert report.result.memory.peak_reserved_mb == 128.0
    assert report.result.memory.gpu_utilization_percent is None


def test_successful_report_uses_cuda_event_time_for_operations_per_second(
    config: CudaBenchmarkConfiguration, environment: CudaEnvironment
) -> None:
    report = build_cuda_benchmark_report(
        config=config,
        environment=environment,
        cuda_latencies_ms=[1.0, 2.0, 3.0],
        host_latencies_ms=[10.0, 20.0, 30.0],
        measured_wall_seconds=0.1,
        peak_allocated_bytes=1,
        peak_reserved_bytes=2,
    )
    assert report.result.throughput.requests_per_second == pytest.approx(500.0)
    assert report.result.throughput.input_tokens_per_second is None
    assert report.result.throughput.output_tokens_per_second is None
    assert report.result.throughput.total_tokens_per_second is None


def test_successful_report_preserves_host_and_cuda_iteration_timings(
    config: CudaBenchmarkConfiguration, environment: CudaEnvironment
) -> None:
    cuda_latencies = [1.0, 2.0, 3.0]
    host_latencies = [1.5, 2.5, 3.5]
    report = build_cuda_benchmark_report(
        config=config,
        environment=environment,
        cuda_latencies_ms=cuda_latencies,
        host_latencies_ms=host_latencies,
        measured_wall_seconds=0.01,
        peak_allocated_bytes=1,
        peak_reserved_bytes=2,
    )
    assert report.cuda_latencies_ms == cuda_latencies
    assert report.host_latencies_ms == host_latencies
    assert report.result.latency.p50_ms <= report.result.latency.p95_ms
    assert report.result.latency.p95_ms <= report.result.latency.p99_ms


def test_successful_report_maps_execution_counts(
    config: CudaBenchmarkConfiguration, environment: CudaEnvironment
) -> None:
    report = build_cuda_benchmark_report(
        config=config,
        environment=environment,
        cuda_latencies_ms=[1.0, 2.0, 3.0],
        host_latencies_ms=[1.5, 2.5, 3.5],
        measured_wall_seconds=0.01,
        peak_allocated_bytes=1,
        peak_reserved_bytes=2,
    )
    assert report.result.execution is not None
    assert report.result.execution.warmup_iterations == 2
    assert report.result.execution.measured_iterations == 3
    assert report.result.execution.successful_iterations == 3
    assert report.result.execution.failed_iterations == 0
    assert report.result.execution.total_elapsed_seconds == 0.01


def test_successful_report_does_not_fabricate_llm_or_cost_metrics(
    config: CudaBenchmarkConfiguration, environment: CudaEnvironment
) -> None:
    report = build_cuda_benchmark_report(
        config=config,
        environment=environment,
        cuda_latencies_ms=[1.0, 2.0, 3.0],
        host_latencies_ms=[1.5, 2.5, 3.5],
        measured_wall_seconds=0.01,
        peak_allocated_bytes=1,
        peak_reserved_bytes=2,
    )
    assert report.result.latency.ttft_ms is None
    assert report.result.latency.tpot_ms is None
    assert report.result.throughput.amortized_output_token_time_ms is None
    assert report.result.cost.estimated_benchmark_cost_usd is None


@pytest.mark.parametrize(
    ("cuda_latencies", "host_latencies"),
    [([1.0, 2.0], [1.0, 2.0, 3.0]), ([1.0, 2.0, 3.0], [1.0, 2.0])],
)
def test_report_builder_rejects_latency_count_mismatch(
    config: CudaBenchmarkConfiguration,
    environment: CudaEnvironment,
    cuda_latencies: list[float],
    host_latencies: list[float],
) -> None:
    with pytest.raises(ValueError, match="latency count"):
        build_cuda_benchmark_report(
            config=config,
            environment=environment,
            cuda_latencies_ms=cuda_latencies,
            host_latencies_ms=host_latencies,
            measured_wall_seconds=0.01,
            peak_allocated_bytes=1,
            peak_reserved_bytes=2,
        )


def test_report_builder_rejects_negative_wall_time(
    config: CudaBenchmarkConfiguration, environment: CudaEnvironment
) -> None:
    with pytest.raises(ValueError, match="measured_wall_seconds"):
        build_cuda_benchmark_report(
            config=config,
            environment=environment,
            cuda_latencies_ms=[1.0, 2.0, 3.0],
            host_latencies_ms=[1.0, 2.0, 3.0],
            measured_wall_seconds=-1.0,
            peak_allocated_bytes=1,
            peak_reserved_bytes=2,
        )


def test_report_builder_rejects_reserved_memory_below_allocated(
    config: CudaBenchmarkConfiguration, environment: CudaEnvironment
) -> None:
    with pytest.raises(ValueError, match="reserved CUDA memory"):
        build_cuda_benchmark_report(
            config=config,
            environment=environment,
            cuda_latencies_ms=[1.0, 2.0, 3.0],
            host_latencies_ms=[1.0, 2.0, 3.0],
            measured_wall_seconds=0.01,
            peak_allocated_bytes=2,
            peak_reserved_bytes=1,
        )


def test_failure_result_contains_error_without_fake_measurements(
    config: CudaBenchmarkConfiguration,
) -> None:
    result = build_cuda_failure_result(config=config, error="CUDA unavailable")
    assert result.success is False
    assert result.error == "CUDA unavailable"
    assert result.execution is None
    assert result.latency.p50_ms == 0.0
    assert result.throughput.requests_per_second == 0.0
    assert result.memory.peak_allocated_mb is None
    assert result.gpu.gpu_type == "A10G"
    assert result.gpu.gpu_name is None


def test_failure_result_can_include_real_environment_metadata(
    config: CudaBenchmarkConfiguration, environment: CudaEnvironment
) -> None:
    result = build_cuda_failure_result(
        config=config,
        error="correctness failed",
        environment=environment,
    )
    assert result.gpu.gpu_name == "NVIDIA A10G"
    assert result.gpu.cuda_version == "12.8"


def test_failure_result_rejects_empty_error(config: CudaBenchmarkConfiguration) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        build_cuda_failure_result(config=config, error=" ")


def test_report_json_round_trip(
    config: CudaBenchmarkConfiguration, environment: CudaEnvironment
) -> None:
    original = build_cuda_benchmark_report(
        config=config,
        environment=environment,
        cuda_latencies_ms=[1.0, 2.0, 3.0],
        host_latencies_ms=[1.5, 2.5, 3.5],
        measured_wall_seconds=0.01,
        peak_allocated_bytes=1,
        peak_reserved_bytes=2,
    )
    restored = CudaBenchmarkReport.model_validate_json(original.model_dump_json())
    assert restored == original
    assert json.loads(original.model_dump_json())["result"]["success"] is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("correctness_passed", False, "correctness_passed=True"),
        ("result.success", False, "successful BenchmarkResult"),
        ("cuda_latencies_ms", [1.0], "CUDA latency count"),
        ("host_latencies_ms", [1.0], "host latency count"),
        ("cuda_latencies_ms", [1.0, -1.0, 2.0], "must be >= 0"),
    ],
)
def test_report_model_rejects_inconsistent_success_payload(
    config: CudaBenchmarkConfiguration,
    environment: CudaEnvironment,
    field: str,
    value: object,
    message: str,
) -> None:
    report = build_cuda_benchmark_report(
        config=config,
        environment=environment,
        cuda_latencies_ms=[1.0, 2.0, 3.0],
        host_latencies_ms=[1.5, 2.5, 3.5],
        measured_wall_seconds=0.01,
        peak_allocated_bytes=1,
        peak_reserved_bytes=2,
    )
    payload = report.model_dump(mode="json")
    if field == "result.success":
        payload["result"]["success"] = value
        payload["result"]["error"] = "simulated failure"
    else:
        payload[field] = value
    with pytest.raises(ValidationError, match=message):
        CudaBenchmarkReport.model_validate(payload)


def test_importing_cuda_benchmark_does_not_import_torch_or_modal() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import benchmark_core.cuda_benchmark; "
                "print('torch' in sys.modules); print('modal' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = completed.stdout.strip().splitlines()
    assert lines == ["False", "False"]


def test_importing_benchmark_core_does_not_import_modal() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import benchmark_core; print('modal' in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "False"


def test_lazy_package_exports_preserve_local_backend_public_api() -> None:
    import benchmark_core

    assert benchmark_core.LocalTransformersBackend.__name__ == "LocalTransformersBackend"
    assert benchmark_core.resolve_device("cpu") == "cpu"
    assert callable(benchmark_core.is_mps_available)
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = benchmark_core.not_a_real_export
