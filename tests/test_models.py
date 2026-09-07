"""Tests for the benchmark_core domain model (Phase 1).

These tests only exercise schema/validation behavior of the Pydantic
models. No benchmark is executed and no GPU/inference code is involved.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from benchmark_core import (
    BenchmarkConfiguration,
    BenchmarkRequest,
    BenchmarkResult,
    CostMetrics,
    GPUInfo,
    MemoryMetrics,
    ThroughputMetrics,
    WorkloadConfiguration,
)
from fixtures import (
    make_bf16_configuration,
    make_cost_metrics,
    make_failed_benchmark_result,
    make_fp16_configuration,
    make_gpu_info,
    make_latency_metrics,
    make_memory_metrics,
    make_prefix_caching_configuration,
    make_quantized_configuration,
    make_shared_prefix_workload,
    make_successful_benchmark_result,
    make_throughput_metrics,
    make_workload_configuration,
)

# ---------------------------------------------------------------------------
# Valid model creation
# ---------------------------------------------------------------------------


def test_standard_fp16_configuration_is_valid() -> None:
    config = make_fp16_configuration()
    assert config.dtype == "fp16"
    assert config.prefix_caching is False
    assert config.quantization is None


def test_bf16_configuration_is_valid() -> None:
    config = make_bf16_configuration()
    assert config.dtype == "bf16"


def test_prefix_caching_configuration_is_valid() -> None:
    config = make_prefix_caching_configuration()
    assert config.prefix_caching is True


def test_quantized_configuration_is_valid() -> None:
    config = make_quantized_configuration()
    assert config.quantization == "int8"


def test_shared_prefix_workload_is_valid() -> None:
    workload = make_shared_prefix_workload()
    assert workload.shared_prefix_ratio == 0.8


def test_valid_benchmark_request() -> None:
    request = BenchmarkRequest(
        model_name="meta-llama/Llama-3-8B-Instruct",
        backend="vllm",
        workload=make_workload_configuration(),
        configuration=make_fp16_configuration(),
    )
    assert request.model_name == "meta-llama/Llama-3-8B-Instruct"
    assert isinstance(request.workload, WorkloadConfiguration)
    assert isinstance(request.configuration, BenchmarkConfiguration)


def test_valid_successful_benchmark_result() -> None:
    result = make_successful_benchmark_result()
    assert result.success is True
    assert result.error is None
    assert isinstance(result.run_id, UUID)
    assert isinstance(result.created_at, datetime)


# ---------------------------------------------------------------------------
# 1. model_name / backend / workload name must not be empty
# ---------------------------------------------------------------------------


def test_empty_model_name_is_rejected() -> None:
    with pytest.raises(ValidationError):
        BenchmarkRequest(
            model_name="",
            backend="vllm",
            workload=make_workload_configuration(),
            configuration=make_fp16_configuration(),
        )


def test_whitespace_only_model_name_is_rejected() -> None:
    with pytest.raises(ValidationError):
        BenchmarkRequest(
            model_name="   ",
            backend="vllm",
            workload=make_workload_configuration(),
            configuration=make_fp16_configuration(),
        )


def test_empty_backend_is_rejected() -> None:
    with pytest.raises(ValidationError):
        BenchmarkRequest(
            model_name="llama-3-8b",
            backend="",
            workload=make_workload_configuration(),
            configuration=make_fp16_configuration(),
        )


def test_empty_workload_name_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_workload_configuration(name="")


# ---------------------------------------------------------------------------
# 2. batch_size > 0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [0, -1])
def test_non_positive_batch_size_is_rejected(batch_size: int) -> None:
    with pytest.raises(ValidationError):
        make_fp16_configuration(batch_size=batch_size)


# ---------------------------------------------------------------------------
# 3. concurrency > 0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("concurrency", [0, -5])
def test_non_positive_concurrency_is_rejected(concurrency: int) -> None:
    with pytest.raises(ValidationError):
        make_fp16_configuration(concurrency=concurrency)


# ---------------------------------------------------------------------------
# 4. prompt_tokens >= 0
# ---------------------------------------------------------------------------


def test_negative_prompt_tokens_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_workload_configuration(prompt_tokens=-1)


def test_zero_prompt_tokens_is_allowed() -> None:
    workload = make_workload_configuration(prompt_tokens=0)
    assert workload.prompt_tokens == 0


# ---------------------------------------------------------------------------
# 5. output_tokens > 0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("output_tokens", [0, -1])
def test_non_positive_output_tokens_is_rejected(output_tokens: int) -> None:
    with pytest.raises(ValidationError):
        make_workload_configuration(output_tokens=output_tokens)


# ---------------------------------------------------------------------------
# 6. request_count > 0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("request_count", [0, -1])
def test_non_positive_request_count_is_rejected(request_count: int) -> None:
    with pytest.raises(ValidationError):
        make_workload_configuration(request_count=request_count)


# ---------------------------------------------------------------------------
# 7. shared_prefix_ratio must be between 0.0 and 1.0 inclusive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ratio", [-0.01, 1.01, -1.0, 2.0])
def test_shared_prefix_ratio_out_of_range_is_rejected(ratio: float) -> None:
    with pytest.raises(ValidationError):
        make_workload_configuration(shared_prefix_ratio=ratio)


@pytest.mark.parametrize("ratio", [0.0, 1.0, 0.5])
def test_shared_prefix_ratio_boundary_values_are_allowed(ratio: float) -> None:
    workload = make_workload_configuration(shared_prefix_ratio=ratio)
    assert workload.shared_prefix_ratio == ratio


# ---------------------------------------------------------------------------
# 8. all latency values must be >= 0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    ["p50_ms", "p95_ms", "p99_ms", "ttft_ms", "tpot_ms"],
)
def test_negative_latency_values_are_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        make_latency_metrics(**{field: -1.0})


# ---------------------------------------------------------------------------
# 9. p50 <= p95 <= p99
# ---------------------------------------------------------------------------


def test_percentile_ordering_violation_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_latency_metrics(p50_ms=300.0, p95_ms=200.0, p99_ms=400.0)


def test_percentile_ordering_violation_p95_gt_p99_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_latency_metrics(p50_ms=100.0, p95_ms=500.0, p99_ms=400.0)


def test_percentile_equal_values_are_allowed() -> None:
    metrics = make_latency_metrics(p50_ms=100.0, p95_ms=100.0, p99_ms=100.0)
    assert metrics.p50_ms == metrics.p95_ms == metrics.p99_ms


# ---------------------------------------------------------------------------
# 10. throughput values must be >= 0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "requests_per_second",
        "input_tokens_per_second",
        "output_tokens_per_second",
        "total_tokens_per_second",
        "amortized_output_token_time_ms",
    ],
)
def test_negative_throughput_values_are_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        make_throughput_metrics(**{field: -1.0})


def test_amortized_output_token_time_ms_defaults_to_none() -> None:
    assert make_throughput_metrics().amortized_output_token_time_ms is None


def test_throughput_metrics_accepts_amortized_output_token_time_ms() -> None:
    metrics = make_throughput_metrics(amortized_output_token_time_ms=12.5)
    assert metrics.amortized_output_token_time_ms == 12.5


def test_old_throughput_metrics_payload_without_new_field_still_validates() -> None:
    """A payload written before this field existed must still validate.

    `amortized_output_token_time_ms` was added as an optional, defaulted
    field specifically so that old `BenchmarkResult` JSON (from before
    this field existed) keeps validating unchanged.
    """
    old_payload = {"requests_per_second": 4.5}
    metrics = ThroughputMetrics.model_validate(old_payload)
    assert metrics.amortized_output_token_time_ms is None


# ---------------------------------------------------------------------------
# 11. memory values must be >= 0 when present
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    ["peak_allocated_mb", "peak_reserved_mb", "gpu_utilization_percent"],
)
def test_negative_memory_values_are_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        make_memory_metrics(**{field: -1.0})


def test_memory_metrics_all_none_is_allowed() -> None:
    metrics = MemoryMetrics()
    assert metrics.peak_allocated_mb is None
    assert metrics.peak_reserved_mb is None
    assert metrics.gpu_utilization_percent is None


# ---------------------------------------------------------------------------
# 12. gpu_utilization_percent must be between 0 and 100
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [-0.1, 100.1, 200.0])
def test_gpu_utilization_out_of_range_is_rejected(value: float) -> None:
    with pytest.raises(ValidationError):
        make_memory_metrics(gpu_utilization_percent=value)


@pytest.mark.parametrize("value", [0.0, 100.0, 50.0])
def test_gpu_utilization_boundary_values_are_allowed(value: float) -> None:
    metrics = make_memory_metrics(gpu_utilization_percent=value)
    assert metrics.gpu_utilization_percent == value


# ---------------------------------------------------------------------------
# 13. cost values must be >= 0 when present
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "estimated_benchmark_cost_usd",
        "estimated_cost_per_request_usd",
        "estimated_cost_per_million_output_tokens_usd",
        "tokens_per_dollar",
    ],
)
def test_negative_cost_values_are_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        make_cost_metrics(**{field: -1.0})


def test_cost_metrics_all_none_is_allowed() -> None:
    cost = CostMetrics()
    assert cost.estimated_benchmark_cost_usd is None
    assert cost.estimated_cost_per_request_usd is None
    assert cost.estimated_cost_per_million_output_tokens_usd is None
    assert cost.tokens_per_dollar is None


# ---------------------------------------------------------------------------
# 14. successful result must have error=None
# ---------------------------------------------------------------------------


def test_successful_result_with_error_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_successful_benchmark_result(success=True, error="should not be set")


def test_successful_result_validation_passes_with_no_error() -> None:
    result = make_successful_benchmark_result()
    assert result.success is True
    assert result.error is None


# ---------------------------------------------------------------------------
# 15. failed result must contain a non-empty error message
# ---------------------------------------------------------------------------


def test_failed_result_without_error_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_failed_benchmark_result(success=False, error=None)


def test_failed_result_with_empty_error_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_failed_benchmark_result(success=False, error="   ")


def test_failed_result_validation_passes_with_error_message() -> None:
    result = make_failed_benchmark_result()
    assert result.success is False
    assert result.error == "CUDA out of memory"


# ---------------------------------------------------------------------------
# 16. created_at must be timezone-aware
# ---------------------------------------------------------------------------


def test_naive_created_at_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_successful_benchmark_result(created_at=datetime(2026, 1, 1, 12, 0, 0))


def test_tz_aware_created_at_is_accepted() -> None:
    result = make_successful_benchmark_result(created_at=datetime(2026, 1, 1, tzinfo=UTC))
    assert result.created_at.tzinfo is not None


# ---------------------------------------------------------------------------
# JSON / UUID / datetime round-trip
# ---------------------------------------------------------------------------


def test_benchmark_result_json_round_trip() -> None:
    original = make_successful_benchmark_result()
    payload = original.model_dump_json()
    restored = BenchmarkResult.model_validate_json(payload)
    assert restored == original


def test_benchmark_result_uuid_round_trips() -> None:
    original = make_successful_benchmark_result()
    restored = BenchmarkResult.model_validate_json(original.model_dump_json())
    assert isinstance(restored.run_id, UUID)
    assert restored.run_id == original.run_id


def test_benchmark_result_datetime_round_trips_with_tzinfo() -> None:
    original = make_successful_benchmark_result(
        created_at=datetime(2026, 6, 15, 9, 30, 0, tzinfo=UTC)
    )
    restored = BenchmarkResult.model_validate_json(original.model_dump_json())
    assert restored.created_at == original.created_at


def test_old_benchmark_result_payload_without_amortized_field_still_validates() -> None:
    """A `BenchmarkResult` written before `amortized_output_token_time_ms`
    existed must still validate unchanged -- this is why the field was
    added as optional/defaulted rather than required.
    """
    payload = make_successful_benchmark_result().model_dump(mode="json")
    assert "amortized_output_token_time_ms" in payload["throughput"]
    del payload["throughput"]["amortized_output_token_time_ms"]

    restored = BenchmarkResult.model_validate(payload)

    assert restored.throughput.amortized_output_token_time_ms is None
    assert restored.created_at.tzinfo is not None


def test_benchmark_request_json_round_trip() -> None:
    original = BenchmarkRequest(
        model_name="mistralai/Mixtral-8x7B",
        backend="tgi",
        workload=make_shared_prefix_workload(),
        configuration=make_quantized_configuration(),
    )
    restored = BenchmarkRequest.model_validate_json(original.model_dump_json())
    assert restored == original


# ---------------------------------------------------------------------------
# Optional GPU fields
# ---------------------------------------------------------------------------


def test_gpu_info_all_fields_none_is_valid() -> None:
    gpu = GPUInfo()
    assert gpu.gpu_name is None
    assert gpu.gpu_type is None
    assert gpu.cuda_version is None
    assert gpu.framework_version is None


def test_gpu_info_fully_populated_is_valid() -> None:
    gpu = make_gpu_info()
    assert gpu.gpu_name == "NVIDIA A100-SXM4-80GB"


# ---------------------------------------------------------------------------
# Optional cost fields
# ---------------------------------------------------------------------------


def test_cost_metrics_partially_populated_is_valid() -> None:
    cost = CostMetrics(estimated_benchmark_cost_usd=1.0)
    assert cost.estimated_benchmark_cost_usd == 1.0
    assert cost.estimated_cost_per_request_usd is None


def test_cost_metrics_fully_populated_is_valid() -> None:
    cost = make_cost_metrics()
    assert cost.tokens_per_dollar == 30_769.0
