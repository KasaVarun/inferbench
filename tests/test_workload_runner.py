"""Tests for `benchmark_core.workload_runner.run_workload` (Phase 4).

Uses `FakeBackend` (tests/fake_backend.py) exclusively -- no `torch`, no
`transformers`, no network, no real model, no MPS anywhere in this file.
"""

from __future__ import annotations

import pytest

from benchmark_core import GeneratedWorkload, generate_workload
from benchmark_core.runner import BenchmarkExecutionError
from benchmark_core.workload_runner import _GenerationMeasurementCollector, run_workload
from fake_backend import FakeBackend, make_fake_backend


@pytest.fixture
def workload() -> GeneratedWorkload:
    return generate_workload("short_prompt_short_output", request_count=4, seed=42)


def test_successful_workload_produces_valid_result(workload: GeneratedWorkload) -> None:
    backend = make_fake_backend()
    backend.load()

    result = run_workload(backend=backend, workload=workload, warmup_requests=0)

    assert result.success is True
    assert result.error is None
    assert result.execution is not None
    assert result.execution.measured_iterations == 4
    assert result.execution.successful_iterations == 4
    assert result.execution.failed_iterations == 0
    assert result.latency.p50_ms <= result.latency.p95_ms <= result.latency.p99_ms


def test_request_count_matches_workload(workload: GeneratedWorkload) -> None:
    backend = make_fake_backend()
    backend.load()

    result = run_workload(backend=backend, workload=workload, warmup_requests=0)

    assert result.execution is not None
    assert result.execution.measured_iterations == workload.request_count


def test_warmup_is_excluded_from_measured_statistics(workload: GeneratedWorkload) -> None:
    backend = make_fake_backend()
    backend.load()

    result = run_workload(backend=backend, workload=workload, warmup_requests=2)

    assert result.execution is not None
    assert result.execution.warmup_iterations == 2
    assert result.execution.measured_iterations == workload.request_count
    # warmup calls (2) + measured calls (4) = 6 total generate() calls.
    assert len(backend.calls) == workload.request_count + 2


def test_warmup_failure_raises_and_records_no_measured_calls(workload: GeneratedWorkload) -> None:
    first_prompt = workload.requests[0].prompt
    backend = make_fake_backend(fail_prompts=[first_prompt])
    backend.load()

    with pytest.raises(BenchmarkExecutionError, match="warmup request 1/1 failed"):
        run_workload(backend=backend, workload=workload, warmup_requests=1)

    # Only the failed warmup call happened; no measured requests were made.
    assert backend.calls == [first_prompt]


def test_partial_failures_preserve_failure_counts(workload: GeneratedWorkload) -> None:
    failing_prompt = workload.requests[1].prompt
    backend = make_fake_backend(fail_prompts=[failing_prompt])
    backend.load()

    result = run_workload(backend=backend, workload=workload, warmup_requests=0)

    assert result.success is True
    assert result.execution is not None
    assert result.execution.successful_iterations == 3
    assert result.execution.failed_iterations == 1
    assert result.latency.p50_ms <= result.latency.p95_ms <= result.latency.p99_ms


def test_all_requests_failing_returns_unsuccessful_result(workload: GeneratedWorkload) -> None:
    all_prompts = [r.prompt for r in workload.requests]
    backend = make_fake_backend(fail_prompts=all_prompts)
    backend.load()

    result = run_workload(backend=backend, workload=workload, warmup_requests=0)

    assert result.success is False
    assert result.error is not None
    assert "all 4 measured requests failed" in result.error
    assert result.execution is not None
    assert result.execution.successful_iterations == 0
    assert result.execution.failed_iterations == 4
    assert result.latency.p50_ms == 0.0
    assert result.latency.p95_ms == 0.0
    assert result.latency.p99_ms == 0.0
    assert result.throughput.requests_per_second == 0.0


def test_all_requests_failing_with_identical_error_summarizes_to_one_message(
    workload: GeneratedWorkload,
) -> None:
    backend = FakeBackend(fail_all_with="identical simulated failure")
    backend.load()

    result = run_workload(backend=backend, workload=workload, warmup_requests=0)

    assert result.success is False
    assert result.error is not None
    assert result.error.endswith("identical simulated failure")
    # A single distinct error message must not be reported as "N distinct errors".
    assert "distinct errors" not in result.error


def test_requests_per_second_and_output_tokens_per_second_are_non_negative(
    workload: GeneratedWorkload,
) -> None:
    backend = make_fake_backend()
    backend.load()

    result = run_workload(backend=backend, workload=workload, warmup_requests=0)

    assert result.throughput.requests_per_second >= 0.0
    assert result.throughput.output_tokens_per_second is not None
    assert result.throughput.output_tokens_per_second >= 0.0
    assert result.throughput.input_tokens_per_second is not None
    assert result.throughput.input_tokens_per_second >= 0.0


def test_no_fake_ttft_is_ever_reported(workload: GeneratedWorkload) -> None:
    backend = make_fake_backend()
    backend.load()

    result = run_workload(backend=backend, workload=workload, warmup_requests=0)

    assert result.latency.ttft_ms is None


def test_no_fake_gpu_memory_metrics_are_ever_reported(workload: GeneratedWorkload) -> None:
    backend = make_fake_backend()
    backend.load()

    result = run_workload(backend=backend, workload=workload, warmup_requests=0)

    assert result.memory.peak_allocated_mb is None
    assert result.memory.peak_reserved_mb is None
    assert result.memory.gpu_utilization_percent is None
    assert result.cost.estimated_benchmark_cost_usd is None


def test_tpot_is_computed_from_completed_generation_when_tokens_exist(
    workload: GeneratedWorkload,
) -> None:
    backend = make_fake_backend(latency_ms=100.0)
    backend.load()

    result = run_workload(backend=backend, workload=workload, warmup_requests=0)

    assert result.latency.tpot_ms is not None
    assert result.latency.tpot_ms > 0.0


def test_gpu_info_labels_cpu_device_without_cuda_fields() -> None:
    backend = FakeBackend(device="cpu")
    backend.load()
    workload = generate_workload("short_prompt_short_output", request_count=1, seed=1)

    result = run_workload(backend=backend, workload=workload, warmup_requests=0)

    assert result.gpu.gpu_name == "CPU"
    assert result.gpu.cuda_version is None
    assert result.gpu.gpu_type is None


def test_gpu_info_labels_mps_device_as_apple_mps_not_cuda() -> None:
    backend = FakeBackend(device="mps")
    backend.load()
    workload = generate_workload("short_prompt_short_output", request_count=1, seed=1)

    result = run_workload(backend=backend, workload=workload, warmup_requests=0)

    assert result.gpu.gpu_name == "Apple MPS"
    assert result.gpu.cuda_version is None


def test_mixed_workload_uses_placeholder_token_fields_in_workload_configuration() -> None:
    backend = make_fake_backend()
    backend.load()
    workload = generate_workload("mixed_workload", request_count=4, seed=1)

    result = run_workload(backend=backend, workload=workload, warmup_requests=0)

    assert result.request.workload.prompt_tokens == 0
    assert result.request.workload.output_tokens == 1


def test_negative_warmup_requests_is_rejected(workload: GeneratedWorkload) -> None:
    backend = make_fake_backend()
    backend.load()

    with pytest.raises(ValueError, match="warmup_requests must be >= 0"):
        run_workload(backend=backend, workload=workload, warmup_requests=-1)


# ---------------------------------------------------------------------------
# White-box tests for the internal measurement collector
# ---------------------------------------------------------------------------


def test_collector_percentiles_raises_with_zero_successes() -> None:
    collector = _GenerationMeasurementCollector()
    with pytest.raises(BenchmarkExecutionError, match="zero successful requests"):
        collector.percentiles()


def test_collector_average_ms_per_output_token_is_none_with_zero_completion_tokens() -> None:
    collector = _GenerationMeasurementCollector()
    collector.record_success(latency_ms=5.0, prompt_tokens=3, completion_tokens=0)
    assert collector.average_ms_per_output_token() is None


def test_benchmark_request_records_backend_and_model_metadata(
    workload: GeneratedWorkload,
) -> None:
    backend = FakeBackend(model_name="fake/tiny-model", device="cpu", dtype="float32")
    backend.load()

    result = run_workload(backend=backend, workload=workload, warmup_requests=0)

    assert result.request.backend == "local-transformers"
    assert result.request.model_name == "fake/tiny-model"
    assert result.request.configuration.dtype == "float32"
