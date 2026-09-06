"""Tests for `BenchmarkRunner`, `WarmupConfig`, and `MeasurementCollector` (Phase 2).

These tests deliberately avoid asserting exact latency values (timing is
inherently non-deterministic across machines/CI runners). Instead they
assert ordering, non-negativity, iteration counts, and structural
correctness of the produced `BenchmarkResult`.
"""

from __future__ import annotations

import random
from collections.abc import Callable

import numpy as np
import pytest

from benchmark_core import (
    BenchmarkExecutionError,
    BenchmarkRunner,
    MeasurementCollector,
    WarmupConfig,
    make_matmul_workload,
)
from fixtures import make_benchmark_request


class _CountingWorkload:
    """A workload that counts calls and can be configured to fail on demand."""

    def __init__(self, should_fail: Callable[[int], bool] | None = None) -> None:
        self.calls = 0
        self._should_fail = should_fail or (lambda _call_index: False)

    def __call__(self) -> None:
        self.calls += 1
        if self._should_fail(self.calls):
            raise RuntimeError(f"synthetic failure on call {self.calls}")


# ---------------------------------------------------------------------------
# WarmupConfig validation
# ---------------------------------------------------------------------------


def test_warmup_config_accepts_zero_and_positive_iterations() -> None:
    assert WarmupConfig(iterations=0).iterations == 0
    assert WarmupConfig(iterations=5).iterations == 5


def test_warmup_config_rejects_negative_iterations() -> None:
    with pytest.raises(ValueError, match="warmup iterations must be >= 0"):
        WarmupConfig(iterations=-1)


# ---------------------------------------------------------------------------
# BenchmarkRunner construction validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("measured_iterations", [0, -1, -50])
def test_runner_rejects_non_positive_measured_iterations(measured_iterations: int) -> None:
    with pytest.raises(ValueError, match="measured_iterations must be > 0"):
        BenchmarkRunner(warmup=WarmupConfig(iterations=0), measured_iterations=measured_iterations)


def test_runner_concurrency_is_fixed_at_one() -> None:
    runner = BenchmarkRunner(warmup=WarmupConfig(iterations=0), measured_iterations=1)
    assert runner.concurrency == 1


def test_runner_exposes_its_warmup_and_measured_iterations_config() -> None:
    warmup = WarmupConfig(iterations=2)
    runner = BenchmarkRunner(warmup=warmup, measured_iterations=7)
    assert runner.warmup is warmup
    assert runner.measured_iterations == 7


# ---------------------------------------------------------------------------
# MeasurementCollector
# ---------------------------------------------------------------------------


def test_measurement_collector_tracks_success_and_failure_counts() -> None:
    collector = MeasurementCollector()
    collector.record_success(10.0)
    collector.record_success(20.0)
    collector.record_failure("boom")

    assert collector.successful_iterations == 2
    assert collector.failed_iterations == 1
    assert collector.total_iterations == 3
    assert collector.errors == ["boom"]


def test_measurement_collector_percentiles_raises_with_no_successes() -> None:
    collector = MeasurementCollector()
    collector.record_failure("boom")
    with pytest.raises(BenchmarkExecutionError, match="zero successful iterations"):
        collector.percentiles()


def test_measurement_collector_percentiles_match_numpy() -> None:
    collector = MeasurementCollector()
    for value in [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]:
        collector.record_success(value)

    p50, p95, p99 = collector.percentiles()
    expected_p50, expected_p95, expected_p99 = np.percentile(collector.latencies_ms, [50, 95, 99])
    assert p50 == pytest.approx(float(expected_p50))
    assert p95 == pytest.approx(float(expected_p95))
    assert p99 == pytest.approx(float(expected_p99))


def test_measurement_collector_percentile_ordering_holds_for_random_data() -> None:
    rng = random.Random(1234)
    collector = MeasurementCollector()
    for _ in range(500):
        collector.record_success(rng.uniform(0.0, 1000.0))

    p50, p95, p99 = collector.percentiles()
    assert p50 <= p95 <= p99


def test_measurement_collector_percentiles_equal_for_single_sample() -> None:
    collector = MeasurementCollector()
    collector.record_success(42.0)
    p50, p95, p99 = collector.percentiles()
    assert p50 == p95 == p99 == 42.0


# ---------------------------------------------------------------------------
# Warmup exclusion from measured metrics
# ---------------------------------------------------------------------------


def test_warmup_iterations_are_excluded_from_measured_metrics() -> None:
    workload = _CountingWorkload()
    runner = BenchmarkRunner(warmup=WarmupConfig(iterations=3), measured_iterations=5)

    result = runner.run(make_benchmark_request(), workload)

    assert workload.calls == 3 + 5
    assert result.execution is not None
    assert result.execution.warmup_iterations == 3
    assert result.execution.measured_iterations == 5
    assert result.execution.successful_iterations == 5
    assert result.execution.failed_iterations == 0


def test_warmup_with_zero_iterations_runs_only_measured_calls() -> None:
    workload = _CountingWorkload()
    runner = BenchmarkRunner(warmup=WarmupConfig(iterations=0), measured_iterations=4)

    result = runner.run(make_benchmark_request(), workload)

    assert workload.calls == 4
    assert result.execution is not None
    assert result.execution.warmup_iterations == 0


# ---------------------------------------------------------------------------
# Warmup failure
# ---------------------------------------------------------------------------


def test_warmup_failure_raises_before_any_measured_iteration() -> None:
    workload = _CountingWorkload(should_fail=lambda _call_index: True)
    runner = BenchmarkRunner(warmup=WarmupConfig(iterations=3), measured_iterations=5)

    with pytest.raises(BenchmarkExecutionError, match="warmup iteration 1/3 failed"):
        runner.run(make_benchmark_request(), workload)

    # Fails fast on the first warmup call; no further calls (warmup or measured) happen.
    assert workload.calls == 1


# ---------------------------------------------------------------------------
# Partial measured failures
# ---------------------------------------------------------------------------


def test_partial_measured_failures_still_produce_successful_result() -> None:
    # Fail every even-numbered call (2, 4, 6, 8, 10) -> 5 failures, 5 successes.
    workload = _CountingWorkload(should_fail=lambda call_index: call_index % 2 == 0)
    runner = BenchmarkRunner(warmup=WarmupConfig(iterations=0), measured_iterations=10)

    result = runner.run(make_benchmark_request(), workload)

    assert result.success is True
    assert result.error is None
    assert result.execution is not None
    assert result.execution.successful_iterations == 5
    assert result.execution.failed_iterations == 5
    assert result.execution.measured_iterations == 10
    assert result.latency.p50_ms <= result.latency.p95_ms <= result.latency.p99_ms
    assert result.throughput.requests_per_second >= 0.0


# ---------------------------------------------------------------------------
# All measured iterations failing
# ---------------------------------------------------------------------------


def test_all_measured_iterations_failing_produces_failed_result() -> None:
    workload = _CountingWorkload(should_fail=lambda _call_index: True)
    runner = BenchmarkRunner(warmup=WarmupConfig(iterations=0), measured_iterations=5)

    result = runner.run(make_benchmark_request(), workload)

    assert result.success is False
    assert result.error is not None
    assert "5" in result.error
    assert result.execution is not None
    assert result.execution.successful_iterations == 0
    assert result.execution.failed_iterations == 5
    assert result.latency.p50_ms == 0.0
    assert result.throughput.requests_per_second == 0.0


def test_all_measured_iterations_failing_with_identical_error_message() -> None:
    def _always_same_error() -> None:
        raise RuntimeError("boom")

    runner = BenchmarkRunner(warmup=WarmupConfig(iterations=0), measured_iterations=4)
    result = runner.run(make_benchmark_request(), _always_same_error)

    assert result.success is False
    assert result.error is not None
    assert result.error.endswith("boom")


# ---------------------------------------------------------------------------
# No fabricated metrics
# ---------------------------------------------------------------------------


def test_runner_never_fabricates_gpu_memory_or_cost_metrics() -> None:
    workload = _CountingWorkload()
    runner = BenchmarkRunner(warmup=WarmupConfig(iterations=0), measured_iterations=3)

    result = runner.run(make_benchmark_request(), workload)

    assert result.memory.peak_allocated_mb is None
    assert result.memory.peak_reserved_mb is None
    assert result.memory.gpu_utilization_percent is None
    assert result.cost.estimated_benchmark_cost_usd is None
    assert result.gpu.gpu_name is None
    assert result.latency.ttft_ms is None
    assert result.latency.tpot_ms is None
    assert result.throughput.input_tokens_per_second is None


# ---------------------------------------------------------------------------
# End-to-end with the real deterministic matmul workload
# ---------------------------------------------------------------------------


def test_runner_end_to_end_with_matmul_workload_produces_valid_result() -> None:
    workload = make_matmul_workload(matrix_size=8, seed=42)
    runner = BenchmarkRunner(warmup=WarmupConfig(iterations=2), measured_iterations=6)

    result = runner.run(make_benchmark_request(), workload)

    assert result.success is True
    assert result.execution is not None
    assert result.execution.measured_iterations == 6
    assert result.latency.p50_ms <= result.latency.p95_ms <= result.latency.p99_ms
    assert result.latency.p50_ms >= 0.0
    assert result.execution.total_elapsed_seconds >= 0.0
