"""Local benchmark execution engine (Phase 2).

``BenchmarkRunner`` executes an arbitrary zero-argument callable ("the
workload") a configured number of times, times each measured call with a
high-resolution monotonic clock, and assembles the outcome into a
validated Phase 1 ``BenchmarkResult``.

This module knows nothing about GPUs, CUDA, or any specific workload; it
only knows how to run a callable repeatedly and honestly report what
happened. Concurrency is fixed at 1 in this phase: iterations run strictly
sequentially on the calling thread/process.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from benchmark_core.models import (
    BenchmarkRequest,
    BenchmarkResult,
    CostMetrics,
    ExecutionMetadata,
    GPUInfo,
    LatencyMetrics,
    MemoryMetrics,
    ThroughputMetrics,
)

__all__ = [
    "BenchmarkExecutionError",
    "BenchmarkRunner",
    "MeasurementCollector",
    "WarmupConfig",
]


class BenchmarkExecutionError(RuntimeError):
    """Raised when a benchmark run cannot proceed at all.

    Currently only raised when a warmup iteration fails: warmup existing
    is a precondition for trustworthy measurement, so a warmup failure
    aborts the run before any measured iteration executes, rather than
    being folded into the measured failure count.
    """


@dataclass(frozen=True, slots=True)
class WarmupConfig:
    """Configuration for the warmup phase that precedes measurement.

    Warmup iterations execute the workload exactly like measured
    iterations, but their timings are never included in reported latency
    or throughput statistics -- they exist only to let caches, JITs, and
    allocators reach a steady state first.
    """

    iterations: int = 0

    def __post_init__(self) -> None:
        if self.iterations < 0:
            raise ValueError(f"warmup iterations must be >= 0, got {self.iterations}")


@dataclass(slots=True)
class MeasurementCollector:
    """Collects per-iteration outcomes for the measured phase of a run.

    Every measured iteration is recorded exactly once, as either a success
    (with its latency in milliseconds) or a failure (with its error
    message). Failures are never dropped: `failed_iterations` always
    reflects every failure recorded here.
    """

    latencies_ms: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def record_success(self, latency_ms: float) -> None:
        self.latencies_ms.append(latency_ms)

    def record_failure(self, error: str) -> None:
        self.errors.append(error)

    @property
    def successful_iterations(self) -> int:
        return len(self.latencies_ms)

    @property
    def failed_iterations(self) -> int:
        return len(self.errors)

    @property
    def total_iterations(self) -> int:
        return self.successful_iterations + self.failed_iterations

    def percentiles(self) -> tuple[float, float, float]:
        """Return ``(p50_ms, p95_ms, p99_ms)`` from recorded successes.

        ``numpy.percentile`` is monotonically non-decreasing in the
        requested percentile for any fixed dataset, so the returned tuple
        always satisfies ``p50 <= p95 <= p99``.

        Raises:
            BenchmarkExecutionError: If no successful iterations were
                recorded. Callers should check `successful_iterations > 0`
                before calling this.
        """
        if not self.latencies_ms:
            raise BenchmarkExecutionError(
                "cannot compute latency percentiles with zero successful iterations"
            )
        p50, p95, p99 = (float(value) for value in np.percentile(self.latencies_ms, [50, 95, 99]))
        return p50, p95, p99


class BenchmarkRunner:
    """Runs a workload through a warmup phase then a measured phase.

    The runner is workload-agnostic: it only requires a zero-argument
    callable that either returns normally (success) or raises (failure).
    It fabricates nothing -- latency/throughput are computed from actual
    timings, and memory/cost/GPU metrics are always left as ``None``
    unless a caller supplies them, since this runner does not measure
    them.
    """

    def __init__(self, *, warmup: WarmupConfig, measured_iterations: int) -> None:
        if measured_iterations <= 0:
            raise ValueError(f"measured_iterations must be > 0, got {measured_iterations}")
        self._warmup = warmup
        self._measured_iterations = measured_iterations

    @property
    def warmup(self) -> WarmupConfig:
        return self._warmup

    @property
    def measured_iterations(self) -> int:
        return self._measured_iterations

    @property
    def concurrency(self) -> int:
        """Fixed at 1: this phase only supports strictly sequential execution."""
        return 1

    def run(self, request: BenchmarkRequest, workload: Callable[[], None]) -> BenchmarkResult:
        """Execute `workload`'s warmup then measured phases and build a result.

        Raises:
            BenchmarkExecutionError: If a warmup iteration fails. No
                measured iterations run in that case.
        """
        self._run_warmup(workload)
        collector = MeasurementCollector()

        start = time.perf_counter()
        for _ in range(self._measured_iterations):
            iteration_start = time.perf_counter()
            try:
                workload()
            except Exception as exc:
                # Intentionally broad: any workload failure is recorded as a
                # measured failure below, never silently discarded.
                collector.record_failure(str(exc) or type(exc).__name__)
                continue
            iteration_end = time.perf_counter()
            collector.record_success((iteration_end - iteration_start) * 1000.0)
        elapsed_seconds = time.perf_counter() - start

        return self._build_result(request, collector, elapsed_seconds)

    def _run_warmup(self, workload: Callable[[], None]) -> None:
        for iteration in range(1, self._warmup.iterations + 1):
            try:
                workload()
            except Exception as exc:
                raise BenchmarkExecutionError(
                    f"warmup iteration {iteration}/{self._warmup.iterations} failed: {exc}"
                ) from exc

    def _build_result(
        self,
        request: BenchmarkRequest,
        collector: MeasurementCollector,
        elapsed_seconds: float,
    ) -> BenchmarkResult:
        execution = ExecutionMetadata(
            warmup_iterations=self._warmup.iterations,
            measured_iterations=collector.total_iterations,
            successful_iterations=collector.successful_iterations,
            failed_iterations=collector.failed_iterations,
            total_elapsed_seconds=elapsed_seconds,
        )
        created_at = datetime.now(UTC)
        empty_downstream_metrics = {
            "memory": MemoryMetrics(),
            "cost": CostMetrics(),
            "gpu": GPUInfo(),
        }

        if collector.successful_iterations == 0:
            return BenchmarkResult(
                created_at=created_at,
                request=request,
                latency=LatencyMetrics(p50_ms=0.0, p95_ms=0.0, p99_ms=0.0),
                throughput=ThroughputMetrics(requests_per_second=0.0),
                execution=execution,
                success=False,
                error=(
                    f"all {collector.total_iterations} measured iterations failed: "
                    f"{self._summarize_errors(collector.errors)}"
                ),
                **empty_downstream_metrics,
            )

        p50_ms, p95_ms, p99_ms = collector.percentiles()
        requests_per_second = (
            collector.successful_iterations / elapsed_seconds if elapsed_seconds > 0 else 0.0
        )

        return BenchmarkResult(
            created_at=created_at,
            request=request,
            latency=LatencyMetrics(p50_ms=p50_ms, p95_ms=p95_ms, p99_ms=p99_ms),
            throughput=ThroughputMetrics(requests_per_second=requests_per_second),
            execution=execution,
            success=True,
            error=None,
            **empty_downstream_metrics,
        )

    @staticmethod
    def _summarize_errors(errors: Sequence[str]) -> str:
        # Only called when every measured iteration failed, and
        # `measured_iterations > 0` is enforced at construction time, so
        # `errors` is always non-empty here.
        distinct = sorted(set(errors))
        if len(distinct) == 1:
            return distinct[0]
        return f"{len(distinct)} distinct errors, first: {distinct[0]}"
