"""Executes a `GeneratedWorkload` against an `InferenceBackend` (Phase 4).

This module is backend-agnostic: it only knows the `InferenceBackend`
contract, never `torch` or `transformers` directly, so it can be (and is,
in tests) exercised against a fake backend with zero network access and
no real model. Concurrency is fixed at 1, matching Phase 2's runner:
requests execute strictly sequentially.

The caller (typically the CLI) owns the backend's lifecycle: this module
assumes `backend.load()` has already succeeded before `run_workload` is
called, and never calls `close()` itself.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from benchmark_core.inference_backend import GenerationRequest, InferenceBackend
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
from benchmark_core.runner import BenchmarkExecutionError
from benchmark_core.workload_generation import GeneratedWorkload

__all__ = ["run_workload"]

# Placeholders for WorkloadConfiguration fields that mixed_workload leaves
# unset (it has no single input/output target -- see Phase 3 docs). Mirrors
# the documented placeholder pattern already used for the Phase 2 CPU
# workload, which also doesn't map cleanly onto LLM-shaped token fields.
_MIXED_WORKLOAD_PROMPT_TOKENS_PLACEHOLDER = 0
_MIXED_WORKLOAD_OUTPUT_TOKENS_PLACEHOLDER = 1


@dataclass(slots=True)
class _GenerationMeasurementCollector:
    """Collects per-request outcomes for the measured phase of a workload run.

    Every measured request is recorded exactly once, as either a success
    (with its latency and real tokenizer-derived token counts) or a
    failure (with its error message). Failures are never dropped.
    """

    latencies_ms: list[float] = field(default_factory=list)
    prompt_token_counts: list[int] = field(default_factory=list)
    completion_token_counts: list[int] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def record_success(
        self, *, latency_ms: float, prompt_tokens: int, completion_tokens: int
    ) -> None:
        self.latencies_ms.append(latency_ms)
        self.prompt_token_counts.append(prompt_tokens)
        self.completion_token_counts.append(completion_tokens)

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

    @property
    def total_prompt_tokens(self) -> int:
        return sum(self.prompt_token_counts)

    @property
    def total_completion_tokens(self) -> int:
        return sum(self.completion_token_counts)

    def percentiles(self) -> tuple[float, float, float]:
        """Return ``(p50_ms, p95_ms, p99_ms)`` from recorded successes.

        Raises:
            BenchmarkExecutionError: If no successful requests were
                recorded. Callers should check `successful_iterations > 0`
                before calling this.
        """
        if not self.latencies_ms:
            raise BenchmarkExecutionError(
                "cannot compute latency percentiles with zero successful requests"
            )
        p50, p95, p99 = (float(v) for v in np.percentile(self.latencies_ms, [50, 95, 99]))
        return p50, p95, p99

    def amortized_output_token_time_ms(self) -> float | None:
        """A coarse aggregate -- explicitly **not** true decode-only TPOT.

        Computed as (sum of successful requests' end-to-end latency in ms)
        / (total output tokens across those requests). This intentionally
        includes each request's prompt-processing ("prefill") time and
        full-request generation overhead amortized across its output
        tokens -- it is **not** a true incremental decode-only
        measurement, which would require per-token timestamps that a
        single `model.generate()` call does not provide. This value is
        reported as `ThroughputMetrics.amortized_output_token_time_ms`,
        never as `LatencyMetrics.tpot_ms` (which stays `None` here; see
        `docs/local-inference.md`). Returns `None` if no successful
        request produced any output tokens (nothing defensible to
        compute).
        """
        total_completion_tokens = self.total_completion_tokens
        if total_completion_tokens <= 0:
            return None
        return sum(self.latencies_ms) / total_completion_tokens


def _run_warmup(
    backend: InferenceBackend, warmup_requests: int, sample_request: GenerationRequest
) -> None:
    for iteration in range(1, warmup_requests + 1):
        try:
            backend.generate(sample_request)
        except Exception as exc:
            raise BenchmarkExecutionError(
                f"warmup request {iteration}/{warmup_requests} failed: {exc}"
            ) from exc


def _build_benchmark_request(
    workload: GeneratedWorkload, backend: InferenceBackend
) -> BenchmarkRequest:
    info = backend.info()
    generator_configuration = workload.generator_configuration

    workload_configuration = WorkloadConfiguration(
        name=workload.profile,
        prompt_tokens=(
            generator_configuration.target_input_tokens
            if generator_configuration.target_input_tokens is not None
            else _MIXED_WORKLOAD_PROMPT_TOKENS_PLACEHOLDER
        ),
        output_tokens=(
            generator_configuration.requested_output_tokens
            if generator_configuration.requested_output_tokens is not None
            else _MIXED_WORKLOAD_OUTPUT_TOKENS_PLACEHOLDER
        ),
        request_count=workload.request_count,
        shared_prefix_ratio=generator_configuration.shared_prefix_ratio,
        seed=workload.seed,
    )
    benchmark_configuration = BenchmarkConfiguration(
        dtype=info.dtype,
        batch_size=1,
        concurrency=1,
        prefix_caching=False,
        quantization=None,
        gpu_type=None,
        max_model_length=None,
    )
    return BenchmarkRequest(
        model_name=info.model_name,
        backend="local-transformers",
        workload=workload_configuration,
        configuration=benchmark_configuration,
    )


def _build_gpu_info(backend: InferenceBackend) -> GPUInfo:
    info = backend.info()
    # Apple MPS is not CUDA/NVIDIA hardware; label it plainly rather than
    # misrepresenting it as a CUDA device. `cuda_version` is always None
    # here -- there is no CUDA anywhere in this backend.
    gpu_name = "Apple MPS" if info.device == "mps" else "CPU"
    return GPUInfo(
        gpu_name=gpu_name,
        gpu_type=None,
        cuda_version=None,
        framework_version=info.framework_version,
    )


def _build_result(
    *,
    request: BenchmarkRequest,
    gpu: GPUInfo,
    collector: _GenerationMeasurementCollector,
    warmup_requests: int,
    elapsed_seconds: float,
) -> BenchmarkResult:
    created_at = datetime.now(UTC)
    execution = ExecutionMetadata(
        warmup_iterations=warmup_requests,
        measured_iterations=collector.total_iterations,
        successful_iterations=collector.successful_iterations,
        failed_iterations=collector.failed_iterations,
        total_elapsed_seconds=elapsed_seconds,
    )
    empty_downstream_metrics = {"memory": MemoryMetrics(), "cost": CostMetrics(), "gpu": gpu}

    if collector.successful_iterations == 0:
        return BenchmarkResult(
            created_at=created_at,
            request=request,
            latency=LatencyMetrics(p50_ms=0.0, p95_ms=0.0, p99_ms=0.0),
            throughput=ThroughputMetrics(requests_per_second=0.0),
            execution=execution,
            success=False,
            error=(
                f"all {collector.total_iterations} measured requests failed: "
                f"{_summarize_errors(collector.errors)}"
            ),
            **empty_downstream_metrics,
        )

    p50_ms, p95_ms, p99_ms = collector.percentiles()
    requests_per_second = (
        collector.successful_iterations / elapsed_seconds if elapsed_seconds > 0 else 0.0
    )
    input_tokens_per_second = (
        collector.total_prompt_tokens / elapsed_seconds if elapsed_seconds > 0 else 0.0
    )
    output_tokens_per_second = (
        collector.total_completion_tokens / elapsed_seconds if elapsed_seconds > 0 else 0.0
    )

    return BenchmarkResult(
        created_at=created_at,
        request=request,
        latency=LatencyMetrics(
            p50_ms=p50_ms,
            p95_ms=p95_ms,
            p99_ms=p99_ms,
            ttft_ms=None,  # No first-token streaming instrumentation exists yet.
            tpot_ms=None,  # No true decode-only per-token timing exists yet.
        ),
        throughput=ThroughputMetrics(
            requests_per_second=requests_per_second,
            input_tokens_per_second=input_tokens_per_second,
            output_tokens_per_second=output_tokens_per_second,
            total_tokens_per_second=input_tokens_per_second + output_tokens_per_second,
            amortized_output_token_time_ms=collector.amortized_output_token_time_ms(),
        ),
        execution=execution,
        success=True,
        error=None,
        **empty_downstream_metrics,
    )


def _summarize_errors(errors: list[str]) -> str:
    # Only called when every measured request failed, so `errors` is
    # always non-empty here.
    distinct = sorted(set(errors))
    if len(distinct) == 1:
        return distinct[0]
    return f"{len(distinct)} distinct errors, first: {distinct[0]}"


def run_workload(
    *, backend: InferenceBackend, workload: GeneratedWorkload, warmup_requests: int = 0
) -> BenchmarkResult:
    """Execute every request in `workload` against `backend` and build a result.

    `backend` must already be loaded (`backend.load()` succeeded) before
    calling this; `run_workload` never calls `load()` or `close()` itself.

    Args:
        backend: An already-loaded `InferenceBackend`.
        workload: The `GeneratedWorkload` to execute, one request at a time.
        warmup_requests: Number of warmup calls to run first (repeating
            the workload's first request), excluded from all statistics.

    Raises:
        ValueError: If `warmup_requests < 0`.
        BenchmarkExecutionError: If a warmup request fails. No measured
            requests run in that case.
    """
    if warmup_requests < 0:
        raise ValueError(f"warmup_requests must be >= 0, got {warmup_requests}")

    first_request = workload.requests[0]
    if warmup_requests > 0:
        _run_warmup(
            backend,
            warmup_requests,
            GenerationRequest(
                prompt=first_request.prompt, max_new_tokens=first_request.requested_output_tokens
            ),
        )

    collector = _GenerationMeasurementCollector()
    start = time.perf_counter()
    for generated_request in workload.requests:
        try:
            response = backend.generate(
                GenerationRequest(
                    prompt=generated_request.prompt,
                    max_new_tokens=generated_request.requested_output_tokens,
                )
            )
        except Exception as exc:
            # Intentionally broad: any per-request failure is recorded,
            # never silently discarded, and never aborts the whole run.
            collector.record_failure(str(exc) or type(exc).__name__)
            continue
        collector.record_success(
            latency_ms=response.latency_ms,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )
    elapsed_seconds = time.perf_counter() - start

    benchmark_request = _build_benchmark_request(workload, backend)
    gpu_info = _build_gpu_info(backend)
    return _build_result(
        request=benchmark_request,
        gpu=gpu_info,
        collector=collector,
        warmup_requests=warmup_requests,
        elapsed_seconds=elapsed_seconds,
    )
