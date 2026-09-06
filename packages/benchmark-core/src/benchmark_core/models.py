"""Typed domain models for InferBench.

This module defines the schema used to describe benchmark *requests*
(what to run and how) and benchmark *results* (what happened). It is a
Phase 1 deliverable: it defines shape and validation rules only. No model
here executes a benchmark, talks to a GPU, or measures anything -- values
are always supplied by whatever code constructs these models.

All models are Pydantic v2 ``BaseModel`` subclasses, which gives us:

- runtime validation of constructor arguments,
- JSON (de)serialization via ``model_dump_json`` / ``model_validate_json``,
  including native round-tripping of ``UUID`` and timezone-aware
  ``datetime`` values.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

__all__ = [
    "BenchmarkConfiguration",
    "BenchmarkRequest",
    "BenchmarkResult",
    "CostMetrics",
    "GPUInfo",
    "LatencyMetrics",
    "MemoryMetrics",
    "ThroughputMetrics",
    "WorkloadConfiguration",
]


def _require_non_empty(value: str, field_name: str) -> str:
    """Strip whitespace and reject empty strings.

    A string consisting only of whitespace (e.g. ``"   "``) is treated as
    empty, since ``min_length=1`` alone would let it through.
    """
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must not be empty")
    return stripped


class WorkloadConfiguration(BaseModel):
    """Describes the synthetic or replayed workload a benchmark runs against."""

    name: str = Field(min_length=1, description="Human-readable workload identifier.")
    prompt_tokens: int = Field(ge=0, description="Number of input/prompt tokens per request.")
    output_tokens: int = Field(gt=0, description="Number of output tokens generated per request.")
    request_count: int = Field(gt=0, description="Total number of requests issued in the run.")
    shared_prefix_ratio: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction (0.0-1.0) of the prompt shared as a common prefix across requests.",
    )
    seed: int = Field(description="Random seed used to generate/replay the workload.")

    @field_validator("name")
    @classmethod
    def _name_not_empty(cls, value: str) -> str:
        return _require_non_empty(value, "name")


class BenchmarkConfiguration(BaseModel):
    """Describes how the model server/engine is configured for a run."""

    dtype: str = Field(min_length=1, description="Numeric dtype used for inference, e.g. fp16.")
    batch_size: int = Field(gt=0, description="Request batch size.")
    concurrency: int = Field(gt=0, description="Number of concurrent in-flight requests.")
    prefix_caching: bool = Field(description="Whether prefix/KV-cache reuse is enabled.")
    quantization: str | None = Field(
        default=None, description="Quantization scheme, e.g. int8, awq. None means unquantized."
    )
    gpu_type: str | None = Field(
        default=None, description="GPU SKU used for this configuration, e.g. H100."
    )
    max_model_length: int | None = Field(
        default=None,
        gt=0,
        description="Maximum context length (tokens) configured for the model, if constrained.",
    )


class BenchmarkRequest(BaseModel):
    """A fully specified request to run a benchmark: what model, on what backend, how."""

    model_name: str = Field(min_length=1, description="Identifier of the model under test.")
    backend: str = Field(min_length=1, description="Inference backend/engine, e.g. vllm.")
    workload: WorkloadConfiguration
    configuration: BenchmarkConfiguration

    @field_validator("model_name")
    @classmethod
    def _model_name_not_empty(cls, value: str) -> str:
        return _require_non_empty(value, "model_name")

    @field_validator("backend")
    @classmethod
    def _backend_not_empty(cls, value: str) -> str:
        return _require_non_empty(value, "backend")


class LatencyMetrics(BaseModel):
    """Latency distribution for a benchmark run, in milliseconds.

    Schema only: this phase does not measure latency. Values are expected
    to be supplied by a future benchmark execution layer.
    """

    p50_ms: float = Field(ge=0, description="50th percentile end-to-end request latency (ms).")
    p95_ms: float = Field(ge=0, description="95th percentile end-to-end request latency (ms).")
    p99_ms: float = Field(ge=0, description="99th percentile end-to-end request latency (ms).")
    ttft_ms: float | None = Field(
        default=None, ge=0, description="Time to first token (ms), if streaming was used."
    )
    tpot_ms: float | None = Field(
        default=None, ge=0, description="Time per output token (ms), if measured."
    )

    @model_validator(mode="after")
    def _check_percentile_ordering(self) -> LatencyMetrics:
        if not (self.p50_ms <= self.p95_ms <= self.p99_ms):
            raise ValueError(
                "latency percentiles must satisfy p50_ms <= p95_ms <= p99_ms "
                f"(got p50={self.p50_ms}, p95={self.p95_ms}, p99={self.p99_ms})"
            )
        return self


class ThroughputMetrics(BaseModel):
    """Aggregate throughput for a benchmark run.

    Schema only: this phase does not measure throughput.
    """

    requests_per_second: float = Field(ge=0, description="Completed requests per second.")
    input_tokens_per_second: float | None = Field(
        default=None, ge=0, description="Input/prompt tokens processed per second."
    )
    output_tokens_per_second: float | None = Field(
        default=None, ge=0, description="Output tokens generated per second."
    )
    total_tokens_per_second: float | None = Field(
        default=None, ge=0, description="Combined input + output tokens per second."
    )


class MemoryMetrics(BaseModel):
    """GPU memory usage observed during a benchmark run.

    Schema only: this phase does not measure memory usage.
    """

    peak_allocated_mb: float | None = Field(
        default=None, ge=0, description="Peak allocated GPU memory (MB)."
    )
    peak_reserved_mb: float | None = Field(
        default=None, ge=0, description="Peak reserved GPU memory (MB)."
    )
    gpu_utilization_percent: float | None = Field(
        default=None, ge=0, le=100, description="Average GPU utilization (%) during the run."
    )


class CostMetrics(BaseModel):
    """Derived cost figures for a benchmark run.

    Schema only: this phase does not compute costs.
    """

    estimated_benchmark_cost_usd: float | None = Field(
        default=None, ge=0, description="Total estimated cost of running the benchmark (USD)."
    )
    estimated_cost_per_request_usd: float | None = Field(
        default=None, ge=0, description="Estimated cost per request (USD)."
    )
    estimated_cost_per_million_output_tokens_usd: float | None = Field(
        default=None, ge=0, description="Estimated cost per 1M output tokens (USD)."
    )
    tokens_per_dollar: float | None = Field(
        default=None, ge=0, description="Total tokens produced per USD spent."
    )


class GPUInfo(BaseModel):
    """Identifying information about the GPU/environment a run executed on."""

    gpu_name: str | None = Field(default=None, description="Reported GPU device name.")
    gpu_type: str | None = Field(default=None, description="GPU SKU/class, e.g. H100, A100.")
    cuda_version: str | None = Field(default=None, description="CUDA runtime/driver version.")
    framework_version: str | None = Field(
        default=None, description="Version of the inference framework/backend used."
    )


class BenchmarkResult(BaseModel):
    """The full outcome of a single benchmark run."""

    run_id: UUID = Field(default_factory=uuid4, description="Unique identifier for this run.")
    created_at: datetime = Field(description="Timezone-aware timestamp when the run completed.")
    request: BenchmarkRequest
    latency: LatencyMetrics
    throughput: ThroughputMetrics
    memory: MemoryMetrics
    cost: CostMetrics
    gpu: GPUInfo
    success: bool
    error: str | None = Field(default=None, description="Error message, required if not success.")

    @field_validator("created_at")
    @classmethod
    def _created_at_must_be_tz_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _check_success_error_consistency(self) -> BenchmarkResult:
        if self.success and self.error is not None:
            raise ValueError("a successful result must not have an error message")
        if not self.success and not (self.error and self.error.strip()):
            raise ValueError("a failed result must contain a non-empty error message")
        return self
