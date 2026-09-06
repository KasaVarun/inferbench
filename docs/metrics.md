# InferBench metrics glossary

> **Status: schema only.** This document defines what each metric *means*
> and where it lives in the [`benchmark_core`](../packages/benchmark-core)
> domain model. As of Phase 1, InferBench does **not** measure any of these
> values yet — there is no benchmark execution engine. Every field described
> below is currently just a typed, validated slot waiting to be filled in by
> a future benchmark execution/reporting phase.

## Latency (`LatencyMetrics`)

All latency values are expressed in **milliseconds** and describe the
distribution of *per-request* end-to-end latency for a single benchmark run.

| Field | Meaning |
| --- | --- |
| `ttft_ms` | **Time To First Token.** Time from when a request is sent until the first output token (or first streamed chunk) is received. Only meaningful for streaming responses; `None` if the run did not stream. Lower is better — it's the dominant contributor to perceived responsiveness in chat-style workloads. |
| `tpot_ms` | **Time Per Output Token.** Average time to generate each subsequent output token after the first, i.e. the steady-state decode speed per token. Lower is better; this is what dominates total latency for long generations. |
| `p50_ms` | Median (50th percentile) end-to-end request latency. Represents "typical" request latency. |
| `p95_ms` | 95th percentile end-to-end request latency. Represents latency experienced by the slower 5% of requests — a common SLO target. |
| `p99_ms` | 99th percentile end-to-end request latency. Represents tail latency; sensitive to queuing, batching, and contention effects. |

Percentiles must satisfy `p50_ms <= p95_ms <= p99_ms` by definition of what a
percentile is — the model enforces this ordering.

## Throughput (`ThroughputMetrics`)

| Field | Meaning |
| --- | --- |
| `requests_per_second` | Number of completed requests per second, sustained over the benchmark run. The primary measure of serving capacity. |
| `input_tokens_per_second` | Aggregate rate at which input/prompt tokens were processed (prefill) across all concurrent requests. |
| `output_tokens_per_second` | Aggregate rate at which output tokens were generated (decode) across all concurrent requests. Often the key metric for cost/capacity planning. |
| `total_tokens_per_second` | Sum of input and output tokens per second; a single-number proxy for total compute throughput. |

## Memory (`MemoryMetrics`)

| Field | Meaning |
| --- | --- |
| `peak_allocated_mb` | **Peak allocated memory.** The highest amount of GPU memory actively allocated to tensors (weights, activations, KV cache) at any point during the run, in megabytes. |
| `peak_reserved_mb` | **Peak reserved memory.** The highest amount of GPU memory reserved/held by the memory allocator (which may exceed what's actively allocated due to caching/fragmentation), in megabytes. |
| `gpu_utilization_percent` | **GPU utilization.** Average percentage of time the GPU's compute units were busy during the run (0-100). Low utilization alongside high latency often points to scheduling, batching, or I/O bottlenecks rather than compute limits. |

## Cost (`CostMetrics`)

| Field | Meaning |
| --- | --- |
| `estimated_benchmark_cost_usd` | Total estimated dollar cost of running the entire benchmark, typically derived from GPU-hours consumed multiplied by an hourly rate. |
| `estimated_cost_per_request_usd` | **Cost per request.** `estimated_benchmark_cost_usd / request_count` — the marginal cost of serving one request under this configuration. |
| `estimated_cost_per_million_output_tokens_usd` | **Cost per 1M output tokens.** Normalizes cost by generation volume, enabling apples-to-apples comparison across configurations with different output lengths. This is the standard unit used when comparing inference providers/backends. |
| `tokens_per_dollar` | **Tokens per dollar.** The inverse efficiency metric: total tokens (input + output) produced per dollar spent. Higher is better; useful for ranking configurations by cost-efficiency rather than raw speed. |

## GPU environment (`GPUInfo`)

Not a "metric" in the performance sense, but recorded alongside every result
so that latency/throughput/memory/cost numbers can be correctly attributed:
`gpu_name`, `gpu_type`, `cuda_version`, and `framework_version` (the
inference backend/engine version, e.g. a specific vLLM release).

## Where this fits

These fields are defined in
[`packages/benchmark-core/src/benchmark_core/models.py`](../packages/benchmark-core/src/benchmark_core/models.py)
as part of `BenchmarkResult`. Populating them with real measurements —
actually running inference, timing requests, sampling GPU memory/utilization,
and computing cost from cloud pricing — is explicit **future work**, not
part of this phase.
