# CPU benchmark harness (Phase 2)

> **Purpose.** This is a local, CPU-only harness used to validate
> InferBench's benchmark *execution infrastructure* -- warmup, timing,
> failure handling, aggregation, and result serialization -- before any
> GPU/CUDA/inference work exists. It exercises a deterministic NumPy matrix
> multiplication, **not** an LLM. Nothing here measures inference quality,
> and none of these numbers should be interpreted as GPU/LLM performance.

## What is measured

- **Wall-clock latency per measured iteration**, in milliseconds, via
  `time.perf_counter()` (a high-resolution monotonic clock immune to
  system clock adjustments).
- **Success/failure outcome per measured iteration.**
- **Aggregate latency percentiles** (p50/p95/p99) computed from the
  successful iterations' latencies.
- **Requests per second**, computed as
  `successful_iterations / total_elapsed_seconds` for the measured phase.
- **Execution bookkeeping**: how many iterations were warmup vs. measured,
  how many measured iterations succeeded/failed, and how long the measured
  phase took in total (`ExecutionMetadata`).

## What is NOT measured

This phase never fabricates a value it cannot actually compute. All of the
following are always `None` in results produced by `BenchmarkRunner`:

- `latency.ttft_ms`, `latency.tpot_ms` -- no streaming/token generation
  exists in a matmul workload.
- `throughput.input_tokens_per_second`, `output_tokens_per_second`,
  `total_tokens_per_second` -- there are no tokens.
- All of `MemoryMetrics` (`peak_allocated_mb`, `peak_reserved_mb`,
  `gpu_utilization_percent`) -- no memory/GPU instrumentation exists yet.
- All of `CostMetrics` -- no pricing model exists yet.
- All of `GPUInfo` -- this harness runs entirely on CPU; there is no GPU to
  describe.
- **FLOPS are never reported anywhere.** No FLOPS figure is computed or
  claimed by this harness.

## Warmup methodology

- `WarmupConfig.iterations` (`>= 0`) warmup iterations run the workload
  exactly like measured iterations, but their timings are discarded
  entirely -- they never appear in any latency/throughput calculation.
- If **any** warmup iteration raises, the run aborts immediately with a
  `BenchmarkExecutionError` and **no measured iteration ever runs**. A
  benchmark whose workload can't even complete a warmup call is not in a
  state where measurement is trustworthy, so this is treated as a hard
  failure of the run itself, not folded into the measured failure count.

## Timing method

Each measured iteration is timed individually:

```python
start = time.perf_counter()
workload()
end = time.perf_counter()
latency_ms = (end - start) * 1000.0
```

The workload's inputs (e.g. the two matrices to multiply) are generated
once, eagerly, before any warmup or measured call -- so allocation and
random-number generation never contaminate a measured call's timing.
Total elapsed time for the measured phase is likewise taken via
`time.perf_counter()` around the entire measured loop.

## Percentile calculation

Percentiles are computed with `numpy.percentile` over the latencies of
**successful** measured iterations only:

```python
p50, p95, p99 = np.percentile(latencies_ms, [50, 95, 99])
```

`numpy.percentile` is monotonically non-decreasing in the requested
percentile for any fixed dataset, so `p50 <= p95 <= p99` always holds --
this is also enforced independently by `LatencyMetrics`' own validator.

## Failure handling

- **Partial failures**: if some (but not all) measured iterations fail,
  the run still produces a **successful** `BenchmarkResult`. Latency and
  throughput are computed from the successful iterations only, and
  `execution.failed_iterations` preserves the exact number of failures --
  failures are counted, never silently dropped.
- **Total failure**: if *every* measured iteration fails, the run produces
  a `BenchmarkResult` with `success=False` and a non-empty `error`
  describing what failed (deduplicated error messages, with a count if
  there were multiple distinct errors). `latency`/`throughput` are set to
  zero-valued placeholders (there is nothing to aggregate).
- **Warmup failure**: see above -- this raises `BenchmarkExecutionError`
  instead of producing any `BenchmarkResult` at all.

## Atomic output behavior

The CLI never leaves a partial or corrupted result file on disk:

1. Missing parent directories for `--output` are created first
   (`Path.mkdir(parents=True, exist_ok=True)`).
2. The result JSON is written to a temporary file **in the same
   directory** as the destination (so the final rename stays on one
   filesystem).
3. The temporary file is flushed and `os.fsync`'d before anything else
   happens.
4. Only after a fully successful write is the temporary file moved into
   place with `os.replace()`, which is atomic on POSIX filesystems.
5. If *any* step before the replace fails, the temporary file is deleted
   and the exception propagates -- the destination path, if it already
   existed, is **never touched**, so a failed write can never overwrite a
   previously valid result with a partial one.

## Exact CLI example

```bash
python -m benchmark_core benchmark-cpu \
  --iterations 100 \
  --warmup 10 \
  --matrix-size 256 \
  --seed 42 \
  --output benchmarks/results/cpu_baseline.json
```

Flags:

| Flag | Constraint | Meaning |
| --- | --- | --- |
| `--iterations` | `> 0` | Measured iterations. |
| `--warmup` | `>= 0` | Warmup iterations (excluded from all metrics). |
| `--matrix-size` | `> 0` | Side length of the square float32 matrices multiplied each iteration. |
| `--seed` | any int | Seed for the deterministic NumPy input matrices. |
| `--output` | path | Where to atomically write the resulting `BenchmarkResult` JSON. |

Exit code is `0` on a successful result, `1` if all measured iterations
failed or warmup failed (in which case no output file is written).

## Schema note: this is not an LLM workload

`WorkloadConfiguration` (Phase 1) was designed around LLM-shaped workloads
and requires `output_tokens > 0`; `prompt_tokens`/`output_tokens` are
documented as token counts. The CPU matmul workload has no tokens at all.
Rather than fabricating a fake token count or forking the schema, the CLI
populates these fields with clearly-documented placeholders
(`prompt_tokens=0`, `output_tokens=1`) purely to satisfy the shared schema;
they carry no meaning for this workload. The fields that *do* carry real
meaning here are `workload.name` (encodes the matrix dimensions, e.g.
`cpu-matmul-256x256`) and `workload.seed` (the actual RNG seed used). See
`benchmark_core/cli.py` for the exact mapping.

## Limitation

This harness exists solely to prove out the benchmark execution
infrastructure (warmup, timing, aggregation, failure handling, atomic
JSON output) end-to-end on hardware every contributor already has: a CPU.
It is explicitly **not** a GPU, CUDA, Triton, or inference benchmark, and
its numbers say nothing about LLM serving performance. GPU-backed
benchmarking is future work.
