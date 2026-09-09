# Remote CUDA smoke benchmarking with Modal (Phase 5)

## 1. Why CUDA runs remotely

InferBench development happens on an Apple Silicon M5 Pro. Its GPU uses
Apple Metal Performance Shaders (MPS), not NVIDIA CUDA, so it cannot execute
or validate CUDA kernels locally. Phase 5 uses a pay-per-run Modal container
with an NVIDIA GPU to validate the CUDA/PyTorch execution path while keeping
all ordinary development, linting, and unit tests local.

This phase benchmarks only deterministic matrix multiplication. It is
infrastructure validation, not LLM serving.

## 2. Architecture

`infra/modal/gpu_smoke.py` defines:

- one `modal.App`,
- one remote function containing the CUDA-only code, and
- one local CLI entrypoint that validates arguments, invokes the remote
  function once, validates its returned data as `CudaBenchmarkReport`, and
  prints a concise summary.

The app uses `modal run`; it does not define a web endpoint, scheduled job,
volume, secret, or persistent service. The remote function uses
`single_use_containers=True`, and the ephemeral app terminates when the
command completes.

Reusable, CUDA-runtime-independent models and aggregation live in
`benchmark_core.cuda_benchmark`. That module imports neither Modal nor
`torch.cuda`, so it is fully testable on macOS without authentication or
NVIDIA hardware. Modal is a development/remote-execution dependency, not an
import-time dependency of `benchmark_core`.

Because Phase 5 adds public shared models (`CudaBenchmarkConfiguration`,
`CudaEnvironment`, and `CudaBenchmarkReport`), the `benchmark_core` version
is bumped from 0.4.0 to 0.5.0.

## 3. Remote image

The Modal image is Debian slim with Python 3.12 and only:

- PyTorch,
- NumPy,
- Pydantic,
- PyYAML, and
- the local `benchmark_core` package, included with Modal 1.5.5
  `Image.add_local_python_source("benchmark_core", copy=True)`.

That API locates the locally installed package by name and copies it into
`/root` on the image. The remote script is mounted as `/root/gpu_smoke.py`
and must not derive a repository root from `__file__` (that path has no
usable grandparents).

PyTorch comes from its normal Python package; no NVIDIA wheel URL, CUDA
compiler toolkit, Triton, transformers, vLLM, or unrelated ML package is
installed. The image receives only source code, never local credentials,
caches, virtual environments, or benchmark artifacts.

## 4. Authentication

Install dependencies, then authenticate using the installed Modal CLI:

```bash
make setup
.venv/bin/modal setup
```

Modal 1.5.5 also exposes:

```bash
.venv/bin/modal token new
```

for creating credentials through an authenticated browser session. Never
place Modal token IDs, token secrets, or generated credential files in this
repository.

Check the installed client:

```bash
.venv/bin/modal --version
.venv/bin/modal token info
```

The local entrypoint also checks that credentials exist *before* launching a
GPU job. If they do not, it fails immediately with those setup commands and
never prints token IDs or secrets.

## 5. Supported GPUs and dtypes

GPU resources:

- `A10G` (default),
- `A100`.

Dtypes:

- `float32`,
- `float16` (default),
- `bfloat16`.

The local entrypoint validates these exact values. The remote function also
checks `torch.cuda.is_bf16_supported()` before using bfloat16. Unsupported
dtypes or hardware fail clearly; InferBench never silently substitutes a
different dtype or GPU.

The current Modal 1.5.5 API supports invocation-specific GPU resources with
`run_cuda_benchmark.with_options(gpu=...)`. This avoids obsolete
`modal.gpu.*` objects and permits one CLI entrypoint to select either
supported resource without defining a persistent deployment.

## 6. CUDA environment detection

Before allocating benchmark tensors, the remote function verifies and
captures:

- `torch.__version__`,
- `torch.version.cuda` (must not be `None`),
- `torch.cuda.is_available()` (must be `True`),
- `torch.cuda.device_count()` (must be positive),
- `torch.cuda.get_device_name(current_device)`,
- `torch.cuda.get_device_capability(current_device)`,
- `torch.cuda.get_device_properties(current_device).total_memory`,
- `torch.cuda.current_device()`, and
- `platform.python_version()`.

If CUDA is unavailable or no CUDA device exists, the remote invocation raises
a clear error and emits no successful benchmark report.

## 7. Deterministic matrix multiplication

`CudaBenchmarkConfiguration` defaults to:

- matrix size: 2048×2048,
- measured iterations: 50,
- warmup iterations: 10,
- dtype: float16,
- seed: 42,
- GPU: A10G.

Inputs are generated deterministically on CPU in float32 using a seeded
`torch.Generator`, then transferred to CUDA and cast to the selected dtype
before any timed operation. Allocation, random generation, dtype conversion,
and host-to-device transfer are therefore excluded from matmul latency.
Only `torch.matmul(left, right)` appears inside each timed interval.

## 8. Correctness validation

Performance is reported only after numerical validation:

1. Compute a float32 CPU reference from the exact generated inputs.
2. Compute one CUDA result at the requested dtype.
3. Synchronize CUDA.
4. Copy the CUDA result to CPU as float32.
5. Compare using explicit dtype-specific tolerances:
   - float32: `rtol=1e-4`, `atol=1e-3`,
   - float16: `rtol=1e-2`, `atol=5e-1`,
   - bfloat16: `rtol=5e-2`, `atol=2.0`.

These tolerances account for expected lower-precision accumulation error
without treating arbitrary disagreement as correct. A failed comparison
raises an error before performance metrics are constructed.

## 9. Warmup

The correctness operation creates the CUDA context and initializes relevant
libraries. Configured warmup matmuls then absorb allocator, cuBLAS, and
kernel-selection effects. CUDA is synchronized after warmup. Warmup
iterations never enter latency percentiles, throughput, or measured
execution counts.

## 10. CUDA Event timing and synchronization

Two separate timing concepts are collected for every measured iteration:

1. **CUDA device time** from `torch.cuda.Event(enable_timing=True)`.
2. **Host wall-clock time** from `time.perf_counter()`.

Each measured iteration follows:

1. Create start/end CUDA Events.
2. Start the host timer.
3. Record the start event.
4. Execute one `torch.matmul`.
5. Record the end event.
6. Synchronize the end event.
7. Stop the host timer.
8. Read `start_event.elapsed_time(end_event)`.

CUDA work is asynchronous. Without synchronization, host timing would mostly
measure kernel dispatch rather than completed work, and event elapsed time
would not be safe to consume. A full `torch.cuda.synchronize()` also runs
before and after the measured sequence.

`BenchmarkResult.latency.p50_ms/p95_ms/p99_ms` are CUDA Event/device
percentiles. Raw CUDA and host per-iteration timing arrays remain in
`CudaBenchmarkReport` so callers can inspect both without conflating them.

## 11. CUDA memory measurement

After correctness and warmup, immediately before measured execution:

```python
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
```

After measured execution:

```python
torch.cuda.max_memory_allocated()
torch.cuda.max_memory_reserved()
```

Bytes are converted to MiB using `bytes / 1024**2`, then mapped to
`MemoryMetrics.peak_allocated_mb` and `peak_reserved_mb`. GPU utilization
remains `None`; no utilization sampler exists in this phase. Large tensors
are deleted after measurement and `empty_cache()` is called as cleanup, not
as a correctness mechanism.

## 12. BenchmarkResult mapping

- `backend`: `"pytorch-cuda"`,
- `model_name`: `"cuda-matmul-<N>x<N>"`,
- `dtype`: actual requested dtype,
- `batch_size`: 1,
- `concurrency`: 1,
- `prefix_caching`: false,
- `quantization`: `None`,
- `gpu_type`: configured Modal GPU resource,
- `GPUInfo.gpu_name`: actual PyTorch-reported device name,
- `GPUInfo.cuda_version`: actual `torch.version.cuda`,
- `GPUInfo.framework_version`: actual `torch.__version__`,
- latency percentiles: CUDA Event timings,
- memory: measured CUDA allocator peaks,
- cost: all `None`.

`ThroughputMetrics.requests_per_second` is the closest existing
backward-compatible field: in this non-LLM benchmark, one “request” means one
completed matrix-multiplication operation, and the rate is computed from the
sum of CUDA Event times. Token throughput fields remain `None`.

The shared Phase 1 `WorkloadConfiguration` is LLM-shaped, so this benchmark
reuses the documented Phase 2 convention: `prompt_tokens=0` and
`output_tokens=1` are schema placeholders only. Matrix dimensions are never
presented as tokens.

## 13. Metrics intentionally not populated

- TTFT and TPOT: not an LLM generation workload, both `None`.
- input/output/total token throughput: `None`.
- amortized output-token time: `None`.
- GPU utilization: `None` (not sampled).
- all cost metrics: `None` (no live pricing source).

No fake token, utilization, cost, or CUDA values are emitted.

## 14. Commands

Default benchmark:

```bash
.venv/bin/modal run infra/modal/gpu_smoke.py
```

Low-cost acceptance smoke:

```bash
.venv/bin/modal run infra/modal/gpu_smoke.py \
  --gpu A10G \
  --matrix-size 1024 \
  --iterations 20 \
  --warmup 5 \
  --dtype float16 \
  --output /tmp/inferbench_cuda_smoke.json
```

The `--output` file is a JSON-serialized `BenchmarkResult` (the typed
InferBench result), written atomically by the local entrypoint. The remote
function still returns a `CudaBenchmarkReport` over the Modal call so the
entrypoint can validate environment metadata, correctness, and both
per-iteration timing arrays before writing.

## 15. Limitations

- Pay-per-run Modal authentication and network access are required for the
  real remote smoke benchmark.
- The benchmark is single-GPU and single-stream.
- Host wall times include event recording and synchronization overhead.
- “requests per second” means matmul operations per second in this phase.
- GPU utilization and pricing are not collected.
- This validates stock `torch.matmul`; custom Triton/CUDA kernels arrive in
  later phases.
