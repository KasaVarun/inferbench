# Local causal-LM inference on Apple MPS / CPU (Phase 4)

> **Status.** This is the first phase that actually runs a model. There is
> still **no CUDA, no remote GPU, no Triton, no vLLM** anywhere here --
> only local PyTorch + Hugging Face `transformers` on Apple Silicon (MPS)
> or CPU.

## 1. Purpose

Phase 3 generated deterministic *descriptions* of requests
(`GeneratedWorkload`) without ever touching a model. Phase 4 executes
those requests against a real, small causal language model running
locally, and turns the outcome into a Phase 1 `BenchmarkResult` --
closing the loop from "workload definition" to "measured result" before
any GPU/CUDA infrastructure exists.

## 2. Backend abstraction

`benchmark_core.inference_backend.InferenceBackend` is the reusable
contract every backend implements:

```python
class InferenceBackend(ABC):
    def load(self) -> None: ...
    def generate(self, request: GenerationRequest) -> GenerationResponse: ...
    def health(self) -> bool: ...
    def close(self) -> None: ...
    def info(self) -> BackendInfo: ...
```

- **No model load at import or construction time.** Constructing a
  backend does no I/O; only `load()` does.
- `health()` reflects whether `load()` has succeeded and `close()` has
  not since been called.
- `info()` (device, dtype, model name, framework version) is only valid
  after `load()`.
- `benchmark_core.local_transformers_backend.LocalTransformersBackend` is
  the only concrete implementation in this phase. Tests exercise the
  contract against a fake (`tests/fake_backend.py`) so orchestration logic
  is verified without ever loading a real model.

## 3. MPS vs CPU selection

```python
def is_mps_available() -> bool:
    return torch.backends.mps.is_built() and torch.backends.mps.is_available()
```

- `--device cpu`: always used, regardless of MPS availability.
- `--device mps`: used, or **fails immediately and clearly** if MPS isn't
  available -- explicitly requesting MPS never silently falls back.
- `--device auto` (default): MPS if available, else CPU.

There is no CUDA branch anywhere: Apple Silicon has no NVIDIA GPU, and a
CUDA code path here would be pure fiction.

## 4. Local model loading

`LocalTransformersBackend.load()`:

1. Resolves the device (§3) and dtype (§5).
2. Loads the tokenizer via `AutoTokenizer.from_pretrained(model_name)`.
3. If the tokenizer has no pad token (common for GPT-2-family models),
   sets `tokenizer.pad_token = tokenizer.eos_token` -- the standard
   workaround for single-sequence generation.
4. Loads the model via `AutoModelForCausalLM.from_pretrained(model_name, dtype=...)`,
   moves it to the resolved device, and calls `.eval()`.

Any failure (network, disk, incompatible model, out of memory) propagates
unmodified out of `load()`; the CLI catches it, prints a clear message,
and exits nonzero without writing any output.

## 5. Dtype strategy

| Device | dtype | Why |
| --- | --- | --- |
| `cpu` | `float32` | Reliable on every PyTorch CPU build; float16 CPU matmul support is inconsistent. |
| `mps` | `float16` | Generally stable for small causal LMs on Apple's MPS backend; halves memory/bandwidth vs. float32. |

The resolved dtype is exposed via `BackendInfo.dtype` and recorded in
`BenchmarkResult.request.configuration.dtype`.

## 6. Actual tokenizer token counts vs. Phase 3 estimates

**Phase 3's `estimated_input_tokens` is a character-based approximation**
(`ceil(len(text) / 4)`) computed with no tokenizer at all -- see
`docs/workloads.md`. Generated workload artifacts are **not modified** by
this phase.

At runtime, `LocalTransformersBackend.generate()` tokenizes the prompt
with the model's *real* tokenizer and reports the *actual* token count
(`GenerationResponse.prompt_tokens`, from `encoded["input_ids"].shape[1]`).
These two numbers will generally differ -- sometimes significantly,
depending on the model's vocabulary and the text's language/punctuation.
`BenchmarkResult`'s latency/throughput metrics are always computed from
these real, measured counts, never from the Phase 3 estimate.

## 7. Deterministic generation

Every request uses:

```python
model.generate(
    **encoded,
    max_new_tokens=request.max_new_tokens,
    do_sample=False,
    pad_token_id=tokenizer.pad_token_id,
)
```

`do_sample=False` means greedy decoding -- no temperature, no top-k/top-p,
no random seed needed for generation itself. The same prompt on the same
model/device/dtype produces the same output tokens every time. Output
token count excludes the prompt: `completion_ids = generated_ids[0][prompt_tokens:]`.

## 8. Timing methodology

- `time.perf_counter()` around the `model.generate()` call only --
  tokenization and decoding are outside the timed window, and model
  *loading* is never included in per-request latency.
- **MPS synchronization**: MPS dispatches work asynchronously, so an
  un-synchronized timer would measure dispatch time, not completion time.
  `torch.mps.synchronize()` is called immediately before and after the
  timed `generate()` call when running on MPS (checked via `hasattr`
  rather than assumed, since it's version-dependent; on CPU there is
  nothing to synchronize, since CPU execution is already synchronous from
  Python's perspective).

## 9. Warmup

`run-local --warmup N` runs `N` extra calls (repeating the workload's
first request) before the measured phase, exactly like Phase 2's
`BenchmarkRunner`. Warmup latencies never appear in `p50_ms`/`p95_ms`/
`p99_ms`/throughput; a warmup failure aborts the run before any measured
request executes (`BenchmarkExecutionError`, no output written).

## 10. Metrics populated

- `latency.p50_ms` / `p95_ms` / `p99_ms`: from real per-request wall-clock
  latencies (measured requests only).
- `latency.tpot_ms`: **(sum of successful requests' latency in ms) /
  (total output tokens across those requests)**. This intentionally
  amortizes each request's prompt-processing ("prefill") time across its
  output tokens -- it is an approximation of time-per-output-token, not a
  true incremental decode-only measurement (that would require per-token
  timestamps a single `generate()` call doesn't provide). `None` if no
  successful request produced any output tokens.
- `throughput.requests_per_second`, `input_tokens_per_second`,
  `output_tokens_per_second`, `total_tokens_per_second`: computed from
  real measured counts and elapsed time.
- `execution.*`: warmup/measured/successful/failed iteration counts and
  elapsed time, same shape as Phase 2.
- `gpu.gpu_name`: `"Apple MPS"` or `"CPU"`; `gpu.framework_version`:
  `"torch <version>; transformers <version>"`.

## 11. Metrics intentionally not populated

- `latency.ttft_ms`: always `None`. No first-token streaming
  instrumentation exists in this phase (a single `generate()` call
  returns only after the full completion); approximating it from total
  latency would be a fabrication, not a measurement.
- `memory.*`: always `None`. No GPU/MPS memory instrumentation exists;
  MPS memory is never mislabeled as NVIDIA GPU memory.
- `cost.*`: always `None`, same as every prior phase.
- `gpu.cuda_version`: always `None` -- there is no CUDA in this backend.

## 12. Smoke-test model

The default smoke-test model is **`sshleifer/tiny-gpt2`**: a tiny,
public, GPT-2-architecture model on the Hugging Face Hub, small enough to
download and run on CPU or MPS in seconds, with no gating/license friction
and full compatibility with `AutoTokenizer`/`AutoModelForCausalLM`. Any
comparably small, public, `transformers`-compatible causal LM works; the
model is always configurable via `--model`, never hardcoded into backend
logic.

## 13. CLI examples

```bash
# Generate a tiny workload (Phase 3; unchanged by this phase)
python -m benchmark_core generate-workload \
  --profile short_prompt_short_output \
  --requests 2 \
  --seed 42 \
  --output /tmp/inferbench_local_workload.json

# Run it through a local model
python -m benchmark_core run-local \
  --model sshleifer/tiny-gpt2 \
  --workload /tmp/inferbench_local_workload.json \
  --warmup 1 \
  --device auto \
  --output /tmp/inferbench_local_result.json
```

`--device` is optional (`auto` by default); `--model`, `--workload`,
`--warmup`, `--output` are required. `--workload` accepts either `.json`
or `.yaml`/`.yml`. Exit code `0` on success; nonzero on any load, warmup,
or all-requests-failed outcome, with a concise message on stderr.

## 14. Limitations

- Concurrency is fixed at 1 -- requests execute strictly sequentially, no
  async batching yet.
- `tpot_ms` is an amortized approximation (§10), not a true per-token
  decode measurement.
- No memory, cost, or CUDA metrics exist in this phase.
- Hugging Face Hub downloads require network access the first time a
  given model is used; there is no bundled/offline model.
- Never hardcodes a user-specific Hugging Face cache path -- caching
  behavior is whatever `transformers`/`huggingface_hub` do by default on
  the running machine.

## 15. No CUDA in this phase

Every device-selection and dtype-selection code path in
`local_transformers_backend.py` only ever resolves to `"mps"` or `"cpu"`.
There is no CUDA detection, no CUDA dtype branch, and no CUDA-specific
metadata field populated anywhere in this phase.
