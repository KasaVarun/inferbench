# Synthetic workload generation (Phase 3)

> **Status.** This document describes `benchmark_core.workload_generation`
> and the `generate-workload` CLI command. **No model, tokenizer, or
> inference backend is executed anywhere in this phase.** These workloads
> are inputs that later phases will feed to real inference backends; they
> do not themselves run inference.

## 1. Purpose of synthetic workloads

Before InferBench can benchmark real inference backends, it needs
**consistent, reproducible request shapes** to feed them: prompts of a
given approximate length, a requested output length, and (for some
profiles) a controlled amount of shared or non-shared prefix content.
Synthetic generation means every backend under test sees byte-identical
requests for a given `(profile, seed, request_count, parameters)` --
without depending on any real dataset, network access, or licensing
concerns.

## 2. Why deterministic workloads matter

If regenerating "the same" workload produced different prompts each time,
comparing two benchmark runs (different backends, different GPUs,
different dates) would be comparing apples to oranges -- any performance
difference could be an artifact of different input shapes rather than a
real difference in the system under test. Determinism turns workload
generation into a fixed, versionable input, exactly like a config file.

## 3. The six workload profiles

| Profile | Target input | Requested output | Notes |
| --- | --- | --- | --- |
| `short_prompt_short_output` | ~64 tokens | 32 tokens | Baseline low-latency shape. |
| `long_prompt_short_output` | ~2048 tokens | 32 tokens | Prefill-heavy; output is cheap. |
| `short_prompt_long_output` | ~64 tokens | 512 tokens | Decode-heavy; prefill is cheap. |
| `shared_prefix` | ~1024 tokens | 64 tokens | A configurable fraction of requests share one exact prefix (default ratio: 1.0). |
| `random_prefix` | ~1024 tokens | 64 tokens | Every request gets its own distinct, deterministic content; no sharing. |
| `mixed_workload` | mixture | mixture | Deterministic round-robin over 4 fixed categories; see §8. |

## 4. Default approximate token targets

Exactly the values in the table above. These are **approximations**, not
guarantees -- see §5. `--input-tokens`/`--output-tokens` override the
input/output targets for any profile except `mixed_workload` (whose
categories have fixed targets that would be ambiguous to override with a
single value); `--shared-prefix-ratio` overrides the shared fraction for
`shared_prefix` only. Passing an override that isn't meaningful for the
selected profile is a clear, immediate error.

## 5. Token estimation method and its limitations

`benchmark_core.token_estimation.TokenEstimator` is a **local,
deterministic, network-free approximation** -- there is no real tokenizer
here. The default (`CharacterRatioTokenEstimator`) computes:

```python
estimated_tokens = ceil(len(text) / chars_per_token)  # chars_per_token = 4.0
```

This is a widely used rough heuristic for English BPE tokenizers (~4
characters/token), **not** an exact count. It is deliberately
character-based, not word-based: naively equating word count with token
count would be a materially different (and generally worse) approximation
-- multi-character words, punctuation, and whitespace all get folded into
the same length-based estimate instead. Every `GeneratedWorkload` carries
`token_counts_estimated: true` so consumers never mistake these figures for
exact tokenizer output, and `generator_configuration.token_estimator`
records exactly which method/parameters produced them (e.g.
`character_ratio(chars_per_token=4.0)`).

**Limitation**: this estimate will diverge from any real tokenizer's exact
count, sometimes significantly (e.g. for non-English text, code, or heavy
punctuation). It exists to make workload *shape* configurable and
reproducible, not to predict exact token usage.

## 6. Shared-prefix methodology (`shared_prefix` profile)

1. A single shared prefix is generated **once**, from its own dedicated
   deterministic RNG stream (seeded from the workload's seed and a
   reserved sentinel index, `-1`, that no real request ever uses). Its
   length is fixed at half of the target input token count.
2. `shared_prefix_ratio` (default `1.0`) determines how many requests, by
   position, are in the "shared" group: exactly the first
   `round(request_count * shared_prefix_ratio)` requests (by index) --
   an explicit, deterministic rule, not a random sample.
3. Each shared-group request's prompt is built as
   `shared_prefix_text + " " + <unique per-request suffix>` (or just
   `shared_prefix_text` if there's no room left for a suffix). Because
   this is string concatenation of the *exact same* prefix string, every
   shared-group prompt is byte-identical over that leading substring --
   never merely similar.
4. Requests outside the shared group get a fully independent prompt from
   their own derived RNG stream (different from the prefix's stream), so
   they do not draw from the same word sequence and do not, in practice,
   reproduce the shared prefix.

## 7. Random-prefix methodology (`random_prefix` profile)

Each request gets its own prompt from an RNG stream derived purely from
`(seed, request_index)` via integer arithmetic (see §9) -- there is no
shared component at all, so the entire generated prompt effectively *is*
that request's "random prefix" (nothing follows it structurally, unlike
`shared_prefix`, where a suffix follows the shared part). The same seed
always regenerates the exact same set of prefixes; a different seed
changes all of them.

## 8. Mixed-workload selection logic (`mixed_workload` profile)

Four fixed categories, defined once as
`MIXED_WORKLOAD_CATEGORIES` (a tuple, so iteration order is fixed):

1. `short_input_short_output` -- ~64 input tokens, 32 output tokens
2. `long_input_short_output` -- ~2048 input tokens, 32 output tokens
3. `short_input_long_output` -- ~64 input tokens, 512 output tokens
4. `long_input_long_output` -- ~2048 input tokens, 512 output tokens

**Exact selection rule**: request at index `i` gets
`MIXED_WORKLOAD_CATEGORIES[i % 4]`. This is a pure function of position --
it does not depend on `seed` at all, so category assignment is identical
across seeds; `seed` only changes the generated prompt *content* within
whatever category a given index lands in. No sets or dicts are iterated to
make this decision (only a fixed tuple), so it can never be a source of
nondeterminism.

## 9. Deterministic request IDs

Every `GeneratedRequest.request_id` is derived via SHA-256, never
`uuid4()` and never Python's built-in `hash()` (which is not guaranteed
stable across processes/versions):

```python
canonical = f"inferbench-workload|{profile}|seed={seed}|count={request_count}|index={index}"
request_id = "req-" + sha256(canonical.encode()).hexdigest()[:16]
```

This is unique within one workload (each `index` is distinct), identical
whenever the same workload is regenerated, and changes if `profile`,
`seed`, `request_count`, or `index` changes. Per-request RNG streams (for
prompt content) use the same inputs, combined by plain integer arithmetic
(`base_seed * 1_000_003 + index`) -- deliberately not hashing -- to seed
`random.Random`.

## 10. JSON/YAML output

`--output`'s file extension selects the format: `.json` for pretty,
2-space-indented JSON; `.yaml`/`.yml` for block-style YAML. Both are
produced from the exact same `GeneratedWorkload.model_dump(mode="json")`
dict, so field order (and therefore byte content, for identical inputs) is
consistent between runs. Output is always UTF-8 with exactly one trailing
newline. Any other extension fails immediately and clearly, before any
file is created or modified.

## 11. Atomic file writing

Workload artifacts reuse the exact same atomic-write primitive introduced
in Phase 2 for `BenchmarkResult` (`benchmark_core.atomic_io.write_atomic`,
generalized to bytes so both CLI commands share one implementation): a
temp file is written in the destination's own directory, flushed and
`fsync`'d, then swapped into place with `os.replace` (atomic on POSIX). If
serialization or writing fails at any point before that swap, the temp
file is removed and the destination -- if it already existed -- is left
completely untouched.

## 12. CLI commands

```bash
python -m benchmark_core generate-workload \
  --profile shared_prefix \
  --requests 100 \
  --seed 42 \
  --output benchmarks/workloads/shared_prefix.json
```

Required: `--profile` (one of the six names above), `--requests` (`> 0`),
`--seed` (any integer), `--output` (path; extension must be `.json`,
`.yaml`, or `.yml`).

Optional (only where meaningful for the chosen profile):
`--input-tokens` (`> 0`), `--output-tokens` (`> 0`),
`--shared-prefix-ratio` (`[0.0, 1.0]`, `shared_prefix` only).

Exit code `0` on success; nonzero on any validation or generation failure,
with a concise message on stderr and no output file written.

## 13. Limitations

- Token counts are approximations (§5), not real tokenizer output.
- The vocabulary used for synthetic text is a small, fixed, neutral word
  list -- prompts are not natural, coherent language, and are not meant to
  be; only their approximate length and prefix-sharing structure matter.
- `shared_prefix`'s "half the target is prefix" split and `mixed_workload`'s
  4-category round robin are simple, documented rules chosen for clarity,
  not tunable strategies (yet).
- No workload artifact identity/hash field is included; reproducibility is
  achieved by re-deriving byte-identical output from the same
  `(profile, seed, request_count, parameters)`, not by embedding a content
  hash in the artifact itself.

## 14. These workloads do not execute models

Nothing in this module or CLI command calls a model, a tokenizer, or a
network service. `generate-workload` produces a description of requests
to send later -- prompt text, requested output length, estimated input
length -- for a future inference-execution phase to actually run.
