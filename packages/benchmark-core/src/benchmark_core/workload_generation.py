"""Deterministic synthetic LLM workload generation (Phase 3).

This module generates *shapes* of LLM requests -- prompt text, requested
output length, estimated input length -- for exercising later inference
backends consistently. It never calls a model, a tokenizer, or a network
service. See `docs/workloads.md` for the full methodology writeup.

Determinism is the core contract: for identical
``(profile, request_count, seed, target_input_tokens,
requested_output_tokens, shared_prefix_ratio)``, `generate_workload`
always produces byte-identical `GeneratedRequest` content and request IDs.
No `uuid4()`, no live timestamps, and no Python `hash()` are used anywhere
in this module -- request IDs are derived from SHA-256, and all randomness
comes from `random.Random` seeded with plain integers derived by simple
arithmetic from the caller's seed.
"""

from __future__ import annotations

import hashlib
import random
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from benchmark_core.token_estimation import DEFAULT_TOKEN_ESTIMATOR, TokenEstimator

__all__ = [
    "GeneratedRequest",
    "GeneratedWorkload",
    "GeneratorConfiguration",
    "MIXED_WORKLOAD_CATEGORIES",
    "PROFILE_DEFINITIONS",
    "SUPPORTED_PROFILES",
    "WorkloadProfileDefinition",
    "WorkloadProfileName",
    "generate_workload",
]

WorkloadProfileName = Literal[
    "short_prompt_short_output",
    "long_prompt_short_output",
    "short_prompt_long_output",
    "shared_prefix",
    "random_prefix",
    "mixed_workload",
]

SUPPORTED_PROFILES: tuple[WorkloadProfileName, ...] = (
    "short_prompt_short_output",
    "long_prompt_short_output",
    "short_prompt_long_output",
    "shared_prefix",
    "random_prefix",
    "mixed_workload",
)

_SCHEMA_VERSION = 1

# A small, fixed, neutral vocabulary used to build synthetic prompt text.
# Deliberately domain-neutral (systems/data-engineering terms) so prompts
# read as plausible technical text without resembling any real dataset,
# and fixed (a tuple, not a set) so iteration order is never a source of
# nondeterminism.
_VOCABULARY: tuple[str, ...] = (
    "system", "data", "process", "model", "request", "response", "server",
    "client", "value", "result", "input", "output", "token", "sequence",
    "vector", "matrix", "batch", "queue", "pipeline", "module", "config",
    "parameter", "metric", "latency", "throughput", "cache", "memory",
    "network", "protocol", "schema", "record", "field", "index", "table",
    "cluster", "node", "worker", "task", "job", "event", "signal", "state",
    "context", "session", "thread", "buffer", "stream", "channel", "packet",
    "frame", "layer", "graph", "tensor", "weight", "gradient", "epoch",
    "sample", "label", "feature", "dataset", "benchmark", "profile",
    "workload", "engine", "runtime", "kernel", "device", "instance",
    "region", "zone", "shard", "partition", "replica", "version", "release",
    "build", "artifact", "log", "trace", "span", "counter", "gauge",
    "histogram", "summary", "alert", "policy", "rule", "constraint",
    "boundary", "limit", "threshold", "window", "interval", "duration",
    "offset", "delta", "ratio", "fraction", "percentage", "report",
)  # fmt: skip

# Round-robin category cycle for `mixed_workload`. A tuple (not a set/dict)
# so iteration order -- and therefore category selection -- is fixed.
MIXED_WORKLOAD_CATEGORIES: tuple[tuple[str, int, int], ...] = (
    ("short_input_short_output", 64, 32),
    ("long_input_short_output", 2048, 32),
    ("short_input_long_output", 64, 512),
    ("long_input_long_output", 2048, 512),
)


class WorkloadProfileDefinition(BaseModel):
    """Static default parameters for a named workload profile.

    For `mixed_workload`, there is no single input/output target -- it is
    a deterministic mixture of `MIXED_WORKLOAD_CATEGORIES` -- so both
    token targets are `None` there.
    """

    name: WorkloadProfileName
    description: str = Field(min_length=1)
    target_input_tokens: int | None = Field(default=None, gt=0)
    requested_output_tokens: int | None = Field(default=None, gt=0)
    shared_prefix_ratio: float = Field(ge=0.0, le=1.0, default=0.0)


PROFILE_DEFINITIONS: dict[WorkloadProfileName, WorkloadProfileDefinition] = {
    "short_prompt_short_output": WorkloadProfileDefinition(
        name="short_prompt_short_output",
        description="Short prompt (~64 tokens), short requested output (32 tokens).",
        target_input_tokens=64,
        requested_output_tokens=32,
    ),
    "long_prompt_short_output": WorkloadProfileDefinition(
        name="long_prompt_short_output",
        description="Long prompt (~2048 tokens), short requested output (32 tokens).",
        target_input_tokens=2048,
        requested_output_tokens=32,
    ),
    "short_prompt_long_output": WorkloadProfileDefinition(
        name="short_prompt_long_output",
        description="Short prompt (~64 tokens), long requested output (512 tokens).",
        target_input_tokens=64,
        requested_output_tokens=512,
    ),
    "shared_prefix": WorkloadProfileDefinition(
        name="shared_prefix",
        description=(
            "~1024-token prompts; the configured fraction (default 1.0) of requests "
            "share one exact, deterministic text prefix."
        ),
        target_input_tokens=1024,
        requested_output_tokens=64,
        shared_prefix_ratio=1.0,
    ),
    "random_prefix": WorkloadProfileDefinition(
        name="random_prefix",
        description="~1024-token prompts; each request gets its own distinct, deterministic text.",
        target_input_tokens=1024,
        requested_output_tokens=64,
        shared_prefix_ratio=0.0,
    ),
    "mixed_workload": WorkloadProfileDefinition(
        name="mixed_workload",
        description=(
            "Deterministic round-robin mixture of short/long input x short/long output "
            "requests; see MIXED_WORKLOAD_CATEGORIES."
        ),
    ),
}


class GeneratorConfiguration(BaseModel):
    """Exact parameters used to generate a workload, for reproducibility."""

    target_input_tokens: int | None = Field(
        default=None, description="Resolved target input length in estimated tokens."
    )
    requested_output_tokens: int | None = Field(
        default=None, description="Resolved requested output length in tokens."
    )
    shared_prefix_ratio: float = Field(
        ge=0.0, le=1.0, description="Fraction of requests sharing an exact prefix (shared_prefix)."
    )
    token_estimator: str = Field(
        min_length=1, description="Identifier of the token estimation method used."
    )


class GeneratedRequest(BaseModel):
    """A single synthetically generated, LLM-shaped request.

    None of these fields come from a real tokenizer or a real model --
    `estimated_input_tokens` is always an approximation (see
    `benchmark_core.token_estimation`), and `prompt` is synthetic,
    locally-generated text.
    """

    request_id: str = Field(min_length=1, description="Deterministic, SHA-256-derived ID.")
    index: int = Field(ge=0, description="Zero-based position within the workload.")
    prompt: str = Field(description="Synthetic, locally-generated prompt text.")
    requested_output_tokens: int = Field(gt=0, description="Requested output length in tokens.")
    estimated_input_tokens: int = Field(
        ge=0, description="Approximate input token count for `prompt` (never exact)."
    )
    profile: WorkloadProfileName
    category: str | None = Field(
        default=None, description="Sub-category within `profile`, e.g. for mixed_workload."
    )
    shares_common_prefix: bool = Field(
        default=False,
        description="True if `prompt` begins with the shared prefix (shared_prefix profile only).",
    )


class GeneratedWorkload(BaseModel):
    """A complete, reproducible, serializable synthetic workload."""

    schema_version: int = Field(default=_SCHEMA_VERSION, ge=1)
    profile: WorkloadProfileName
    seed: int
    request_count: int = Field(gt=0)
    token_counts_estimated: bool = Field(
        default=True,
        description="Always true: every token count here is an approximation, never exact.",
    )
    generator_configuration: GeneratorConfiguration
    requests: list[GeneratedRequest]

    @model_validator(mode="after")
    def _check_request_count_matches(self) -> GeneratedWorkload:
        if len(self.requests) != self.request_count:
            raise ValueError(
                f"request_count={self.request_count} but got {len(self.requests)} requests"
            )
        return self


def _derive_request_seed(base_seed: int, index: int) -> int:
    """Deterministically combine a base seed and index into a per-request seed.

    Pure integer arithmetic only -- no `hash()`, no randomness -- so the
    same `(base_seed, index)` always yields the same derived seed.
    """
    return base_seed * 1_000_003 + index


def _make_request_id(*, profile: str, seed: int, request_count: int, index: int) -> str:
    """Derive a deterministic, unique-within-workload request ID via SHA-256.

    Deliberately not `uuid4()` (nondeterministic) and not Python's
    built-in `hash()` (not stable across processes/versions). The exact
    same inputs always produce the exact same ID; changing any of
    `profile`/`seed`/`request_count`/`index` changes it.
    """
    canonical = f"inferbench-workload|{profile}|seed={seed}|count={request_count}|index={index}"
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"req-{digest[:16]}"


def _words_for_token_target(
    rng: random.Random, target_tokens: int, estimator: TokenEstimator
) -> list[str]:
    """Grow a word list from `rng` until its estimated token count reaches `target_tokens`."""
    if target_tokens <= 0:
        return []
    words: list[str] = []
    while estimator.estimate(" ".join(words)) < target_tokens:
        words.append(rng.choice(_VOCABULARY))
    return words


def _generate_independent_prompt(
    *, seed: int, index: int, target_input_tokens: int, estimator: TokenEstimator
) -> str:
    """Build one request's prompt from its own independent, derived RNG stream."""
    rng = random.Random(_derive_request_seed(seed, index))
    return " ".join(_words_for_token_target(rng, target_input_tokens, estimator))


def _build_simple_requests(
    *,
    profile: WorkloadProfileName,
    seed: int,
    request_count: int,
    target_input_tokens: int,
    requested_output_tokens: int,
    estimator: TokenEstimator,
) -> list[GeneratedRequest]:
    """Shared builder for profiles with one fixed target per request.

    Covers `short_prompt_short_output`, `long_prompt_short_output`,
    `short_prompt_long_output`, and `random_prefix` (which has no shared
    component: each request's independently-derived prompt *is* its
    "random prefix", since nothing else follows it).
    """
    requests: list[GeneratedRequest] = []
    for index in range(request_count):
        prompt = _generate_independent_prompt(
            seed=seed, index=index, target_input_tokens=target_input_tokens, estimator=estimator
        )
        requests.append(
            GeneratedRequest(
                request_id=_make_request_id(
                    profile=profile, seed=seed, request_count=request_count, index=index
                ),
                index=index,
                prompt=prompt,
                requested_output_tokens=requested_output_tokens,
                estimated_input_tokens=estimator.estimate(prompt),
                profile=profile,
            )
        )
    return requests


def _build_shared_prefix_requests(
    *,
    seed: int,
    request_count: int,
    target_input_tokens: int,
    requested_output_tokens: int,
    shared_prefix_ratio: float,
    estimator: TokenEstimator,
) -> list[GeneratedRequest]:
    """Build requests where the configured fraction share one exact prefix.

    The shared prefix is generated once from its own dedicated RNG stream
    (derived from index ``-1``, which no real request index ever uses) and
    is exactly half of the target input length; the remainder of each
    "shared" request's prompt is a per-request unique suffix appended
    after it, so the prefix bytes are always an exact leading substring.
    Requests outside the shared fraction get a fully independent prompt
    from their own derived RNG stream (a different stream than the
    prefix's), so they do not draw from the same word sequence as the
    prefix and do not, in practice, reproduce it.

    The first ``round(request_count * shared_prefix_ratio)`` requests (by
    index) are the shared group; this is an explicit, documented,
    deterministic rule -- not a random subset.
    """
    profile: WorkloadProfileName = "shared_prefix"
    prefix_target_tokens = max(1, round(target_input_tokens / 2))
    prefix_rng = random.Random(_derive_request_seed(seed, -1))
    shared_prefix_text = " ".join(
        _words_for_token_target(prefix_rng, prefix_target_tokens, estimator)
    )
    shared_count = round(request_count * shared_prefix_ratio)

    requests: list[GeneratedRequest] = []
    for index in range(request_count):
        is_shared = index < shared_count
        if is_shared:
            suffix_target = max(0, target_input_tokens - prefix_target_tokens)
            rng = random.Random(_derive_request_seed(seed, index))
            suffix_words = _words_for_token_target(rng, suffix_target, estimator)
            prompt = (
                shared_prefix_text
                if not suffix_words
                else f"{shared_prefix_text} {' '.join(suffix_words)}"
            )
        else:
            prompt = _generate_independent_prompt(
                seed=seed, index=index, target_input_tokens=target_input_tokens, estimator=estimator
            )

        requests.append(
            GeneratedRequest(
                request_id=_make_request_id(
                    profile=profile, seed=seed, request_count=request_count, index=index
                ),
                index=index,
                prompt=prompt,
                requested_output_tokens=requested_output_tokens,
                estimated_input_tokens=estimator.estimate(prompt),
                profile=profile,
                shares_common_prefix=is_shared,
            )
        )
    return requests


def _build_mixed_workload_requests(
    *, seed: int, request_count: int, estimator: TokenEstimator
) -> list[GeneratedRequest]:
    """Build a deterministic round-robin mixture of the four fixed categories.

    Category selection is ``MIXED_WORKLOAD_CATEGORIES[index % 4]`` --
    purely a function of request position, independent of `seed`. The
    `seed` only affects each request's generated prompt *content* within
    its category, not which category it gets.
    """
    profile: WorkloadProfileName = "mixed_workload"
    requests: list[GeneratedRequest] = []
    for index in range(request_count):
        category, target_input_tokens, requested_output_tokens = MIXED_WORKLOAD_CATEGORIES[
            index % len(MIXED_WORKLOAD_CATEGORIES)
        ]
        prompt = _generate_independent_prompt(
            seed=seed, index=index, target_input_tokens=target_input_tokens, estimator=estimator
        )
        requests.append(
            GeneratedRequest(
                request_id=_make_request_id(
                    profile=profile, seed=seed, request_count=request_count, index=index
                ),
                index=index,
                prompt=prompt,
                requested_output_tokens=requested_output_tokens,
                estimated_input_tokens=estimator.estimate(prompt),
                profile=profile,
                category=category,
            )
        )
    return requests


def generate_workload(
    profile: str,
    request_count: int,
    seed: int,
    *,
    target_input_tokens: int | None = None,
    requested_output_tokens: int | None = None,
    shared_prefix_ratio: float | None = None,
    token_estimator: TokenEstimator | None = None,
) -> GeneratedWorkload:
    """Generate a complete, reproducible synthetic workload.

    Args:
        profile: One of `SUPPORTED_PROFILES`.
        request_count: Number of requests to generate. Must be > 0.
        seed: Deterministic RNG seed. Identical inputs always produce
            identical output; a different seed changes prompt content
            (and, for `shared_prefix`, the shared prefix text).
        target_input_tokens: Overrides the profile's default input token
            target. Not meaningful for `mixed_workload` (its categories
            each have fixed targets).
        requested_output_tokens: Overrides the profile's default output
            token target. Not meaningful for `mixed_workload`.
        shared_prefix_ratio: Overrides the profile's default shared-prefix
            fraction. Only meaningful for the `shared_prefix` profile.
        token_estimator: Overrides the default character-ratio token
            estimator.

    Raises:
        ValueError: If `profile` is unsupported, `request_count` is not
            positive, a token target is not positive, `shared_prefix_ratio`
            is outside `[0.0, 1.0]`, or an override is supplied that is not
            meaningful for the given `profile`.
    """
    if profile not in SUPPORTED_PROFILES:
        raise ValueError(f"unsupported profile '{profile}'; must be one of {SUPPORTED_PROFILES}")
    if request_count <= 0:
        raise ValueError(f"request_count must be > 0, got {request_count}")
    if target_input_tokens is not None and target_input_tokens <= 0:
        raise ValueError(f"target_input_tokens must be > 0, got {target_input_tokens}")
    if requested_output_tokens is not None and requested_output_tokens <= 0:
        raise ValueError(f"requested_output_tokens must be > 0, got {requested_output_tokens}")
    if shared_prefix_ratio is not None and not (0.0 <= shared_prefix_ratio <= 1.0):
        raise ValueError(
            f"shared_prefix_ratio must be within [0.0, 1.0], got {shared_prefix_ratio}"
        )
    if profile == "mixed_workload" and (
        target_input_tokens is not None or requested_output_tokens is not None
    ):
        raise ValueError(
            "target_input_tokens/requested_output_tokens are not meaningful for "
            "mixed_workload: each category has fixed token targets"
        )
    if profile != "shared_prefix" and shared_prefix_ratio is not None:
        raise ValueError("shared_prefix_ratio is only meaningful for the shared_prefix profile")

    resolved_profile: WorkloadProfileName = profile  # narrowed by the membership check above
    definition = PROFILE_DEFINITIONS[resolved_profile]
    estimator = token_estimator or DEFAULT_TOKEN_ESTIMATOR

    resolved_ratio = (
        shared_prefix_ratio if shared_prefix_ratio is not None else definition.shared_prefix_ratio
    )

    if resolved_profile == "mixed_workload":
        requests = _build_mixed_workload_requests(
            seed=seed, request_count=request_count, estimator=estimator
        )
        resolved_input_tokens: int | None = None
        resolved_output_tokens: int | None = None
    else:
        resolved_input_tokens = target_input_tokens or definition.target_input_tokens
        resolved_output_tokens = requested_output_tokens or definition.requested_output_tokens
        assert resolved_input_tokens is not None
        assert resolved_output_tokens is not None

        if resolved_profile == "shared_prefix":
            requests = _build_shared_prefix_requests(
                seed=seed,
                request_count=request_count,
                target_input_tokens=resolved_input_tokens,
                requested_output_tokens=resolved_output_tokens,
                shared_prefix_ratio=resolved_ratio,
                estimator=estimator,
            )
        else:
            requests = _build_simple_requests(
                profile=resolved_profile,
                seed=seed,
                request_count=request_count,
                target_input_tokens=resolved_input_tokens,
                requested_output_tokens=resolved_output_tokens,
                estimator=estimator,
            )

    configuration = GeneratorConfiguration(
        target_input_tokens=resolved_input_tokens,
        requested_output_tokens=resolved_output_tokens,
        shared_prefix_ratio=resolved_ratio,
        token_estimator=estimator.describe(),
    )

    return GeneratedWorkload(
        profile=resolved_profile,
        seed=seed,
        request_count=request_count,
        generator_configuration=configuration,
        requests=requests,
    )
