"""Tests for deterministic synthetic LLM workload generation (Phase 3).

No network access, no real tokenizer, and no model is ever touched here.
Exact natural-language prose is deliberately never asserted (only length,
structure, determinism, and category behavior) since the specific
generated words are an implementation detail, not the contract.
"""

from __future__ import annotations

import pytest

from benchmark_core import (
    SUPPORTED_PROFILES,
    GeneratedRequest,
    GeneratedWorkload,
    GeneratorConfiguration,
    generate_workload,
)
from benchmark_core.workload_generation import MIXED_WORKLOAD_CATEGORIES

# ---------------------------------------------------------------------------
# Profile tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("profile", SUPPORTED_PROFILES)
def test_all_profiles_generate_successfully(profile: str) -> None:
    workload = generate_workload(profile, request_count=8, seed=42)
    assert workload.profile == profile
    assert workload.request_count == 8
    assert len(workload.requests) == 8


@pytest.mark.parametrize("profile", SUPPORTED_PROFILES)
def test_correct_request_count_for_various_sizes(profile: str) -> None:
    for count in (1, 3, 17):
        workload = generate_workload(profile, request_count=count, seed=1)
        assert len(workload.requests) == count
        assert workload.request_count == count


@pytest.mark.parametrize("profile", SUPPORTED_PROFILES)
def test_every_request_has_unique_deterministic_request_id(profile: str) -> None:
    workload = generate_workload(profile, request_count=25, seed=7)
    ids = [request.request_id for request in workload.requests]
    assert len(ids) == len(set(ids))
    assert all(request_id.startswith("req-") for request_id in ids)


@pytest.mark.parametrize("profile", SUPPORTED_PROFILES)
def test_same_seed_regenerates_identical_request_ids(profile: str) -> None:
    first = generate_workload(profile, request_count=10, seed=123)
    second = generate_workload(profile, request_count=10, seed=123)
    assert [r.request_id for r in first.requests] == [r.request_id for r in second.requests]
    assert [r.prompt for r in first.requests] == [r.prompt for r in second.requests]


@pytest.mark.parametrize("profile", SUPPORTED_PROFILES)
def test_different_seed_changes_generated_content(profile: str) -> None:
    first = generate_workload(profile, request_count=10, seed=1)
    second = generate_workload(profile, request_count=10, seed=2)
    assert [r.prompt for r in first.requests] != [r.prompt for r in second.requests]
    assert [r.request_id for r in first.requests] != [r.request_id for r in second.requests]


@pytest.mark.parametrize("profile", SUPPORTED_PROFILES)
def test_workload_artifact_declares_estimated_tokens(profile: str) -> None:
    workload = generate_workload(profile, request_count=3, seed=42)
    assert workload.token_counts_estimated is True


def test_unsupported_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported profile"):
        generate_workload("not_a_real_profile", request_count=5, seed=1)


@pytest.mark.parametrize("request_count", [0, -1, -10])
def test_non_positive_request_count_is_rejected(request_count: int) -> None:
    with pytest.raises(ValueError, match="request_count must be > 0"):
        generate_workload("shared_prefix", request_count=request_count, seed=1)


def test_generated_workload_rejects_request_count_mismatch() -> None:
    workload = generate_workload("shared_prefix", request_count=3, seed=1)
    with pytest.raises(ValueError, match="but got"):
        GeneratedWorkload(
            profile="shared_prefix",
            seed=1,
            request_count=5,
            generator_configuration=workload.generator_configuration,
            requests=workload.requests,
        )


def test_generator_configuration_and_generated_request_are_directly_constructible() -> None:
    # Exercises the models directly (not just via generate_workload), as
    # a lightweight structural/typing sanity check.
    configuration = GeneratorConfiguration(
        target_input_tokens=64,
        requested_output_tokens=32,
        shared_prefix_ratio=0.0,
        token_estimator="character_ratio(chars_per_token=4.0)",
    )
    request = GeneratedRequest(
        request_id="req-0000000000000000",
        index=0,
        prompt="hello world",
        requested_output_tokens=32,
        estimated_input_tokens=3,
        profile="short_prompt_short_output",
    )
    assert configuration.target_input_tokens == 64
    assert request.shares_common_prefix is False
    assert request.category is None


# ---------------------------------------------------------------------------
# Short/long behavior
# ---------------------------------------------------------------------------


def test_short_profiles_have_much_shorter_estimated_input_than_long_profiles() -> None:
    short_workload = generate_workload("short_prompt_short_output", request_count=5, seed=42)
    long_workload = generate_workload("long_prompt_short_output", request_count=5, seed=42)

    short_avg = sum(r.estimated_input_tokens for r in short_workload.requests) / 5
    long_avg = sum(r.estimated_input_tokens for r in long_workload.requests) / 5

    assert long_avg > short_avg * 10


def test_output_token_targets_match_profile_defaults() -> None:
    workload = generate_workload("short_prompt_long_output", request_count=4, seed=42)
    assert all(r.requested_output_tokens == 512 for r in workload.requests)

    workload = generate_workload("long_prompt_short_output", request_count=4, seed=42)
    assert all(r.requested_output_tokens == 32 for r in workload.requests)


def test_output_token_target_can_be_overridden() -> None:
    workload = generate_workload(
        "short_prompt_short_output", request_count=3, seed=42, requested_output_tokens=99
    )
    assert all(r.requested_output_tokens == 99 for r in workload.requests)
    assert workload.generator_configuration.requested_output_tokens == 99


def test_input_token_target_can_be_overridden() -> None:
    baseline = generate_workload("short_prompt_short_output", request_count=3, seed=42)
    overridden = generate_workload(
        "short_prompt_short_output", request_count=3, seed=42, target_input_tokens=256
    )
    assert (
        overridden.requests[0].estimated_input_tokens > baseline.requests[0].estimated_input_tokens
    )
    assert overridden.generator_configuration.target_input_tokens == 256


@pytest.mark.parametrize("bad_value", [0, -1])
def test_non_positive_token_target_overrides_are_rejected(bad_value: int) -> None:
    with pytest.raises(ValueError, match="target_input_tokens must be > 0"):
        generate_workload(
            "short_prompt_short_output", request_count=3, seed=1, target_input_tokens=bad_value
        )
    with pytest.raises(ValueError, match="requested_output_tokens must be > 0"):
        generate_workload(
            "short_prompt_short_output", request_count=3, seed=1, requested_output_tokens=bad_value
        )


def test_token_overrides_not_meaningful_for_mixed_workload_are_rejected() -> None:
    with pytest.raises(ValueError, match="not meaningful for mixed_workload"):
        generate_workload("mixed_workload", request_count=3, seed=1, target_input_tokens=64)
    with pytest.raises(ValueError, match="not meaningful for mixed_workload"):
        generate_workload("mixed_workload", request_count=3, seed=1, requested_output_tokens=64)


def test_shared_prefix_ratio_not_meaningful_outside_shared_prefix_profile() -> None:
    with pytest.raises(ValueError, match="only meaningful for the shared_prefix profile"):
        generate_workload(
            "short_prompt_short_output", request_count=3, seed=1, shared_prefix_ratio=0.5
        )


# ---------------------------------------------------------------------------
# Shared prefix
# ---------------------------------------------------------------------------


def test_shared_prefix_ratio_one_causes_all_requests_to_share_exact_prefix() -> None:
    workload = generate_workload(
        "shared_prefix", request_count=10, seed=42, shared_prefix_ratio=1.0
    )
    prompts = [r.prompt for r in workload.requests]
    shared_prefix_length = min(len(p) for p in prompts) // 2 or 1
    reference = prompts[0][:shared_prefix_length]
    assert all(p.startswith(reference) for p in prompts)
    assert all(r.shares_common_prefix for r in workload.requests)


def test_shared_prefix_ratio_half_produces_expected_deterministic_subset() -> None:
    workload = generate_workload(
        "shared_prefix", request_count=10, seed=42, shared_prefix_ratio=0.5
    )
    shared_flags = [r.shares_common_prefix for r in workload.requests]
    # Documented rule: the first round(request_count * ratio) requests, by
    # index, are the shared group.
    assert shared_flags == [True] * 5 + [False] * 5


@pytest.mark.parametrize("ratio", [-0.01, 1.01, -5.0, 2.0])
def test_invalid_shared_prefix_ratio_is_rejected(ratio: float) -> None:
    with pytest.raises(ValueError, match=r"shared_prefix_ratio must be within \[0\.0, 1\.0\]"):
        generate_workload("shared_prefix", request_count=5, seed=1, shared_prefix_ratio=ratio)


def test_shared_prefix_text_itself_is_deterministic() -> None:
    first = generate_workload("shared_prefix", request_count=4, seed=42, shared_prefix_ratio=1.0)
    second = generate_workload("shared_prefix", request_count=4, seed=42, shared_prefix_ratio=1.0)
    assert first.requests[0].prompt == second.requests[0].prompt


def test_shared_prefix_text_changes_with_seed() -> None:
    first = generate_workload("shared_prefix", request_count=4, seed=1, shared_prefix_ratio=1.0)
    second = generate_workload("shared_prefix", request_count=4, seed=2, shared_prefix_ratio=1.0)
    assert first.requests[0].prompt != second.requests[0].prompt


def test_non_shared_requests_do_not_accidentally_start_with_shared_prefix() -> None:
    workload = generate_workload(
        "shared_prefix", request_count=10, seed=42, shared_prefix_ratio=0.5
    )
    shared = [r for r in workload.requests if r.shares_common_prefix]
    non_shared = [r for r in workload.requests if not r.shares_common_prefix]
    assert shared and non_shared

    shared_reference = shared[0].prompt
    for request in non_shared:
        assert not request.prompt.startswith(shared_reference)


def test_shared_prefix_with_tiny_input_target_has_empty_suffix() -> None:
    # target_input_tokens=1 makes the shared-prefix-derived suffix target
    # exactly 0, exercising the "no remaining tokens" path: the shared
    # request's prompt is then exactly the shared prefix, no suffix words.
    workload = generate_workload(
        "shared_prefix",
        request_count=2,
        seed=42,
        target_input_tokens=1,
        shared_prefix_ratio=1.0,
    )
    prompts = {r.prompt for r in workload.requests}
    assert len(prompts) == 1


def test_all_shared_group_requests_share_exact_identical_prefix_bytes() -> None:
    workload = generate_workload("shared_prefix", request_count=6, seed=99, shared_prefix_ratio=1.0)
    prompts = [r.prompt for r in workload.requests]
    shortest_len = min(len(p) for p in prompts)
    # All shared requests are built as `<exact same prefix> + <unique suffix>`,
    # so every prompt must be byte-identical up to the shortest one's length
    # for at least the deterministic prefix portion.
    reference_prefix = prompts[0][: shortest_len // 2]
    assert all(p.startswith(reference_prefix) for p in prompts)


# ---------------------------------------------------------------------------
# Random prefix
# ---------------------------------------------------------------------------


def test_random_prefix_prompts_are_not_all_identical() -> None:
    workload = generate_workload("random_prefix", request_count=8, seed=42)
    prompts = {r.prompt for r in workload.requests}
    assert len(prompts) == 8


def test_random_prefix_identical_seed_regenerates_exact_same_prefixes() -> None:
    first = generate_workload("random_prefix", request_count=6, seed=55)
    second = generate_workload("random_prefix", request_count=6, seed=55)
    assert [r.prompt for r in first.requests] == [r.prompt for r in second.requests]


def test_random_prefix_different_seed_changes_prefixes() -> None:
    first = generate_workload("random_prefix", request_count=6, seed=1)
    second = generate_workload("random_prefix", request_count=6, seed=2)
    assert [r.prompt for r in first.requests] != [r.prompt for r in second.requests]


# ---------------------------------------------------------------------------
# Mixed workload
# ---------------------------------------------------------------------------


def test_mixed_workload_generation_is_deterministic() -> None:
    first = generate_workload("mixed_workload", request_count=12, seed=42)
    second = generate_workload("mixed_workload", request_count=12, seed=42)
    assert [(r.category, r.prompt) for r in first.requests] == [
        (r.category, r.prompt) for r in second.requests
    ]


def test_mixed_workload_contains_more_than_one_category_for_large_request_count() -> None:
    workload = generate_workload("mixed_workload", request_count=12, seed=42)
    categories = {r.category for r in workload.requests}
    assert len(categories) > 1


def test_mixed_workload_category_selection_is_round_robin_by_index() -> None:
    workload = generate_workload("mixed_workload", request_count=12, seed=42)
    expected_categories = [
        MIXED_WORKLOAD_CATEGORIES[index % len(MIXED_WORKLOAD_CATEGORIES)][0] for index in range(12)
    ]
    assert [r.category for r in workload.requests] == expected_categories


def test_mixed_workload_category_selection_is_independent_of_seed() -> None:
    first = generate_workload("mixed_workload", request_count=12, seed=1)
    second = generate_workload("mixed_workload", request_count=12, seed=999)
    assert [r.category for r in first.requests] == [r.category for r in second.requests]


def test_mixed_workload_category_token_targets_match_definition() -> None:
    workload = generate_workload("mixed_workload", request_count=4, seed=42)
    for request, (category, _input_tokens, output_tokens) in zip(
        workload.requests, MIXED_WORKLOAD_CATEGORIES, strict=True
    ):
        assert request.category == category
        assert request.requested_output_tokens == output_tokens
