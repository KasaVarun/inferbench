"""Tests for the deterministic, local token estimation abstraction (Phase 3)."""

from __future__ import annotations

import pytest

from benchmark_core import DEFAULT_TOKEN_ESTIMATOR, CharacterRatioTokenEstimator


def test_default_estimator_is_character_ratio() -> None:
    assert isinstance(DEFAULT_TOKEN_ESTIMATOR, CharacterRatioTokenEstimator)


def test_estimate_of_empty_string_is_zero() -> None:
    estimator = CharacterRatioTokenEstimator()
    assert estimator.estimate("") == 0


def test_estimate_is_non_negative() -> None:
    estimator = CharacterRatioTokenEstimator()
    assert estimator.estimate("hello world") >= 0


def test_estimate_matches_documented_ceil_formula() -> None:
    estimator = CharacterRatioTokenEstimator(chars_per_token=4.0)
    text = "a" * 17
    # ceil(17 / 4) == 5
    assert estimator.estimate(text) == 5


def test_same_text_always_gives_same_estimate() -> None:
    estimator = CharacterRatioTokenEstimator()
    text = "the quick brown fox jumps over the lazy dog"
    assert estimator.estimate(text) == estimator.estimate(text)


def test_longer_text_generally_produces_larger_estimate() -> None:
    estimator = CharacterRatioTokenEstimator()
    short_text = "short text"
    long_text = short_text * 20
    assert estimator.estimate(long_text) > estimator.estimate(short_text)


def test_estimate_does_not_equate_word_count_with_token_count() -> None:
    estimator = CharacterRatioTokenEstimator(chars_per_token=4.0)
    # 5 words but very different character lengths -> different estimates,
    # proving the estimate tracks characters, not word count.
    short_words = "a b c d e"
    long_words = "aaaaa bbbbb ccccc ddddd eeeee"
    assert estimator.estimate(short_words) != estimator.estimate(long_words)


def test_chars_per_token_property_exposes_configured_ratio() -> None:
    estimator = CharacterRatioTokenEstimator(chars_per_token=5.0)
    assert estimator.chars_per_token == 5.0


def test_rejects_non_positive_chars_per_token() -> None:
    with pytest.raises(ValueError, match="chars_per_token must be > 0"):
        CharacterRatioTokenEstimator(chars_per_token=0)


def test_describe_reports_method_and_configuration() -> None:
    estimator = CharacterRatioTokenEstimator(chars_per_token=4.0)
    description = estimator.describe()
    assert "character_ratio" in description
    assert "4.0" in description
