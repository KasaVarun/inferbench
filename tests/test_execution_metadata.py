"""Tests for `ExecutionMetadata` and its integration into `BenchmarkResult` (Phase 2)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from benchmark_core import BenchmarkResult, ExecutionMetadata
from fixtures import make_execution_metadata, make_successful_benchmark_result


def test_valid_execution_metadata() -> None:
    metadata = make_execution_metadata()
    assert metadata.warmup_iterations == 3
    assert metadata.measured_iterations == 10
    assert (
        metadata.successful_iterations + metadata.failed_iterations == metadata.measured_iterations
    )


def test_execution_metadata_inconsistent_counts_are_rejected() -> None:
    with pytest.raises(ValidationError):
        make_execution_metadata(
            successful_iterations=5, failed_iterations=3, measured_iterations=10
        )


def test_execution_metadata_negative_warmup_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_execution_metadata(warmup_iterations=-1)


def test_execution_metadata_non_positive_measured_iterations_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_execution_metadata(measured_iterations=0, successful_iterations=0, failed_iterations=0)


def test_execution_metadata_negative_elapsed_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_execution_metadata(total_elapsed_seconds=-0.1)


def test_benchmark_result_execution_defaults_to_none() -> None:
    result = make_successful_benchmark_result()
    assert result.execution is None


def test_benchmark_result_with_execution_metadata_round_trips() -> None:
    original = make_successful_benchmark_result(execution=make_execution_metadata())
    restored = BenchmarkResult.model_validate_json(original.model_dump_json())
    assert restored == original
    assert isinstance(restored.execution, ExecutionMetadata)
