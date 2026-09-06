"""Tests for the deterministic CPU matmul workload (Phase 2)."""

from __future__ import annotations

import numpy as np
import pytest

from benchmark_core import make_matmul_workload


def test_matmul_workload_generates_expected_shape_and_dtype() -> None:
    workload = make_matmul_workload(matrix_size=8, seed=42)
    left, right = workload.inputs
    assert left.shape == (8, 8)
    assert right.shape == (8, 8)
    assert left.dtype == np.float32
    assert right.dtype == np.float32


def test_matmul_workload_is_deterministic_for_same_seed() -> None:
    first = make_matmul_workload(matrix_size=16, seed=7)
    second = make_matmul_workload(matrix_size=16, seed=7)

    assert np.array_equal(first.inputs[0], second.inputs[0])
    assert np.array_equal(first.inputs[1], second.inputs[1])


def test_matmul_workload_differs_for_different_seeds() -> None:
    first = make_matmul_workload(matrix_size=16, seed=1)
    second = make_matmul_workload(matrix_size=16, seed=2)

    assert not np.array_equal(first.inputs[0], second.inputs[0])


def test_matmul_workload_is_callable_and_returns_none() -> None:
    workload = make_matmul_workload(matrix_size=4, seed=0)
    assert workload() is None


def test_matmul_workload_calls_do_not_mutate_inputs() -> None:
    workload = make_matmul_workload(matrix_size=8, seed=42)
    left_before = workload.inputs[0].copy()
    right_before = workload.inputs[1].copy()

    workload()
    workload()

    assert np.array_equal(workload.inputs[0], left_before)
    assert np.array_equal(workload.inputs[1], right_before)


@pytest.mark.parametrize("matrix_size", [0, -1, -100])
def test_matmul_workload_rejects_non_positive_matrix_size(matrix_size: int) -> None:
    with pytest.raises(ValueError, match="matrix_size must be > 0"):
        make_matmul_workload(matrix_size=matrix_size, seed=42)
