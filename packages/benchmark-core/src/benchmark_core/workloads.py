"""Deterministic, CPU-only workloads used to validate the benchmark harness.

Phase 2 exists to prove out ``BenchmarkRunner`` end-to-end without any
GPU/CUDA/inference dependency. The workload here is a plain NumPy matrix
multiplication -- it is not LLM inference, and its "size" is a matrix
dimension, not a token count. See :mod:`benchmark_core.cli` for how it is
mapped onto the (LLM-shaped) Phase 1 ``WorkloadConfiguration`` schema.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["MatmulWorkload", "make_matmul_workload"]


class MatmulWorkload:
    """A deterministic float32 matrix-multiplication workload.

    Two ``matrix_size x matrix_size`` float32 matrices are generated once,
    eagerly, in ``__init__`` from a seeded NumPy ``Generator`` -- *before*
    the instance is ever called. Calling the instance only performs
    ``numpy.matmul`` on those already-materialized matrices, so random
    generation and allocation never contaminate the timing of a call.
    Calling the same instance repeatedly always performs the exact same
    multiplication.

    ``inputs`` is exposed purely for testability, so tests can assert that
    construction is deterministic without duplicating NumPy RNG logic; the
    benchmark runner itself only ever calls the instance.
    """

    def __init__(self, matrix_size: int, seed: int) -> None:
        if matrix_size <= 0:
            raise ValueError(f"matrix_size must be > 0, got {matrix_size}")

        rng = np.random.default_rng(seed)
        left: NDArray[np.float32] = rng.standard_normal(
            (matrix_size, matrix_size), dtype=np.float32
        )
        right: NDArray[np.float32] = rng.standard_normal(
            (matrix_size, matrix_size), dtype=np.float32
        )
        self.inputs: tuple[NDArray[np.float32], NDArray[np.float32]] = (left, right)

    def __call__(self) -> None:
        left, right = self.inputs
        np.matmul(left, right)


def make_matmul_workload(matrix_size: int, seed: int) -> MatmulWorkload:
    """Build a deterministic float32 matrix-multiplication workload.

    Args:
        matrix_size: Side length of the (square) matrices to multiply. Must
            be > 0.
        seed: Seed for the NumPy random generator used to build the input
            matrices. The same ``(matrix_size, seed)`` pair always produces
            bit-identical input matrices.

    Raises:
        ValueError: If ``matrix_size`` is not positive.
    """
    return MatmulWorkload(matrix_size=matrix_size, seed=seed)
