"""benchmark-core: shared Python building blocks for InferBench.

Phase 0 placeholder package. No benchmarking functionality is implemented
yet; this module only proves that the package is importable and versioned.
"""

__version__ = "0.0.1"


def get_version() -> str:
    """Return the current benchmark-core package version."""
    return __version__
