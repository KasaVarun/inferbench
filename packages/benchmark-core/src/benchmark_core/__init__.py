"""benchmark-core: shared Python building blocks for InferBench.

Phase 1: defines the typed benchmark domain model (Pydantic v2). No
benchmark execution, inference, or measurement logic lives here -- this
package only defines what a benchmark request/result *is*.
"""

from benchmark_core.models import (
    BenchmarkConfiguration,
    BenchmarkRequest,
    BenchmarkResult,
    CostMetrics,
    GPUInfo,
    LatencyMetrics,
    MemoryMetrics,
    ThroughputMetrics,
    WorkloadConfiguration,
)

__version__ = "0.1.0"

__all__ = [
    "BenchmarkConfiguration",
    "BenchmarkRequest",
    "BenchmarkResult",
    "CostMetrics",
    "GPUInfo",
    "LatencyMetrics",
    "MemoryMetrics",
    "ThroughputMetrics",
    "WorkloadConfiguration",
    "__version__",
    "get_version",
]


def get_version() -> str:
    """Return the current benchmark-core package version."""
    return __version__
