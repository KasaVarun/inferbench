"""benchmark-core: shared Python building blocks for InferBench.

Phase 1 defined the typed benchmark domain model (Pydantic v2).
Phase 2 adds a local, CPU-only benchmark execution engine
(`BenchmarkRunner`) and a deterministic NumPy workload used to validate
it end-to-end -- no GPU, CUDA, or inference logic lives here yet.
"""

from benchmark_core.models import (
    BenchmarkConfiguration,
    BenchmarkRequest,
    BenchmarkResult,
    CostMetrics,
    ExecutionMetadata,
    GPUInfo,
    LatencyMetrics,
    MemoryMetrics,
    ThroughputMetrics,
    WorkloadConfiguration,
)
from benchmark_core.runner import (
    BenchmarkExecutionError,
    BenchmarkRunner,
    MeasurementCollector,
    WarmupConfig,
)
from benchmark_core.workloads import MatmulWorkload, make_matmul_workload

__version__ = "0.2.0"

__all__ = [
    "BenchmarkConfiguration",
    "BenchmarkExecutionError",
    "BenchmarkRequest",
    "BenchmarkResult",
    "BenchmarkRunner",
    "CostMetrics",
    "ExecutionMetadata",
    "GPUInfo",
    "LatencyMetrics",
    "MatmulWorkload",
    "MeasurementCollector",
    "MemoryMetrics",
    "ThroughputMetrics",
    "WarmupConfig",
    "WorkloadConfiguration",
    "__version__",
    "get_version",
    "make_matmul_workload",
]


def get_version() -> str:
    """Return the current benchmark-core package version."""
    return __version__
