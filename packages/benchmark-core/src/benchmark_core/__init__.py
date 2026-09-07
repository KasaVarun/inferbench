"""benchmark-core: shared Python building blocks for InferBench.

Phase 1 defined the typed benchmark domain model (Pydantic v2).
Phase 2 added a local, CPU-only benchmark execution engine
(`BenchmarkRunner`) and a deterministic NumPy workload used to validate
it end-to-end.
Phase 3 adds deterministic synthetic LLM workload generation
(`generate_workload`) and JSON/YAML serialization for the resulting
artifacts -- still no GPU, CUDA, inference, or network access anywhere in
this package.
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
from benchmark_core.serialization import SUPPORTED_EXTENSIONS, load_workload, write_workload
from benchmark_core.token_estimation import (
    DEFAULT_TOKEN_ESTIMATOR,
    CharacterRatioTokenEstimator,
    TokenEstimator,
)
from benchmark_core.workload_generation import (
    SUPPORTED_PROFILES,
    GeneratedRequest,
    GeneratedWorkload,
    GeneratorConfiguration,
    WorkloadProfileDefinition,
    generate_workload,
)
from benchmark_core.workloads import MatmulWorkload, make_matmul_workload

__version__ = "0.3.0"

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "SUPPORTED_PROFILES",
    "BenchmarkConfiguration",
    "BenchmarkExecutionError",
    "BenchmarkRequest",
    "BenchmarkResult",
    "BenchmarkRunner",
    "CharacterRatioTokenEstimator",
    "CostMetrics",
    "DEFAULT_TOKEN_ESTIMATOR",
    "ExecutionMetadata",
    "GPUInfo",
    "GeneratedRequest",
    "GeneratedWorkload",
    "GeneratorConfiguration",
    "LatencyMetrics",
    "MatmulWorkload",
    "MeasurementCollector",
    "MemoryMetrics",
    "ThroughputMetrics",
    "TokenEstimator",
    "WarmupConfig",
    "WorkloadConfiguration",
    "WorkloadProfileDefinition",
    "__version__",
    "generate_workload",
    "get_version",
    "load_workload",
    "make_matmul_workload",
    "write_workload",
]


def get_version() -> str:
    """Return the current benchmark-core package version."""
    return __version__
