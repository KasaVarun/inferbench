"""benchmark-core: shared Python building blocks for InferBench.

Phase 1 defined the typed benchmark domain model (Pydantic v2).
Phase 2 added a local, CPU-only benchmark execution engine
(`BenchmarkRunner`) and a deterministic NumPy workload used to validate
it end-to-end.
Phase 3 added deterministic synthetic LLM workload generation
(`generate_workload`) and JSON/YAML serialization for the resulting
artifacts.
Phase 4 adds the first real inference backend: `LocalTransformersBackend`
runs a small causal LM locally via PyTorch + `transformers`, on Apple MPS
(falling back to CPU) -- still no CUDA, no remote GPU, and no network
access except explicitly inside `load()`.
"""

from benchmark_core.inference_backend import (
    BackendInfo,
    GenerationRequest,
    GenerationResponse,
    InferenceBackend,
)
from benchmark_core.local_transformers_backend import (
    LocalTransformersBackend,
    is_mps_available,
    resolve_device,
)
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
from benchmark_core.workload_runner import run_workload
from benchmark_core.workloads import MatmulWorkload, make_matmul_workload

__version__ = "0.4.0"

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "SUPPORTED_PROFILES",
    "BackendInfo",
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
    "GenerationRequest",
    "GenerationResponse",
    "GeneratorConfiguration",
    "InferenceBackend",
    "LatencyMetrics",
    "LocalTransformersBackend",
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
    "is_mps_available",
    "load_workload",
    "make_matmul_workload",
    "resolve_device",
    "run_workload",
    "write_workload",
]


def get_version() -> str:
    """Return the current benchmark-core package version."""
    return __version__
