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
Phase 5 adds CUDA benchmark configuration, environment, aggregation, and
result models used by one-shot Modal GPU jobs. Modal itself remains an
infra-only dependency and is never imported by this package.
"""

from typing import TYPE_CHECKING, Any

from benchmark_core.cuda_benchmark import (
    CudaBenchmarkConfiguration,
    CudaBenchmarkReport,
    CudaDtype,
    CudaEnvironment,
    ModalGPUType,
    build_cuda_benchmark_report,
    build_cuda_failure_result,
    bytes_to_mib,
    cuda_correctness_tolerances,
    latency_percentiles,
    parse_cuda_dtype,
    parse_modal_gpu_type,
)
from benchmark_core.inference_backend import (
    BackendInfo,
    GenerationRequest,
    GenerationResponse,
    InferenceBackend,
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

if TYPE_CHECKING:
    from benchmark_core.local_transformers_backend import LocalTransformersBackend

__version__ = "0.5.0"

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
    "CudaBenchmarkConfiguration",
    "CudaBenchmarkReport",
    "CudaDtype",
    "CudaEnvironment",
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
    "ModalGPUType",
    "ThroughputMetrics",
    "TokenEstimator",
    "WarmupConfig",
    "WorkloadConfiguration",
    "WorkloadProfileDefinition",
    "__version__",
    "build_cuda_benchmark_report",
    "build_cuda_failure_result",
    "bytes_to_mib",
    "cuda_correctness_tolerances",
    "generate_workload",
    "get_version",
    "is_mps_available",
    "latency_percentiles",
    "load_workload",
    "make_matmul_workload",
    "parse_cuda_dtype",
    "parse_modal_gpu_type",
    "resolve_device",
    "run_workload",
    "write_workload",
]


def __getattr__(name: str) -> Any:
    """Lazily expose the PyTorch/transformers backend at the package root.

    The Phase 5 remote image needs the shared result models but not
    ``transformers``. Lazy loading preserves the existing public imports
    without forcing that unrelated dependency into the CUDA image.
    """
    if name == "LocalTransformersBackend":
        from benchmark_core.local_transformers_backend import LocalTransformersBackend

        return LocalTransformersBackend
    if name == "is_mps_available":
        from benchmark_core.local_transformers_backend import is_mps_available

        return is_mps_available
    if name == "resolve_device":
        from benchmark_core.local_transformers_backend import resolve_device

        return resolve_device
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_version() -> str:
    """Return the current benchmark-core package version."""
    return __version__
