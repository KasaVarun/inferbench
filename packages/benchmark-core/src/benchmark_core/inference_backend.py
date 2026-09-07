"""The reusable local-inference backend abstraction (Phase 4).

This module defines the backend *contract* only -- no PyTorch, no
transformers, and no model loading happens here. Concrete backends (e.g.
`benchmark_core.local_transformers_backend.LocalTransformersBackend`)
implement `InferenceBackend`; test code can implement it too, with a fake,
to exercise orchestration (`benchmark_core.workload_runner`) without ever
touching a real model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

__all__ = ["BackendInfo", "GenerationRequest", "GenerationResponse", "InferenceBackend"]


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """A single generation call: a prompt and how many new tokens to produce."""

    prompt: str
    max_new_tokens: int

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be > 0, got {self.max_new_tokens}")


@dataclass(frozen=True, slots=True)
class GenerationResponse:
    """The outcome of one `InferenceBackend.generate()` call.

    `latency_ms` is measured by the backend itself (not the caller),
    since only the backend knows what device synchronization is needed
    for an honest timing (see `local_transformers_backend` for the MPS
    case). Token counts come from the model's real tokenizer, never from
    the Phase 3 character-based estimate.
    """

    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


@dataclass(frozen=True, slots=True)
class BackendInfo:
    """Identifying metadata about a loaded backend, for benchmark provenance."""

    model_name: str
    device: str
    dtype: str
    framework_version: str


class InferenceBackend(ABC):
    """Contract every local inference backend must implement.

    Lifecycle: construct (cheap, no I/O) -> `load()` (explicit, may be
    slow/network-dependent) -> any number of `generate()` calls -> `close()`
    (explicit cleanup). `health()` reflects whether the backend is
    currently loaded and ready to `generate()`.

    Implementations must not perform any model loading or network access
    as a side effect of construction or of importing their module --
    only `load()` may do that.
    """

    @abstractmethod
    def load(self) -> None:
        """Load the model/tokenizer. Raises on failure; must be explicit and clear."""

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a completion for `request`. Requires `load()` to have succeeded."""

    @abstractmethod
    def health(self) -> bool:
        """Return whether this backend is currently loaded and ready to generate."""

    @abstractmethod
    def close(self) -> None:
        """Release any loaded resources. Safe to call even if `load()` was never called."""

    @abstractmethod
    def info(self) -> BackendInfo:
        """Return this backend's identifying metadata. Requires `load()` to have succeeded."""
