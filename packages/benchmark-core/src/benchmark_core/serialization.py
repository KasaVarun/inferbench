"""Deterministic JSON/YAML serialization for generated workload artifacts.

Output format is selected purely by file extension (`.json`, `.yaml`,
`.yml`); anything else fails clearly and immediately, before any file is
touched. Both formats are produced from the exact same JSON-mode dict
(`GeneratedWorkload.model_dump(mode="json")`), preserving field-declaration
order, so identical inputs always produce byte-for-byte identical output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from benchmark_core.atomic_io import write_atomic
from benchmark_core.workload_generation import GeneratedWorkload

__all__ = ["SUPPORTED_EXTENSIONS", "load_workload", "workload_to_bytes", "write_workload"]

SUPPORTED_EXTENSIONS = (".json", ".yaml", ".yml")

# A wide wrap width avoids PyYAML line-wrapping long synthetic prompt
# strings mid-word; the exact value only matters for readability, not
# determinism (the same width always wraps the same input identically).
_YAML_WRAP_WIDTH = 100_000


def _ensure_trailing_newline(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"


def workload_to_bytes(workload: GeneratedWorkload, suffix: str) -> bytes:
    """Serialize `workload` deterministically based on `suffix` (a file extension).

    Raises:
        ValueError: If `suffix` is not one of `SUPPORTED_EXTENSIONS`.
    """
    normalized_suffix = suffix.lower()
    data: dict[str, Any] = workload.model_dump(mode="json")

    if normalized_suffix == ".json":
        text = json.dumps(data, indent=2, sort_keys=False, ensure_ascii=False)
    elif normalized_suffix in (".yaml", ".yml"):
        text = yaml.safe_dump(
            data,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
            width=_YAML_WRAP_WIDTH,
        )
    else:
        raise ValueError(
            f"unsupported output extension '{suffix}'; expected one of {SUPPORTED_EXTENSIONS}"
        )

    return _ensure_trailing_newline(text).encode("utf-8")


def write_workload(workload: GeneratedWorkload, path: Path) -> None:
    """Serialize `workload` per `path`'s extension and write it atomically.

    Raises:
        ValueError: If `path`'s extension is unsupported. Nothing is
            written to disk in that case.
    """
    payload = workload_to_bytes(workload, path.suffix)
    write_atomic(path, payload)


def load_workload(path: Path) -> GeneratedWorkload:
    """Load a `GeneratedWorkload` previously written by `write_workload`.

    Raises:
        ValueError: If `path`'s extension is unsupported.
    """
    normalized_suffix = path.suffix.lower()
    raw_text = path.read_text(encoding="utf-8")

    if normalized_suffix == ".json":
        data = json.loads(raw_text)
    elif normalized_suffix in (".yaml", ".yml"):
        data = yaml.safe_load(raw_text)
    else:
        raise ValueError(
            f"unsupported input extension '{path.suffix}'; expected one of {SUPPORTED_EXTENSIONS}"
        )

    return GeneratedWorkload.model_validate(data)
