"""Atomic, fsync'd file writing shared by every benchmark_core CLI command.

Phase 2's CLI introduced this exact pattern (write a temp file next to the
destination, fsync it, then `os.replace` it into place) for
`BenchmarkResult` JSON. Phase 3 needs the identical guarantee for workload
JSON/YAML artifacts, so the logic lives here once and both callers
(`benchmark_core.cli`, `benchmark_core.serialization`) share it rather than
duplicating it.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path

__all__ = ["write_atomic"]


def write_atomic(path: Path, payload: bytes) -> None:
    """Write `payload` bytes to `path` atomically, never corrupting an existing file.

    A temporary file in the same directory as `path` is written and fully
    fsynced before an atomic `os.replace` swaps it into place. If anything
    fails before that replace, the temporary file is removed and `path` is
    left completely untouched -- so a failed write can never leave a
    partial or corrupted result behind, nor overwrite a previously valid
    one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as tmp_file:
            tmp_file.write(payload)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(tmp_name, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise
