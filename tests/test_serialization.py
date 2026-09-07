"""Tests for JSON/YAML workload serialization and atomic writing (Phase 3)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from benchmark_core import GeneratedWorkload, generate_workload, load_workload, write_workload
from benchmark_core.serialization import SUPPORTED_EXTENSIONS, workload_to_bytes


@pytest.fixture
def sample_workload() -> GeneratedWorkload:
    return generate_workload("shared_prefix", request_count=6, seed=42)


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------


def test_json_round_trip_through_generated_workload(
    tmp_path: Path, sample_workload: GeneratedWorkload
) -> None:
    output = tmp_path / "workload.json"
    write_workload(sample_workload, output)

    restored = load_workload(output)
    assert restored == sample_workload


def test_yaml_round_trip_through_generated_workload(
    tmp_path: Path, sample_workload: GeneratedWorkload
) -> None:
    output = tmp_path / "workload.yaml"
    write_workload(sample_workload, output)

    restored = load_workload(output)
    assert restored == sample_workload


def test_yml_extension_is_also_supported(
    tmp_path: Path, sample_workload: GeneratedWorkload
) -> None:
    output = tmp_path / "workload.yml"
    write_workload(sample_workload, output)
    assert load_workload(output) == sample_workload


# ---------------------------------------------------------------------------
# Byte-for-byte determinism
# ---------------------------------------------------------------------------


def test_json_serialization_is_byte_for_byte_deterministic(tmp_path: Path) -> None:
    first_workload = generate_workload("shared_prefix", request_count=10, seed=42)
    second_workload = generate_workload("shared_prefix", request_count=10, seed=42)

    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    write_workload(first_workload, first_path)
    write_workload(second_workload, second_path)

    assert first_path.read_bytes() == second_path.read_bytes()


def test_yaml_serialization_is_byte_for_byte_deterministic(tmp_path: Path) -> None:
    first_workload = generate_workload("mixed_workload", request_count=10, seed=42)
    second_workload = generate_workload("mixed_workload", request_count=10, seed=42)

    first_path = tmp_path / "first.yaml"
    second_path = tmp_path / "second.yaml"
    write_workload(first_workload, first_path)
    write_workload(second_workload, second_path)

    assert first_path.read_bytes() == second_path.read_bytes()


def test_serialized_output_has_no_volatile_metadata(tmp_path: Path) -> None:
    workload = generate_workload("shared_prefix", request_count=3, seed=42)
    output = tmp_path / "workload.json"
    write_workload(workload, output)

    text = output.read_text(encoding="utf-8")
    # No run_id/created_at-style volatile fields exist on GeneratedWorkload
    # at all; this asserts none crept in.
    assert "created_at" not in text
    assert "run_id" not in text
    assert "timestamp" not in text


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


def test_json_output_is_utf8_with_trailing_newline(
    tmp_path: Path, sample_workload: GeneratedWorkload
) -> None:
    output = tmp_path / "workload.json"
    write_workload(sample_workload, output)
    raw = output.read_bytes()
    raw.decode("utf-8")  # must not raise
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")


def test_yaml_output_is_utf8_with_trailing_newline(
    tmp_path: Path, sample_workload: GeneratedWorkload
) -> None:
    output = tmp_path / "workload.yaml"
    write_workload(sample_workload, output)
    raw = output.read_bytes()
    raw.decode("utf-8")
    assert raw.endswith(b"\n")


def test_json_output_is_indented_and_human_readable(
    tmp_path: Path, sample_workload: GeneratedWorkload
) -> None:
    output = tmp_path / "workload.json"
    write_workload(sample_workload, output)
    text = output.read_text(encoding="utf-8")
    assert '"schema_version": 1' in text
    assert text.startswith("{\n")


# ---------------------------------------------------------------------------
# Unknown extension
# ---------------------------------------------------------------------------


def test_unknown_extension_fails_clearly_and_writes_nothing(
    tmp_path: Path, sample_workload: GeneratedWorkload
) -> None:
    output = tmp_path / "workload.txt"
    with pytest.raises(ValueError, match="unsupported output extension"):
        write_workload(sample_workload, output)
    assert not output.exists()


def test_load_workload_rejects_unknown_extension(tmp_path: Path) -> None:
    output = tmp_path / "workload.txt"
    output.write_text("irrelevant", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported input extension"):
        load_workload(output)


def test_workload_to_bytes_rejects_unknown_suffix(sample_workload: GeneratedWorkload) -> None:
    with pytest.raises(ValueError, match="unsupported output extension"):
        workload_to_bytes(sample_workload, ".txt")


def test_supported_extensions_are_exactly_json_yaml_yml() -> None:
    assert SUPPORTED_EXTENSIONS == (".json", ".yaml", ".yml")


# ---------------------------------------------------------------------------
# Atomic writing / parent directory creation / failure safety
# ---------------------------------------------------------------------------


def test_write_workload_creates_missing_parent_directories(
    tmp_path: Path, sample_workload: GeneratedWorkload
) -> None:
    output = tmp_path / "nested" / "deep" / "workload.json"
    assert not output.parent.exists()

    write_workload(sample_workload, output)

    assert output.exists()


def test_write_workload_leaves_no_leftover_temp_files(
    tmp_path: Path, sample_workload: GeneratedWorkload
) -> None:
    output = tmp_path / "workload.json"
    write_workload(sample_workload, output)
    assert list(tmp_path.glob(".*.tmp")) == []


def test_existing_output_preserved_after_simulated_write_failure(
    tmp_path: Path, sample_workload: GeneratedWorkload, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "workload.json"
    original_content = '{"already": "here"}'
    output.write_text(original_content, encoding="utf-8")

    def _failing_fsync(_fd: int) -> None:
        raise OSError("simulated disk failure during fsync")

    monkeypatch.setattr(os, "fsync", _failing_fsync)

    with pytest.raises(OSError, match="simulated disk failure"):
        write_workload(sample_workload, output)

    assert output.read_text(encoding="utf-8") == original_content
    assert list(tmp_path.glob(".*.tmp")) == []


def test_existing_output_preserved_when_serialization_itself_fails(
    tmp_path: Path, sample_workload: GeneratedWorkload
) -> None:
    output = tmp_path / "workload.txt"
    original_content = "pre-existing, should remain untouched"
    output.write_text(original_content, encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported output extension"):
        write_workload(sample_workload, output)

    assert output.read_text(encoding="utf-8") == original_content
