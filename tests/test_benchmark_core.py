"""Smoke test for the benchmark_core package (Phase 0)."""

from benchmark_core import __version__, get_version


def test_package_is_importable() -> None:
    assert get_version() == __version__


def test_version_is_a_string() -> None:
    assert isinstance(__version__, str)
    assert __version__ != ""
