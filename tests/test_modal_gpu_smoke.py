"""Local-only tests for the Modal CUDA smoke script.

These tests import the script definition only. They never call ``.remote()``,
authenticate, contact Modal, initialize CUDA, or require NVIDIA hardware.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _load_gpu_smoke() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "infra" / "modal" / "gpu_smoke.py"
    spec = importlib.util.spec_from_file_location("inferbench_gpu_smoke", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gpu_smoke() -> ModuleType:
    return _load_gpu_smoke()


def test_gpu_smoke_defines_one_shot_app_without_web_endpoints(gpu_smoke: ModuleType) -> None:
    assert gpu_smoke.app.name == "inferbench-cuda-smoke"
    assert hasattr(gpu_smoke, "run_cuda_benchmark")
    assert hasattr(gpu_smoke, "main")
    source = Path(__file__).resolve().parents[1] / "infra" / "modal" / "gpu_smoke.py"
    text = source.read_text(encoding="utf-8")
    assert "@app.function" in text
    assert "@app.local_entrypoint" in text
    assert "single_use_containers=True" in text
    assert "fastapi_endpoint" not in text
    assert "web_endpoint" not in text
    assert "asgi_app" not in text
    assert "web_server" not in text
    assert "app.deploy" not in text


def test_modal_is_authenticated_is_false_without_credentials(
    gpu_smoke: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "modal.config.config.get",
        lambda key, **_kwargs: None,
    )
    assert gpu_smoke.modal_is_authenticated() is False


def test_modal_is_authenticated_is_true_when_token_pair_is_present(
    gpu_smoke: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = {"token_id": "ak-test", "token_secret": "as-test"}
    monkeypatch.setattr(
        "modal.config.config.get",
        lambda key, **_kwargs: values.get(key),
    )
    assert gpu_smoke.modal_is_authenticated() is True


def test_modal_is_authenticated_is_true_for_oauth_refresh_token(
    gpu_smoke: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = {"oauth_refresh_token": "refresh-token"}
    monkeypatch.setattr(
        "modal.config.config.get",
        lambda key, **_kwargs: values.get(key),
    )
    assert gpu_smoke.modal_is_authenticated() is True


def test_auth_help_mentions_supported_setup_commands_and_not_secrets(
    gpu_smoke: ModuleType,
) -> None:
    assert "modal setup" in gpu_smoke._AUTH_HELP
    assert "token new" in gpu_smoke._AUTH_HELP
    assert "ak-" not in gpu_smoke._AUTH_HELP
    assert "as-" not in gpu_smoke._AUTH_HELP
    assert "token_secret" not in gpu_smoke._AUTH_HELP


def _gpu_smoke_source() -> str:
    return (Path(__file__).resolve().parents[1] / "infra" / "modal" / "gpu_smoke.py").read_text(
        encoding="utf-8"
    )


def test_gpu_smoke_does_not_derive_repository_root_from_file_parents() -> None:
    """Modal mounts this script as `/root/gpu_smoke.py`.

    `Path('/root/gpu_smoke.py').parents[2]` raises ``IndexError`` because that
    path has only `/root` and `/`. Packaging must not assume repository depth.
    """
    source = _gpu_smoke_source()
    assert "parents[2]" not in source
    assert "REPOSITORY_ROOT" not in source
    assert "BENCHMARK_CORE_SOURCE" not in source
    assert 'add_local_python_source("benchmark_core"' in source

    shallow = Path("/root/gpu_smoke.py")
    with pytest.raises(IndexError):
        _ = shallow.parents[2]


def test_gpu_smoke_imports_when_copied_away_from_the_repository_layout(
    tmp_path: Path,
) -> None:
    """Import must succeed even if the file is not under infra/modal/."""
    dest = tmp_path / "gpu_smoke.py"
    dest.write_text(_gpu_smoke_source(), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("inferbench_gpu_smoke_copied", dest)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.app.name == "inferbench-cuda-smoke"
    assert not hasattr(module, "REPOSITORY_ROOT")


def test_benchmark_core_is_resolved_by_installed_package_not_repo_path(
    gpu_smoke: ModuleType,
) -> None:
    import benchmark_core

    assert benchmark_core.__file__ is not None
    assert "add_local_python_source" in _gpu_smoke_source()
    assert gpu_smoke.image is not None
