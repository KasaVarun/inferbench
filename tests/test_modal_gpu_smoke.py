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
