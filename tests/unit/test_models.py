from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

from distill_run.errors import ModelFetchError
from distill_run.models import ModelSpec, resolve_model


def test_local_model_needs_no_download(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    assert resolve_model(ModelSpec(str(model)), tmp_path / "cache") == model


def test_missing_local_model_reports_model_error(tmp_path: Path):
    with pytest.raises(ModelFetchError, match="model path does not exist"):
        resolve_model(ModelSpec(str(tmp_path / "missing")), tmp_path / "cache")


def test_unknown_model_scheme_is_rejected(tmp_path: Path):
    with pytest.raises(ModelFetchError, match="unsupported model URI scheme"):
        resolve_model(ModelSpec("https://example.com/model"), tmp_path / "cache")


def test_huggingface_model_download_then_cache_hit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls = []
    module = ModuleType("huggingface_hub")

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        target = Path(kwargs["local_dir"])
        (target / "config.json").write_text("{}", encoding="utf-8")
        return str(target)

    module.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)

    spec = ModelSpec("hf://org/student", revision="abc123")
    first = resolve_model(spec, tmp_path / "models")
    second = resolve_model(spec, tmp_path / "models")

    assert first == second
    assert (first / "config.json").is_file()
    assert len(calls) == 1
    assert calls[0]["repo_type"] == "model"
    assert calls[0]["revision"] == "abc123"


def test_modelscope_model_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    modelscope = ModuleType("modelscope")
    hub = ModuleType("modelscope.hub")
    download = ModuleType("modelscope.hub.snapshot_download")

    def snapshot_download(repo_id, **kwargs):
        target = Path(kwargs["cache_dir"]) / repo_id.replace("/", "--")
        target.mkdir(parents=True)
        (target / "config.json").write_text("{}", encoding="utf-8")
        return str(target)

    download.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "modelscope", modelscope)
    monkeypatch.setitem(sys.modules, "modelscope.hub", hub)
    monkeypatch.setitem(sys.modules, "modelscope.hub.snapshot_download", download)

    result = resolve_model(ModelSpec("ms://org/student"), tmp_path / "models")
    assert (result / "config.json").is_file()


def test_model_cache_key_separates_revisions():
    assert (
        ModelSpec("hf://org/model").cache_key()
        != ModelSpec("hf://org/model", revision="v2").cache_key()
    )
