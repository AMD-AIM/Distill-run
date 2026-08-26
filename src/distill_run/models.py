"""Resolve local or Hub model specifications before engine preflight."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from .catalog import model_error_hint
from .errors import ModelFetchError
from .utils import atomic_write_json, ensure_dir

logger = logging.getLogger(__name__)
LOCAL_SCHEMES = frozenset({"", "file"})
HUB_SCHEMES = frozenset({"hf", "ms", "modelscope"})

if TYPE_CHECKING:
    from .context import RunContext


@dataclass(frozen=True)
class ModelSpec:
    uri: str
    revision: str | None = None

    @property
    def scheme(self) -> str:
        parsed = urlparse(self.uri)
        return parsed.scheme.lower() if parsed.scheme else ""

    @property
    def is_local(self) -> bool:
        return self.scheme in LOCAL_SCHEMES

    def cache_key(self) -> str:
        material = f"{self.uri}|{self.revision or ''}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _fetch_huggingface(spec: ModelSpec, staging: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ModelFetchError(
            "hf:// models need huggingface_hub; install the distill-run hub dependencies"
        ) from exc

    repo_id = spec.uri[len("hf://") :].strip("/")
    if not repo_id:
        raise ModelFetchError(f"malformed Hugging Face model URI: {spec.uri}")
    logger.info(
        "downloading Hugging Face model %s revision=%s", repo_id, spec.revision or "default"
    )
    try:
        return Path(
            snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                revision=spec.revision,
                local_dir=str(staging),
            )
        )
    except Exception as exc:  # noqa: BLE001 - Hub SDKs expose many error types
        raise ModelFetchError(f"failed to fetch model {spec.uri}: {exc}") from exc


def _fetch_modelscope(spec: ModelSpec, staging: Path) -> Path:
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError as exc:
        raise ModelFetchError(
            "ms:// models need the modelscope SDK in the selected engine environment"
        ) from exc

    repo_id = spec.uri.split("://", 1)[1].strip("/")
    if not repo_id:
        raise ModelFetchError(f"malformed ModelScope model URI: {spec.uri}")
    logger.info("downloading ModelScope model %s revision=%s", repo_id, spec.revision or "default")
    try:
        return Path(
            snapshot_download(
                repo_id,
                revision=spec.revision,
                repo_type="model",
                cache_dir=str(staging),
            )
        )
    except Exception as exc:  # noqa: BLE001
        raise ModelFetchError(f"failed to fetch model {spec.uri}: {exc}") from exc


def _commit_download(spec: ModelSpec, staging: Path, payload: Path, cache_dir: Path) -> Path:
    payload_dir = cache_dir / spec.cache_key()
    if payload == staging:
        relative = Path()
    elif staging in payload.parents:
        relative = payload.relative_to(staging)
    else:
        target = staging / payload.name
        if payload.is_dir():
            shutil.copytree(payload, target, dirs_exist_ok=True)
        else:
            shutil.copy2(payload, target)
        relative = target.relative_to(staging)

    shutil.rmtree(payload_dir, ignore_errors=True)
    os.replace(staging, payload_dir)
    final = payload_dir / relative
    atomic_write_json(
        cache_dir / f"{spec.cache_key()}.done",
        {
            "uri": spec.uri,
            "revision": spec.revision,
            "path": str(final),
            "cached_at": time.time(),
        },
    )
    return final


def resolve_model(spec: ModelSpec, cache_dir: Path) -> Path:
    """Return a local model directory, downloading explicit Hub URIs once."""
    if spec.is_local:
        raw = spec.uri[len("file://") :] if spec.uri.startswith("file://") else spec.uri
        path = Path(raw).expanduser()
        if not path.exists():
            raise ModelFetchError(f"model path does not exist: {path}{model_error_hint()}")
        return path

    if spec.scheme not in HUB_SCHEMES:
        raise ModelFetchError(
            f"unsupported model URI scheme '{spec.scheme}' in {spec.uri}; "
            "supported: local path, hf://, ms://"
        )

    ensure_dir(cache_dir)
    marker = cache_dir / f"{spec.cache_key()}.done"
    if marker.is_file():
        try:
            cached = Path(json.loads(marker.read_text(encoding="utf-8"))["path"])
        except (OSError, ValueError, KeyError, TypeError):
            cached = None
        if cached is not None and cached.exists():
            logger.info("model %s -> cache hit %s", spec.uri, cached)
            return cached
        marker.unlink(missing_ok=True)

    staging = cache_dir / f"{spec.cache_key()}.tmp-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    ensure_dir(staging)
    try:
        payload = (
            _fetch_huggingface(spec, staging)
            if spec.scheme == "hf"
            else _fetch_modelscope(spec, staging)
        )
        if not payload.exists():
            raise ModelFetchError(f"model fetch produced no files for {spec.uri}")
        final = _commit_download(spec, staging, payload, cache_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    logger.info("model %s -> downloaded %s", spec.uri, final)
    return final


def resolve_model_params(ctx: RunContext) -> None:
    """Replace model URI parameters with local paths in a RunContext."""
    engine = ctx.engine
    names = ("student", "teacher") if engine == "trl" else ("student",)
    if engine not in {"swift", "trl", "easydistill-swift"}:
        return
    params = ctx.params
    revision = params.get("model_revision")
    cache_dir = ctx.model_cache_dir
    for name in names:
        value = params.get(name)
        if value:
            params.values[name] = str(
                resolve_model(ModelSpec(uri=str(value), revision=revision), cache_dir)
            )
