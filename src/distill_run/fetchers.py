"""Dataset fetchers, one per URI scheme.

Adding a source means adding a class here and listing it in ``FETCHERS``.
Credentials come from the environment (``HF_TOKEN``, ``HF_ENDPOINT``,
``MODELSCOPE_API_TOKEN``) and never from a config file.
"""

from __future__ import annotations

import abc
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlparse

import requests

from .catalog import dataset_error_hint
from .errors import DatasetFetchError
from .utils import HTTP_TIMEOUT, http_session

if TYPE_CHECKING:
    from .dataset import DatasetSpec

logger = logging.getLogger(__name__)
CHUNK = 1 << 20


class Fetcher(abc.ABC):
    name: str = "fetcher"

    @abc.abstractmethod
    def can_handle(self, spec: DatasetSpec) -> bool:
        """Whether this fetcher owns the given URI scheme."""

    @abc.abstractmethod
    def fetch(self, spec: DatasetSpec, staging: Path) -> Path:
        """Materialise the dataset and return its path.

        Must raise :class:`DatasetFetchError` on any failure, including a missing
        optional SDK or missing credentials.
        """


class LocalFetcher(Fetcher):
    name = "local"

    def can_handle(self, spec: DatasetSpec) -> bool:
        return spec.is_local

    def fetch(self, spec: DatasetSpec, staging: Path) -> Path:
        raw = spec.uri[len("file://") :] if spec.uri.startswith("file://") else spec.uri
        path = Path(raw).expanduser()
        if not path.exists():
            raise DatasetFetchError(f"dataset path does not exist: {path}{dataset_error_hint()}")
        return path


class HttpFetcher(Fetcher):
    name = "http"

    def can_handle(self, spec: DatasetSpec) -> bool:
        return spec.scheme in {"http", "https"}

    def fetch(self, spec: DatasetSpec, staging: Path) -> Path:
        filename = Path(unquote(urlparse(spec.uri).path)).name or "dataset"
        target = staging / filename
        logger.info("downloading %s -> %s", spec.uri, target)
        try:
            with http_session().get(spec.uri, stream=True, timeout=HTTP_TIMEOUT) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("Content-Length") or 0)
                written = 0
                with target.open("wb") as f:
                    for chunk in resp.iter_content(CHUNK):
                        f.write(chunk)
                        written += len(chunk)
        except requests.RequestException as exc:
            raise DatasetFetchError(f"failed to download {spec.uri}: {exc}") from exc
        if written == 0:
            raise DatasetFetchError(f"downloaded 0 bytes from {spec.uri}")
        if total and written != total:
            raise DatasetFetchError(
                f"incomplete download from {spec.uri}: got {written} of {total} bytes"
            )
        return target


class HuggingFaceFetcher(Fetcher):
    """``hf://org/name``, honouring ``HF_ENDPOINT`` so a mirror works unchanged."""

    name = "hf"

    def can_handle(self, spec: DatasetSpec) -> bool:
        return spec.scheme == "hf"

    def fetch(self, spec: DatasetSpec, staging: Path) -> Path:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise DatasetFetchError(
                "hf:// datasets need huggingface_hub; pip install huggingface_hub"
            ) from exc

        repo_id = spec.uri[len("hf://") :].strip("/")
        if not repo_id:
            raise DatasetFetchError(f"malformed Hub URI: {spec.uri}")
        allow = [f"{spec.subset}/*", f"{spec.subset}*"] if spec.subset else None
        logger.info(
            "snapshot_download repo=%s revision=%s endpoint=%s",
            repo_id,
            spec.revision or "default",
            os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
        )
        try:
            local = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=spec.revision,
                allow_patterns=allow,
                local_dir=str(staging),
            )
        except Exception as exc:  # noqa: BLE001 - the hub raises many error types
            raise DatasetFetchError(f"failed to fetch {spec.uri}: {exc}") from exc
        return Path(local)


class ModelScopeFetcher(Fetcher):
    """``ms://org/name`` (or ``modelscope://``)."""

    name = "modelscope"

    def can_handle(self, spec: DatasetSpec) -> bool:
        return spec.scheme in {"ms", "modelscope"}

    def fetch(self, spec: DatasetSpec, staging: Path) -> Path:
        try:
            from modelscope.hub.snapshot_download import snapshot_download
        except ImportError as exc:
            raise DatasetFetchError("ms:// datasets need the modelscope SDK") from exc

        repo_id = spec.uri.split("://", 1)[1].strip("/")
        if not repo_id:
            raise DatasetFetchError(f"malformed ModelScope URI: {spec.uri}")
        logger.info("modelscope snapshot_download repo=%s revision=%s", repo_id, spec.revision)
        try:
            local = snapshot_download(
                repo_id,
                revision=spec.revision,
                repo_type="dataset",
                cache_dir=str(staging),
            )
        except Exception as exc:  # noqa: BLE001
            raise DatasetFetchError(f"failed to fetch {spec.uri}: {exc}") from exc
        return Path(local)


# Order matters: the first fetcher that claims the URI wins.
FETCHERS: tuple[Fetcher, ...] = (
    LocalFetcher(),
    HttpFetcher(),
    HuggingFaceFetcher(),
    ModelScopeFetcher(),
)
