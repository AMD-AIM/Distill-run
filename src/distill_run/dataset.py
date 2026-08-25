"""Turn a dataset specification into a local path, before any engine work starts.

A dataset argument is a *specification*, not necessarily a file that already
exists: it may be a mounted path or a network/Hub URI. Resolution happens ahead of
the engine, so a bad URI or a failed download exits without loading any model, and
engines only ever see local paths.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .catalog import dataset_error_hint
from .errors import DatasetFetchError
from .fetchers import FETCHERS
from .utils import atomic_write_json, ensure_dir

logger = logging.getLogger(__name__)

LOCAL_SCHEMES = frozenset({"", "file"})


@dataclass(frozen=True)
class DatasetSpec:
    """A data location: a mounted path, or a network/Hub URI."""

    uri: str
    format: str | None = None
    revision: str | None = None
    subset: str | None = None
    split: str = "train"
    max_samples: int | None = None
    shuffle_seed: int = 42
    purpose: str | None = None

    @property
    def scheme(self) -> str:
        parsed = urlparse(self.uri)
        return parsed.scheme.lower() if parsed.scheme else ""

    @property
    def is_local(self) -> bool:
        return self.scheme in LOCAL_SCHEMES

    def cache_key(self) -> str:
        material = "|".join(
            [
                self.uri,
                self.revision or "",
                self.subset or "",
                self.format or "",
                self.split,
                str(self.max_samples or ""),
                str(self.shuffle_seed),
                self.purpose or "",
            ]
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]

    def describe(self) -> str:
        extras = [f"{k}={v}" for k, v in (("rev", self.revision), ("subset", self.subset)) if v]
        return self.uri + (f" ({', '.join(extras)})" if extras else "")


@dataclass(frozen=True)
class ResolvedDataset:
    """A local, readable file or directory."""

    spec: DatasetSpec
    path: Path
    from_cache: bool = False

    def __fspath__(self) -> str:
        return str(self.path)

    def __str__(self) -> str:
        return str(self.path)


@dataclass(frozen=True)
class CacheEntry:
    """Content-addressed cache slot.

    A fetch stages into ``<key>.tmp-<pid>`` and renames on success, so a killed
    run never leaves a half download that a later run mistakes for a cache hit.
    """

    key: str
    payload_dir: Path
    marker: Path

    @property
    def is_complete(self) -> bool:
        return self.marker.is_file() and self.payload_dir.exists()

    def stored_path(self) -> Path:
        meta = json.loads(self.marker.read_text(encoding="utf-8"))
        return Path(meta["path"])

    def stage_dir(self) -> Path:
        staged = self.payload_dir.with_name(f"{self.payload_dir.name}.tmp-{os.getpid()}")
        shutil.rmtree(staged, ignore_errors=True)
        return ensure_dir(staged)

    def commit(self, staged: Path, spec: DatasetSpec) -> Path:
        if self.payload_dir.exists():
            shutil.rmtree(self.payload_dir, ignore_errors=True)
        ensure_dir(self.payload_dir.parent)
        os.replace(staged, self.payload_dir)
        final = self.payload_dir
        entries = [p for p in final.iterdir() if not p.name.startswith(".")]
        if len(entries) == 1 and entries[0].is_file():
            final = entries[0]
        atomic_write_json(
            self.marker,
            {
                "uri": spec.uri,
                "revision": spec.revision,
                "subset": spec.subset,
                "format": spec.format,
                "path": str(final),
                "cached_at": time.time(),
            },
        )
        return final


def cache_entry(spec: DatasetSpec, cache_dir: Path) -> CacheEntry:
    key = spec.cache_key()
    return CacheEntry(key=key, payload_dir=cache_dir / key, marker=cache_dir / f"{key}.done")


def resolve(spec: DatasetSpec, cache_dir: Path) -> ResolvedDataset:
    from .normalization import normalize_dataset

    fetcher = next((f for f in FETCHERS if f.can_handle(spec)), None)
    if fetcher is None:
        raise DatasetFetchError(
            f"unsupported dataset URI scheme '{spec.scheme}' in {spec.uri}; "
            f"supported: local path, http(s), hf, ms{dataset_error_hint()}"
        )

    if fetcher.name == "local":
        path = fetcher.fetch(spec, cache_dir)
        path = normalize_dataset(path, spec, cache_dir)
        logger.info("dataset %s -> local path %s", spec.describe(), path)
        return ResolvedDataset(spec=spec, path=path)

    ensure_dir(cache_dir)
    entry = cache_entry(spec, cache_dir)
    if entry.is_complete:
        cached = entry.stored_path()
        if cached.exists():
            cached = normalize_dataset(cached, spec, cache_dir, reuse=True)
            logger.info("dataset %s -> cache hit %s", spec.describe(), cached)
            return ResolvedDataset(spec=spec, path=cached, from_cache=True)
        logger.warning("stale cache marker for %s; refetching", spec.describe())
        entry.marker.unlink(missing_ok=True)

    staged = entry.stage_dir()
    try:
        payload = fetcher.fetch(spec, staged)
        if not payload.exists():
            raise DatasetFetchError(f"fetcher '{fetcher.name}' produced nothing for {spec.uri}")
        if payload != staged and staged not in payload.parents:
            # The SDK wrote into its own cache; copy in so the commit stays atomic.
            target = staged / payload.name
            if payload.is_dir():
                shutil.copytree(payload, target, dirs_exist_ok=True)
            else:
                shutil.copy2(payload, target)
        final = entry.commit(staged, spec)
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        raise
    final = normalize_dataset(final, spec, cache_dir)
    logger.info("dataset %s -> downloaded %s", spec.describe(), final)
    return ResolvedDataset(spec=spec, path=final)
