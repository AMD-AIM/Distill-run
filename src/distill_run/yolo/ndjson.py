"""Ultralytics Platform NDJSON datasets -> a YOLO txt tree with ``data.yaml``.

The first line describes the dataset (``task``, ``class_names``); each further
line is one image::

    {"type": "dataset", "task": "detect", "class_names": {"0": "person"}}
    {"type": "image", "file": "a.jpg", "split": "train", "url": "https://...",
     "annotations": {"boxes": [[0, 0.5, 0.5, 0.2, 0.3]]}}

Boxes are ``[cls, cx, cy, w, h]`` normalised to the image. Image URLs are usually
signed and expire, so the converted tree is keyed by a hash of the records with
query strings removed: a re-exported manifest of the same data reuses it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import yaml

from ..errors import DatasetFetchError
from ..utils import HTTP_TIMEOUT, http_session

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")
_SAFE_SUFFIXES = frozenset({"jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"})


class NdjsonError(ValueError):
    """Malformed manifest; the caller reports it as a data-format error."""


def looks_like_ndjson(path: Path) -> bool:
    name = path.name.lower()
    if not path.is_file() or not name.endswith((".ndjson", ".jsonl")):
        return False
    if name.endswith(".ndjson"):
        return True
    with path.open(encoding="utf-8", errors="replace") as f:
        first = f.readline(1 << 20).strip()
    try:
        head = json.loads(first)
    except ValueError:
        return False
    return isinstance(head, dict) and (head.get("type") == "dataset" or "class_names" in head)


def convert(manifest: Path, out_root: Path, *, workers: int = 32) -> Path:
    """Convert ``manifest`` once under ``out_root``; return the ``data.yaml`` path."""
    header, records = _read(manifest)
    task = header.get("task", "detect")
    if task != "detect":
        raise NdjsonError(f"{manifest.name}: task '{task}' is not supported (detect only)")
    local_root = manifest.parent / header["path"] if header.get("path") else None
    for i, record in enumerate(records, 2):
        if record.get("split") not in SPLITS:
            raise NdjsonError(f"{manifest.name}:{i}: invalid split {record.get('split')!r}")
        if not isinstance(record.get("file"), str) or not record["file"]:
            raise NdjsonError(f"{manifest.name}:{i}: missing image file name")
        if not record.get("url"):
            if local_root is None:
                raise NdjsonError(f"{manifest.name}:{i}: image has no url")
            if Path(record["file"]).name != record["file"]:
                raise NdjsonError(f"{manifest.name}:{i}: invalid image name {record['file']!r}")
            record["url"] = local_root / "images" / record["split"] / record["file"]

    target = out_root / f"{manifest.stem}-{_fingerprint(header, records)}"
    data_yaml = target / "data.yaml"
    if (target / ".complete").is_file():
        logger.info("ndjson %s -> cached %s", manifest.name, target)
        return data_yaml

    _ensure_val(records)
    names = _names(header, records)
    staging = target.with_name(f"{target.name}.tmp-{os.getpid()}")
    shutil.rmtree(staging, ignore_errors=True)
    try:
        jobs = []
        for i, record in enumerate(records):
            stem = f"{i:07d}"
            suffix = record["file"].rsplit(".", 1)[-1].lower()
            image = staging / "images" / record["split"] / f"{stem}.{suffix if suffix in _SAFE_SUFFIXES else 'jpg'}"
            label = staging / "labels" / record["split"] / f"{stem}.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text(_label_text(record, manifest.name), encoding="utf-8")
            jobs.append((image, record["url"]))
        _download_all(jobs, workers)
        splits = sorted({r["split"] for r in records}, key=SPLITS.index)
        meta = {k: v for k, v in header.items() if k not in {"type", "class_names", "path", "url"}}
        meta.update({"path": ".", **{s: f"images/{s}" for s in splits}, "names": names})
        (staging / "data.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
        (staging / ".complete").write_text(str(manifest), encoding="utf-8")
        if not (target / ".complete").is_file():  # another process may have finished first
            shutil.rmtree(target, ignore_errors=True)
            try:
                staging.rename(target)
            except OSError:
                if not (target / ".complete").is_file():
                    raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    counts = {s: sum(r["split"] == s for r in records) for s in SPLITS}
    logger.info("ndjson %s -> %s (%s)", manifest.name, target,
                ", ".join(f"{k}={v}" for k, v in counts.items() if v))
    return data_yaml


def _read(manifest: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lines = []
    with manifest.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    lines.append(json.loads(line))
                except ValueError as exc:
                    raise NdjsonError(f"{manifest.name}:{n}: invalid JSON ({exc})") from exc
    if not lines or not isinstance(lines[0], dict):
        raise NdjsonError(f"{manifest.name}: empty manifest")
    header, records = lines[0], lines[1:]
    if not records:
        raise NdjsonError(f"{manifest.name}: no image records")
    return header, records


def _strip_query(url: Any) -> str:
    if isinstance(url, Path):
        return str(url.resolve())
    parts = urlsplit(str(url))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _fingerprint(header: dict[str, Any], records: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    h.update(json.dumps({k: v for k, v in header.items() if k != "url"}, sort_keys=True).encode())
    for r in records:
        stable = {k: v for k, v in r.items() if k != "url"}
        stable["_source"] = _strip_query(r["url"])
        h.update(json.dumps(stable, sort_keys=True, default=str).encode())
    return h.hexdigest()[:12]


def _ensure_val(records: list[dict[str, Any]]) -> None:
    """Without a val split, move a seeded 10% of train to val (as the Platform does)."""
    if any(r["split"] == "val" for r in records):
        return
    train = [r for r in records if r["split"] == "train"]
    if len(train) < 2:
        raise NdjsonError(f"{len(train)} training image(s) and no val split; need at least 2")
    random.Random(0).shuffle(train)
    for r in train[: max(1, len(train) // 10)]:
        r["split"] = "val"
    logger.warning("ndjson has no val split; moved %d of %d train images to val",
                   max(1, len(train) // 10), len(train))


def _boxes(record: dict[str, Any]) -> list:
    annotations = record.get("annotations") or {}
    return annotations.get("boxes") or next(iter(annotations.values()), None) or []


def _names(header: dict[str, Any], records: list[dict[str, Any]]) -> dict[int, str]:
    names = {int(k): str(v) for k, v in (header.get("class_names") or {}).items()}
    ids = {int(box[0]) for r in records for box in _boxes(r) if box}
    top = max(ids | set(names), default=-1)
    return {i: names.get(i, f"class{i}") for i in range(top + 1)}


def _label_text(record: dict[str, Any], source: str) -> str:
    lines = []
    for box in _boxes(record):
        if len(box) < 5:
            raise NdjsonError(f"{source}: {record['file']}: box needs [cls, cx, cy, w, h], got {box}")
        lines.append(" ".join(str(int(box[0])) if i == 0 else f"{float(v):.6g}"
                              for i, v in enumerate(box[:5])))
    return "".join(f"{line}\n" for line in lines)


_local = threading.local()


def _session():
    if not hasattr(_local, "session"):
        _local.session = http_session()
    return _local.session


def _fetch(path: Path, url: Any) -> str | None:
    """Return None on success, else an error message."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(url, Path):
        if not url.is_file():
            return f"{url}: not found"
        shutil.copy2(url, path)
        return None
    tmp = path.with_name(f".{path.name}.part")
    try:
        with _session().get(url, stream=True, timeout=HTTP_TIMEOUT) as resp:
            resp.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in resp.iter_content(1 << 16):
                    f.write(chunk)
        if tmp.stat().st_size == 0:
            return f"{_strip_query(url)}: empty response"
        os.replace(tmp, path)
        return None
    except Exception as exc:  # noqa: BLE001 - collected and reported together
        return f"{_strip_query(url)}: {exc}"
    finally:
        tmp.unlink(missing_ok=True)


def _download_all(jobs: list[tuple[Path, Any]], workers: int) -> None:
    total, done, failures = len(jobs), 0, []
    started, last_log = time.time(), 0.0
    with ThreadPoolExecutor(max_workers=max(1, min(workers, total))) as pool:
        for error in pool.map(lambda job: _fetch(*job), jobs):
            done += 1
            if error:
                failures.append(error)
            if time.time() - last_log > 10 or done == total:
                last_log = time.time()
                logger.info("ndjson images: %d/%d (%.0f/s)", done, total,
                            done / max(time.time() - started, 1e-6))
    if failures:
        raise DatasetFetchError(
            f"downloaded {total - len(failures)}/{total} images; first errors: "
            + "; ".join(failures[:3])
        )
