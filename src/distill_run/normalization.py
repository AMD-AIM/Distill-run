"""Convert common instruction datasets to engine-ready JSONL."""

from __future__ import annotations

import json
import logging
import os
import random
import re
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from .catalog import dataset_error_hint
from .dataset import DatasetSpec
from .errors import DataError
from .utils import ensure_dir, iter_jsonl

logger = logging.getLogger(__name__)

_FORMATS = ("jsonl", "parquet", "arrow", "json")
_DETECTION_ORDER = ("parquet", "arrow", "jsonl", "json")
_PROMPT_KEYS = ("instruction", "prompt", "question", "query", "problem")
_RESPONSE_KEYS = ("output", "response", "answer", "completion", "solution")
_ROLE_KEYS = ("role", "from", "speaker")
_CONTENT_KEYS = ("content", "value", "text")
_ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "bot": "assistant",
    "model": "assistant",
    "system": "system",
}


def normalize_dataset(
    path: Path,
    spec: DatasetSpec,
    cache_dir: Path,
    *,
    reuse: bool = False,
) -> Path:
    """Normalize a fetched dataset when an engine declared a target purpose."""
    if spec.purpose is None:
        return path

    output = cache_dir / f"{spec.cache_key()}.normalized.jsonl"
    if reuse and output.is_file() and output.stat().st_size:
        return output

    files, detected = _dataset_files(path, spec)
    rows = _rows(files, detected, spec.split)
    normalized = (_normalize_row(row, spec.purpose) for row in rows)
    valid = (row for row in normalized if row is not None)
    if spec.max_samples is not None:
        valid = iter(_reservoir(valid, spec.max_samples, spec.shuffle_seed))

    ensure_dir(output.parent)
    fd, temporary = tempfile.mkstemp(
        dir=str(output.parent), prefix=f".{output.name}.", suffix=".tmp"
    )
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for row in valid:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
            stream.flush()
            os.fsync(stream.fileno())
        if count == 0:
            raise DataError(
                f"{path}: no rows can be normalized as {spec.purpose}; "
                f"expected messages/conversations or prompt and response fields"
            )
        os.replace(temporary, output)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise

    logger.info(
        "normalized %d %s rows from %s (%s) -> %s",
        count,
        spec.purpose,
        path,
        detected,
        output,
    )
    return output


def _dataset_files(path: Path, spec: DatasetSpec) -> tuple[list[Path], str]:
    requested = None if spec.format in {None, "auto", "hf"} else spec.format
    if requested and requested not in _FORMATS:
        raise DataError(
            f"unsupported --dataset-format {requested!r}; expected auto, jsonl, json, "
            f"parquet or arrow{dataset_error_hint()}"
        )

    if path.is_file():
        detected = requested or _extension_format(path)
        if not detected:
            raise DataError(f"cannot detect dataset format from file: {path}")
        return [path], detected
    if not path.is_dir():
        raise DataError(f"dataset path is neither a file nor directory: {path}")

    candidates: dict[str, list[Path]] = {fmt: sorted(path.rglob(f"*.{fmt}")) for fmt in _FORMATS}
    detected = requested or next((fmt for fmt in _DETECTION_ORDER if candidates[fmt]), None)
    if detected is None:
        raise DataError(f"{path}: no JSONL, JSON, Parquet or Arrow data files found")
    files = candidates[detected]
    if not files:
        raise DataError(f"{path}: no {detected} files found")

    split_matches = [
        file
        for file in files
        if spec.split.lower() in re.split(r"[^a-z0-9]+", str(file.relative_to(path)).lower())
    ]
    if split_matches:
        files = split_matches
    if spec.subset:
        subset_matches = [file for file in files if spec.subset.lower() in str(file).lower()]
        if subset_matches:
            files = subset_matches
    return files, detected


def _extension_format(path: Path) -> str | None:
    suffix = path.suffix.lower().lstrip(".")
    if suffix in _FORMATS:
        return suffix
    return "jsonl" if suffix in {"ndjson", "jsonlines"} else None


def _rows(files: list[Path], format_name: str, split: str) -> Iterator[dict[str, Any]]:
    if format_name == "jsonl":
        for file in files:
            yield from iter_jsonl(file)
        return
    if format_name == "json":
        for file in files:
            yield from _json_rows(file, split)
        return

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise DataError(
            f"{format_name} datasets require the 'datasets' package; install distill-run[dataset]"
        ) from exc

    try:
        dataset = load_dataset(format_name, data_files=[str(file) for file in files], split="train")
        for row in dataset:
            if isinstance(row, dict):
                yield dict(row)
    except Exception as exc:  # noqa: BLE001 - datasets exposes backend-specific errors
        raise DataError(f"failed to read {format_name} dataset: {exc}") from exc


def _json_rows(path: Path, split: str) -> Iterator[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataError(f"failed to read JSON dataset {path}: {exc}") from exc
    if isinstance(payload, dict) and isinstance(payload.get(split), list):
        payload = payload[split]
    elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
        payload = payload["data"]
    elif isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise DataError(f"{path}: JSON dataset must be an object or array of objects")
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise DataError(f"{path}: row {index} must be an object")
        yield row


def _normalize_row(row: dict[str, Any], purpose: str) -> dict[str, Any] | None:
    messages = _messages(row.get("messages") or row.get("conversations"))
    if purpose in {"seed", "prompt"}:
        if messages:
            prompt = _prompt_messages(messages)
            return {"messages": prompt} if prompt else None
        prompt = _prompt_text(row)
        return {"instruction": prompt} if prompt else None

    if purpose == "sft":
        if messages and _has_roles(messages, "user", "assistant"):
            return {"messages": messages}
        prompt = _prompt_text(row)
        response = _first_text(row, _RESPONSE_KEYS)
        if prompt and response:
            return {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ]
            }
        return None
    raise DataError(f"unknown dataset normalization purpose: {purpose}")


def _messages(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, str]] = []
    for message in value:
        if not isinstance(message, dict):
            continue
        role = _first_text(message, _ROLE_KEYS)
        content = _first_text(message, _CONTENT_KEYS)
        mapped = _ROLE_MAP.get(role.lower()) if role else None
        if mapped and content:
            result.append({"role": mapped, "content": content})
    return result


def _prompt_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    last_user = max(
        (index for index, message in enumerate(messages) if message["role"] == "user"),
        default=-1,
    )
    return messages[: last_user + 1] if last_user >= 0 else []


def _has_roles(messages: list[dict[str, str]], *roles: str) -> bool:
    present = {message["role"] for message in messages}
    return all(role in present for role in roles)


def _first_text(row: dict[str, Any], keys: Iterable[str]) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _prompt_text(row: dict[str, Any]) -> str | None:
    prompt = _first_text(row, _PROMPT_KEYS)
    extra = _first_text(row, ("input",))
    if prompt and extra:
        return f"{prompt}\n\n{extra}"
    return prompt or extra


def _reservoir(rows: Iterable[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0:
        raise DataError("--dataset-max-samples must be greater than zero")
    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if index < limit:
            sample.append(row)
            continue
        replacement = rng.randint(0, index)
        if replacement < limit:
            sample[replacement] = row
    return sample
