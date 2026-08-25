"""Small shared helpers: atomic writes, JSONL, HTTP session, child processes."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .errors import DataError

logger = logging.getLogger(__name__)

HTTP_TIMEOUT = 30


# --- filesystem --------------------------------------------------------------


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def is_nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def dir_has_entries(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def free_bytes(path: Path) -> int:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    stat = os.statvfs(probe)
    return stat.f_bavail * stat.f_frsize


# --- jsonl -------------------------------------------------------------------


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataError(f"{path}:{lineno} is not valid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise DataError(f"{path}:{lineno} must be a JSON object, got {type(row).__name__}")
            yield row


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise DataError(f"dataset file not found: {path}")
    rows = list(iter_jsonl(path))
    if not rows:
        raise DataError(f"dataset file is empty: {path}")
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    ensure_dir(path.parent)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def require_any_key(rows: list[dict[str, Any]], keys: Iterable[str], source: Path) -> None:
    """Cheap schema check so a wrong file fails before models are loaded."""
    wanted = list(keys)
    head = rows[0]
    if not any(k in head for k in wanted):
        raise DataError(
            f"{source}: first row has none of the expected keys {wanted}; got {sorted(head)}"
        )


# --- http --------------------------------------------------------------------


def http_session(total_retries: int = 3, backoff: float = 0.5) -> requests.Session:
    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
    )
    adapter = HTTPAdapter(max_retries=retry)
    sess = requests.Session()
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    return sess


# --- subprocess --------------------------------------------------------------


def stream_command(
    argv: Sequence[str],
    *,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    on_line: Callable[[str], None] | None = None,
) -> int:
    """Run ``argv`` forwarding every output line immediately; return its exit code.

    Line-by-line forwarding keeps training output live in the terminal instead of
    arriving in one buffered lump at the end.
    """
    merged = {**os.environ, **(env or {})}
    merged.setdefault("PYTHONUNBUFFERED", "1")
    logger.info("exec: %s", " ".join(argv))

    with subprocess.Popen(
        list(argv),
        cwd=cwd,
        env=merged,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            print(line, flush=True)
            if on_line is not None:
                on_line(line)
        return proc.wait()


def python_executable() -> str:
    return sys.executable or "python3"
