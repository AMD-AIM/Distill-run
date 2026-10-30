"""Machine-readable progress lines for the service layer.

Each event is one log line ending in ``DISTILL_EVENT <json>``; the JSON always
has ``event`` and ``time`` (UTC, RFC 3339)::

    DISTILL_EVENT {"event": "stage", "stage": "downloading", "time": "..."}
    DISTILL_EVENT {"event": "progress", "epoch": 3, "total_epochs": 100, "eta_seconds": 412, "time": "..."}
    DISTILL_EVENT {"event": "exit", "exit_code": 5, "code": "IntegrityCheckFailed", "stage": "downloading", ...}

Stages, in order: ``downloading``, ``converting``, ``training``, ``uploading``.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

MARKER = "DISTILL_EVENT"
STAGES = ("downloading", "converting", "training", "uploading")

logger = logging.getLogger("distill_run.events")
# The service parses these lines, so LOG_LEVEL=WARNING must not hide them.
logger.setLevel(logging.INFO)

_current: dict[str, str | None] = {"stage": None}


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def emit(event: str, **fields: Any) -> None:
    payload = {"event": event, **fields, "time": utc_now()}
    logger.info("%s %s", MARKER, json.dumps(payload, ensure_ascii=False, default=str))


def reset() -> None:
    _current["stage"] = None


def stage(name: str) -> None:
    if name not in STAGES:
        raise ValueError(f"unknown stage {name!r}")
    if _current["stage"] == name:
        return
    _current["stage"] = name
    emit("stage", stage=name)


def current_stage() -> str | None:
    return _current["stage"]


def progress(epoch: int, total_epochs: int, eta_seconds: float | None, **extra: Any) -> None:
    emit("progress", epoch=epoch, total_epochs=total_epochs,
         eta_seconds=None if eta_seconds is None else max(0, round(eta_seconds)), **extra)


def finish(exit_code: int, code: str | None = None) -> None:
    emit("exit", exit_code=exit_code, code=code, stage=_current["stage"])


def parse(line: str) -> dict[str, Any] | None:
    """Inverse of :func:`emit`, for the service layer and tests."""
    _, sep, rest = line.partition(f"{MARKER} ")
    if not sep:
        return None
    try:
        payload = json.loads(rest)
    except ValueError:
        return None
    return payload if isinstance(payload, dict) and "event" in payload else None
