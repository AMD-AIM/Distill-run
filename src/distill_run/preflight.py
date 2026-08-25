"""Cheap checks that run before the expensive part of a run.

Everything here is CPU-only and takes seconds. The point is to fail a
misconfigured run immediately instead of after a few minutes of weight loading —
which is exactly how a mistyped hyper-parameter name used to waste half a minute
per attempt.
"""

from __future__ import annotations

import difflib
import logging
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .catalog import model_error_hint
from .errors import ConfigError, PreflightError, TeacherUnavailableError
from .utils import HTTP_TIMEOUT, ensure_dir, free_bytes, http_session

logger = logging.getLogger(__name__)

MIN_FREE_BYTES = 1 << 30


def check_writable(path: Path, role: str) -> None:
    try:
        ensure_dir(path)
        probe = path / f".distill-run-write-probe-{os.getpid()}"
        probe.touch()
        probe.unlink()
    except OSError as exc:
        raise PreflightError(f"{role} {path} is not writable: {exc}") from exc


def check_space(path: Path) -> None:
    available = free_bytes(path)
    if available < MIN_FREE_BYTES:
        raise PreflightError(f"only {available / 1e9:.2f} GB free under {path}; refusing to start")
    logger.info("disk: %.1f GB free under %s", available / 1e9, path)


def check_weights(path: Path, role: str) -> None:
    if not path.exists():
        raise PreflightError(f"{role} weights path does not exist: {path}{model_error_hint()}")
    if path.is_dir() and not any(path.iterdir()):
        raise PreflightError(f"{role} weights directory is empty: {path}")
    logger.info("%s weights: %s", role, path)


def check_teacher_endpoint(base_url: str, model_id: str | None, api_key: str | None) -> None:
    """Verify the teacher service answers before generating anything."""
    url = base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        resp = http_session().get(url, headers=headers, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001 - any failure means "teacher is down"
        raise TeacherUnavailableError(f"teacher endpoint {url} is not reachable: {exc}") from exc

    served = [item.get("id") for item in payload.get("data", []) if isinstance(item, dict)]
    logger.info("teacher %s is up, serving: %s", base_url, served or "unknown")
    if model_id and served and model_id not in served:
        logger.warning("teacher does not serve model_id %r; available: %s", model_id, served)


def check_arg_names(
    section: dict[str, Any],
    valid: Iterable[str],
    *,
    label: str,
    config_key: str,
) -> None:
    """Reject unknown framework hyper-parameter names, suggesting close matches.

    Framework argument names drift between versions (ms-swift renamed
    ``train_type`` to ``tuner_type``), and the frameworks themselves only notice
    after loading the model. Checking the names against the real argument class
    costs milliseconds.
    """
    known = set(valid)
    if not known:
        logger.debug("no argument list available for %s; skipping name check", label)
        return
    unknown = sorted(k for k in section if k not in known)
    if not unknown:
        return
    lines = []
    for key in unknown:
        close = difflib.get_close_matches(key, known, n=3, cutoff=0.6)
        hint = f" (did you mean: {', '.join(close)}?)" if close else ""
        lines.append(f"  {config_key}.{key}{hint}")
    raise ConfigError(f"{label} does not accept these config keys:\n" + "\n".join(lines))


def swift_arg_names() -> set[str]:
    """Field names accepted by the installed ms-swift ``sft`` arguments class."""
    try:
        import dataclasses

        from swift.arguments import SftArguments
    except ImportError:
        return set()
    return {f.name for f in dataclasses.fields(SftArguments)}


def trl_arg_names() -> set[str]:
    """Field names accepted by the installed TRL ``DistillationConfig``."""
    try:
        import dataclasses

        from trl import DistillationConfig
    except ImportError:
        return set()
    return {f.name for f in dataclasses.fields(DistillationConfig)}
