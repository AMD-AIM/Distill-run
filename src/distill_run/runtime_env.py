"""Select the isolated Python environment for a heavy engine."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger(__name__)

_ENGINE_ENV = {
    "swift": "swift",
    "trl": "trl",
    "easydistill": "easydistill",
    # Generation runs in the EasyDistill environment and launches Swift through
    # the explicit executable exported below.
    "easydistill-swift": "easydistill",
}
_WORKER_MARKER = "DISTILL_RUN_ENGINE_WORKER"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def environments_root() -> Path:
    configured = os.environ.get("DISTILL_RUN_ENVS_ROOT")
    return Path(configured) if configured else project_root() / ".venvs"


def engine_python(engine: str) -> Path | None:
    env_name = _ENGINE_ENV.get(engine)
    if env_name is None:
        return None
    override = os.environ.get(f"DISTILL_RUN_{env_name.upper()}_PYTHON")
    candidate = Path(override) if override else environments_root() / env_name / "bin" / "python"
    if not candidate.is_file():
        return None
    if override:
        return candidate
    environment = candidate.parent.parent
    marker = environment / ".distill-run-runtime"
    try:
        recorded = Path(marker.read_text(encoding="utf-8").strip())
        if recorded != environment.resolve():
            return None
    except OSError:
        return None
    return candidate


def maybe_run_in_engine_environment(engine: str, argv: Sequence[str]) -> int | None:
    """Run the complete command under its engine venv, once.

    The whole CLI is relaunched instead of trying to serialize ``RunContext``.
    Accelerate/torchrun therefore keep seeing the same public command and flags.
    """
    if os.environ.get(_WORKER_MARKER):
        return None
    python = engine_python(engine)
    if python is None:
        return None  # Backwards-compatible: use frameworks from the active env.
    try:
        if python.resolve() == Path(sys.executable).resolve():
            return None
    except OSError:
        pass

    env = dict(os.environ)
    env[_WORKER_MARKER] = "1"
    swift = environments_root() / "swift" / "bin" / "swift"
    if swift.is_file():
        env.setdefault("DISTILL_RUN_SWIFT_EXECUTABLE", str(swift))
    logger.info("using isolated %s environment: %s", engine, python)
    return subprocess.call([str(python), "-m", "distill_run", *argv], env=env)  # noqa: S603
