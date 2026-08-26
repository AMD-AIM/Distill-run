"""Everything an engine is allowed to know about the current run."""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import Resolved
from .dataset import ResolvedDataset
from .errors import ConfigError
from .signals import Cancellation
from .utils import ensure_dir

logger = logging.getLogger(__name__)


def generate_run_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


@dataclass
class RunContext:
    engine: str
    run_id: str
    params: Resolved
    config: dict[str, Any]
    work_dir: Path
    output: Path
    cache_dir: Path
    model_cache_dir: Path
    cancellation: Cancellation
    datasets: dict[str, ResolvedDataset] = field(default_factory=dict)

    @property
    def output_dir(self) -> Path:
        """Directory to write into, whether ``--output`` named a file or a dir."""
        return self.output if self.output.is_dir() or not self.output.suffix else self.output.parent

    @property
    def resume(self) -> bool:
        return bool(self.params.get("resume", False))

    def param(self, dest: str, default: Any = None) -> Any:
        return self.params.get(dest, default)

    def require(self, dest: str) -> Any:
        value = self.params.get(dest)
        if value is None:
            raise ConfigError(f"missing required value for '{dest}'")
        return value

    def dataset(self, dest: str) -> ResolvedDataset:
        if dest not in self.datasets:
            raise ConfigError(f"dataset '{dest}' was not resolved before the engine started")
        return self.datasets[dest]

    def scratch(self, *parts: str) -> Path:
        """Per-run scratch space, e.g. for a rewritten engine config."""
        return ensure_dir(self.work_dir / "runs" / self.run_id).joinpath(*parts)

    def section(self, key: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
        value = self.config.get(key, default if default is not None else {})
        if not isinstance(value, dict):
            raise ConfigError(f"config key '{key}' must be a mapping, got {type(value).__name__}")
        return dict(value)

    def substep(self, *, engine: str, output: Path, config: dict[str, Any]) -> RunContext:
        """Derive a context for a nested engine, used by combined pipelines."""
        return replace(self, engine=engine, output=output, config=config, datasets={})


def prepare_environment(work_dir: Path, cache_dir: Path, model_cache_dir: Path) -> None:
    """Point framework caches at the scratch directory, not the container root."""
    os.environ.setdefault("HF_HOME", str(cache_dir / "hf"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir / "hf-datasets"))
    os.environ.setdefault("MODELSCOPE_CACHE", str(cache_dir / "modelscope"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for path in (work_dir, cache_dir, model_cache_dir):
        try:
            ensure_dir(path)
        except OSError as exc:
            logger.warning("cannot create %s: %s", path, exc)
