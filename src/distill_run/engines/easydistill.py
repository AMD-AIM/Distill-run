"""Black-box data generation: EasyDistill pipeline against a teacher HTTP service.

EasyDistill is called as a library; its CLI is only used as a fallback when the
import fails. Either way the config handed over has its dataset path already
resolved to a local file, so the upstream local-path validation is satisfied even
when the caller passed an ``hf://`` URI.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from .. import preflight
from ..config import dump_yaml
from ..context import RunContext
from ..dataset import DatasetSpec
from ..errors import ConfigError, EngineError, classify
from ..utils import is_nonempty_file, read_jsonl, require_any_key, stream_command
from .base import Engine, RunResult, make_spec, specs_from_params

logger = logging.getLogger(__name__)

DEFAULT_JOB_TYPE = "instruct_distill"
SEED_KEYS = ("instruction", "prompt", "query", "messages")


class EasyDistillEngine(Engine):
    name = "easydistill"

    def dataset_specs(self, ctx: RunContext) -> dict[str, DatasetSpec]:
        specs = specs_from_params(ctx)
        if "input" not in specs:
            from_config = ctx.section("dataset").get("input_path")
            if not from_config:
                raise ConfigError(
                    "no seed data: pass --input (or INPUT_URI), "
                    "or set dataset.input_path in --config"
                )
            specs["input"] = make_spec(ctx, str(from_config))
        return specs

    def preflight(self, ctx: RunContext) -> None:
        logger.info("easydistill job_type=%s", ctx.config.get("job_type", DEFAULT_JOB_TYPE))
        base_url = self._teacher_field(ctx, "base_url", "teacher_base_url")
        model_id = self._teacher_field(ctx, "model_id", "teacher_model_id")
        if not base_url:
            raise ConfigError(
                "teacher endpoint missing: pass --teacher-base-url (or TEACHER_BASE_URL), "
                "or set backend.base_url in --config"
            )
        if not model_id:
            raise ConfigError(
                "teacher model id missing: pass --teacher-model-id (or TEACHER_MODEL_ID), "
                "or set backend.model_id in --config"
            )
        preflight.check_teacher_endpoint(
            base_url, model_id, self._teacher_field(ctx, "api_key", "teacher_api_key")
        )

    def run(self, ctx: RunContext) -> RunResult:
        seed = ctx.dataset("input").path
        if seed.is_dir():
            raise ConfigError(f"easydistill expects a seed file, got directory {seed}")
        rows = read_jsonl(seed)
        require_any_key(rows, SEED_KEYS, seed)
        logger.info("seed rows: %d from %s", len(rows), seed)

        output = ctx.output
        config_path = self._materialize_config(ctx, seed, output)
        job_type = ctx.config.get("job_type", DEFAULT_JOB_TYPE)
        self._dispatch(job_type, config_path)

        if not is_nonempty_file(output):
            raise EngineError(f"easydistill finished but produced no rows at {output}")
        produced = len(read_jsonl(output))
        logger.info("generated %d SFT rows -> %s", produced, output)
        return RunResult(artifacts=[output], message=f"{produced} SFT rows at {output}")

    def _dispatch(self, job_type: str, config_path: Path) -> None:
        if os.environ.get("DISTILL_EASYDISTILL_MODE", "library").lower() == "subprocess":
            self._run_subprocess(config_path)
            return
        try:
            runner = self._load_runner(job_type)
        except ImportError as exc:
            logger.warning("cannot import EasyDistill (%s); falling back to its CLI", exc)
            self._run_subprocess(config_path)
            return
        try:
            runner(str(config_path))
        except Exception as exc:  # noqa: BLE001 - normalise into our taxonomy
            raise classify(exc) from exc

    @staticmethod
    def _load_runner(job_type: str):
        from easydistill.cli.main import _JOB_DISPATCH

        runner = _JOB_DISPATCH.get(job_type)
        if runner is None:
            valid = ", ".join(sorted(_JOB_DISPATCH))
            raise ConfigError(f"unsupported job_type '{job_type}'; available: {valid}")
        return runner

    @staticmethod
    def _run_subprocess(config_path: Path) -> None:
        code = stream_command(["easydistill", "--config", str(config_path)])
        if code != 0:
            raise EngineError(f"easydistill CLI exited with {code}")

    def _materialize_config(self, ctx: RunContext, seed: Path, output: Path) -> Path:
        """Rewrite the user config with resolved paths and CLI overrides applied."""
        config: dict[str, Any] = dict(ctx.config)
        config.setdefault("job_type", DEFAULT_JOB_TYPE)

        backend = dict(config.get("backend") or {})
        backend.setdefault("type", "openai")
        for config_key, dest in (
            ("base_url", "teacher_base_url"),
            ("model_id", "teacher_model_id"),
            ("api_key", "teacher_api_key"),
        ):
            override = ctx.param(dest)
            if override and ctx.params.origin(dest) in {"arg", "env"}:
                backend[config_key] = override
            backend.setdefault(config_key, override)
        config["backend"] = backend

        dataset = dict(config.get("dataset") or {})
        dataset["input_path"] = str(seed)
        dataset["output_path"] = str(output)
        config["dataset"] = dataset

        output.parent.mkdir(parents=True, exist_ok=True)
        path = dump_yaml(ctx.scratch("easydistill.resolved.yaml"), config)
        logger.info("resolved easydistill config -> %s", path)
        return path

    @staticmethod
    def _teacher_field(ctx: RunContext, config_key: str, dest: str) -> str | None:
        value = ctx.param(dest) or ctx.section("backend").get(config_key)
        return str(value) if value else None
