"""Student SFT via ms-swift, driven as a subprocess.

ms-swift owns its own distributed launcher, so it runs as a child process with its
output forwarded line by line. Two things bit us and are guarded here: the
hyper-parameter names are checked against the installed ``SftArguments`` before
the model loads, and ``NPROC_PER_NODE`` is left unset for single-process runs
because ms-swift switches to torchrun the moment it sees that variable.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

from .. import launcher, preflight
from ..context import RunContext
from ..errors import ConfigError, EngineError, OutOfMemoryError
from ..utils import dir_has_entries, stream_command
from .base import Engine, RunResult, config_dataset, make_spec, specs_from_params

logger = logging.getLogger(__name__)

_CHECKPOINT = re.compile(r"^checkpoint-(\d+)$")
_OOM_MARKERS = ("out of memory", "OutOfMemoryError")

# `tuner_type` is the ms-swift 4.x name; it was `train_type` in 3.x.
DEFAULT_ARGS: dict[str, Any] = {
    "tuner_type": "lora",
    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,
    "learning_rate": 1e-4,
    "logging_steps": 1,
    "save_steps": 100,
    "torch_dtype": "bfloat16",
}


class SwiftEngine(Engine):
    name = "swift"

    def dataset_specs(self, ctx: RunContext):
        specs = specs_from_params(ctx)
        if "dataset" not in specs:
            from_config = config_dataset(ctx)
            if not from_config:
                raise ConfigError(
                    "no training data: pass --dataset (or DATASET_URI), "
                    "or set dataset.path in --config"
                )
            specs["dataset"] = make_spec(ctx, from_config)
        return specs

    def preflight(self, ctx: RunContext) -> None:
        preflight.check_weights(Path(ctx.require("student")), "student")
        preflight.check_arg_names(
            config_args(ctx),
            preflight.swift_arg_names(),
            label="ms-swift sft",
            config_key="swift",
        )
        nproc = launcher.world_size(ctx.param("num_processes"))
        logger.info(
            "swift will train with %d process(es) (%d GPU(s) visible)",
            nproc,
            launcher.visible_device_count(),
        )

    def run(self, ctx: RunContext) -> RunResult:
        dataset = ctx.dataset("dataset").path
        nproc = launcher.world_size(ctx.param("num_processes"))
        check_distributed_dataset_size(dataset, nproc)
        output = ctx.output
        output.mkdir(parents=True, exist_ok=True)
        argv = self._build_argv(ctx, dataset, output)

        oom = False

        def watch(line: str) -> None:
            nonlocal oom
            if any(marker in line for marker in _OOM_MARKERS):
                oom = True

        # Only set NPROC_PER_NODE when we actually want a distributed run: its mere
        # presence makes ms-swift launch through torchrun.
        env = {}
        if nproc > 1:
            env["NPROC_PER_NODE"] = str(nproc)
        if ctx.param("main_process_port") is not None:
            env["MASTER_PORT"] = str(ctx.param("main_process_port"))

        code = stream_command(argv, env=env or None, on_line=watch)
        if code != 0:
            if oom:
                raise OutOfMemoryError(f"swift sft ran out of memory (exit {code})")
            raise EngineError(f"swift sft exited with {code}")
        checkpoint = require_checkpoint(output)
        return RunResult(
            artifacts=[checkpoint],
            message=f"checkpoint at {checkpoint}",
        )

    def _build_argv(self, ctx: RunContext, dataset: Path, output: Path) -> list[str]:
        args: dict[str, Any] = {**DEFAULT_ARGS, **config_args(ctx)}
        args["model"] = str(ctx.require("student"))
        args["dataset"] = str(dataset)
        args["output_dir"] = str(output)

        if ctx.resume:
            latest = latest_checkpoint(output)
            if latest is None:
                logger.warning("--resume requested but no checkpoint found under %s", output)
            else:
                logger.info("resuming from %s", latest)
                args["resume_from_checkpoint"] = str(latest)

        configured = os.environ.get("DISTILL_RUN_SWIFT_EXECUTABLE")
        in_venv = Path(sys.executable).parent / "swift"
        executable = configured or (str(in_venv) if in_venv.is_file() else "swift")
        argv = [executable, "sft"]
        for key, value in args.items():
            if value is None:
                continue
            argv.append(f"--{key}")
            if isinstance(value, bool):
                argv.append("true" if value else "false")
            elif isinstance(value, (list, tuple)):
                argv.extend(str(v) for v in value)
            else:
                argv.append(str(value))
        return argv


def latest_checkpoint(output: Path) -> Path | None:
    """Highest-numbered ``checkpoint-N`` directory, searching one level down.

    ms-swift nests checkpoints under a run directory it names itself, so the
    direct children of ``--output`` are checked as well as their children.
    """
    if not output.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for parent in (output, *(p for p in output.iterdir() if p.is_dir())):
        for child in parent.iterdir():
            match = _CHECKPOINT.match(child.name) if child.is_dir() else None
            if match:
                candidates.append((int(match.group(1)), child))
    return max(candidates)[1] if candidates else None


def require_checkpoint(output: Path) -> Path:
    checkpoint = latest_checkpoint(output)
    if checkpoint is None:
        detail = "output is empty" if not dir_has_entries(output) else "no checkpoint was created"
        raise EngineError(
            f"swift sft exited successfully but {detail} under {output}; "
            "the dataset may be smaller than the distributed world size"
        )

    state_path = checkpoint / "trainer_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EngineError(f"Swift checkpoint is missing a valid {state_path.name}: {exc}") from exc
    step = state.get("global_step", 0)
    if not isinstance(step, int) or step < 1:
        raise EngineError(
            f"Swift checkpoint reports global_step={step}; no training step completed"
        )

    patterns = ("adapter_model.*", "model*.safetensors", "pytorch_model*.bin")
    weights = [
        path
        for pattern in patterns
        for path in checkpoint.glob(pattern)
        if path.is_file() and path.stat().st_size > 0
    ]
    if not weights:
        raise EngineError(f"Swift checkpoint {checkpoint} contains no non-empty model weights")
    return checkpoint


def check_distributed_dataset_size(dataset: Path, nproc: int) -> None:
    if nproc <= 1 or not dataset.is_file():
        return
    with dataset.open(encoding="utf-8") as handle:
        rows = sum(1 for line in handle if line.strip())
    if rows < nproc:
        raise ConfigError(
            f"training dataset has {rows} row(s), fewer than the {nproc} Swift workers; "
            "generate more examples or reduce --num-processes"
        )


def config_args(ctx: RunContext) -> dict[str, Any]:
    section = ctx.config.get("swift", ctx.config.get("train", {}))
    if not isinstance(section, dict):
        raise ConfigError("config key 'swift' must be a mapping of swift sft arguments")
    return dict(section)
