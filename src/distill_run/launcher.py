"""Multi-GPU launch helpers."""

from __future__ import annotations

import logging
import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from .errors import UsageError
from .utils import stream_command

if TYPE_CHECKING:
    from .context import RunContext

logger = logging.getLogger(__name__)


def visible_device_count() -> int:
    for var in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        value = os.environ.get(var)
        if value:
            return len([d for d in value.split(",") if d.strip() != ""])
    return 0


def world_size(requested: int | None = None) -> int:
    """Processes per node, which must be requested explicitly.

    Deliberately *not* derived from the visible device count: a container that can
    see six GPUs should not silently turn a small run into a six-process
    distributed job. Ask for it with ``NPROC_PER_NODE``.
    """
    explicit = requested or os.environ.get("NPROC_PER_NODE") or os.environ.get("WORLD_SIZE")
    if not explicit:
        return 1
    try:
        requested = max(1, int(explicit))
    except ValueError:
        logger.warning("ignoring non-numeric NPROC_PER_NODE=%r; using 1 process", explicit)
        return 1
    visible = visible_device_count()
    if visible and requested > visible:
        logger.warning(
            "NPROC_PER_NODE=%d exceeds the %d visible device(s); capping to %d",
            requested,
            visible,
            visible,
        )
        return visible
    return requested


def in_distributed_worker() -> bool:
    try:
        return int(os.environ.get("WORLD_SIZE", "1")) > 1
    except ValueError:
        return "LOCAL_RANK" in os.environ


def configure_single_device(ctx: RunContext) -> None:
    """Prevent frameworks from treating all visible GPUs as one-process DataParallel."""
    if ctx.engine not in {"swift", "trl", "easydistill-swift"}:
        return
    if world_size(ctx.param("num_processes")) != 1 or in_distributed_worker():
        return
    os.environ["HIP_VISIBLE_DEVICES"] = str(ctx.param("device", 0))


def launch_trl_if_needed(ctx: RunContext, child_argv: Sequence[str]) -> int | None:
    """Relaunch a multi-GPU TRL command under Accelerate from the public CLI."""
    processes = world_size(ctx.param("num_processes"))
    if ctx.engine != "trl" or processes <= 1 or in_distributed_worker():
        return None

    custom_config = ctx.param("accelerate_config")
    strategy = str(ctx.param("distributed_strategy", "zero3")).lower()
    if custom_config:
        config = Path(custom_config)
    else:
        configs = {
            "fsdp2": "fsdp2.yaml",
            "zero1": "deepspeed_zero1.yaml",
            "zero2": "deepspeed_zero2.yaml",
            "zero3": "deepspeed_zero3.yaml",
        }
        if strategy not in configs:
            raise UsageError("--distributed-strategy must be one of: fsdp2, zero1, zero2, zero3")
        config = Path(__file__).resolve().parents[2] / "configs" / "accelerate" / configs[strategy]

    if not config.is_file():
        raise UsageError(f"Accelerate config file not found: {config}")
    environment_accelerate = Path(sys.executable).parent / "accelerate"
    accelerate = (
        str(environment_accelerate)
        if environment_accelerate.is_file()
        else shutil.which("accelerate")
    )
    if accelerate is None:
        raise UsageError("accelerate executable not found; install Accelerate for multi-GPU TRL")

    forwarded = list(child_argv)
    if ctx.params.origin("run_id") is None:
        forwarded.extend(["--run-id", ctx.run_id])
    command = [
        accelerate,
        "launch",
        "--config_file",
        str(config),
        "--num_processes",
        str(processes),
        "--main_process_port",
        str(ctx.param("main_process_port", 29500)),
        "--module",
        "distill_run",
        *forwarded,
    ]
    logger.info("launching TRL with %d process(es) via %s", processes, config.name)
    return stream_command(command)


def wrap_with_torchrun(argv: Sequence[str], nproc: int) -> list[str]:
    if nproc <= 1:
        return list(argv)
    return [
        "torchrun",
        f"--nproc_per_node={nproc}",
        "--nnodes=1",
        f"--master_port={os.environ.get('MASTER_PORT', '29500')}",
        *argv,
    ]
