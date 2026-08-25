"""Framework-free engine used to test the CLI itself.

Lets the argument contract, exit codes and cancellation be exercised on a machine
with no GPU and no distillation framework installed.
"""

from __future__ import annotations

import logging
import time

from ..context import RunContext
from ..errors import EngineError
from ..utils import atomic_write_json
from .base import Engine, RunResult

logger = logging.getLogger(__name__)


class NoopEngine(Engine):
    name = "noop"

    def preflight(self, ctx: RunContext) -> None:
        logger.info("noop preflight: output=%s run_id=%s", ctx.output, ctx.run_id)

    def run(self, ctx: RunContext) -> RunResult:
        steps = int(ctx.config.get("steps", 3))
        delay = float(ctx.config.get("step_delay_sec", 0.0))
        fail_at = ctx.config.get("fail_at_step")

        done = 0
        for step in range(1, steps + 1):
            if ctx.cancellation.requested:
                logger.warning("noop engine stopping early after signal")
                break
            if fail_at is not None and step == int(fail_at):
                raise EngineError(f"noop engine failed on purpose at step {step}")
            if delay:
                time.sleep(delay)
            logger.info("noop step %d/%d", step, steps)
            done = step

        artifact = ctx.output_dir / "noop.json"
        atomic_write_json(
            artifact,
            {
                "run_id": ctx.run_id,
                "engine": self.name,
                "steps": done,
                "params": {k: str(v) for k, v in ctx.params.as_dict().items()},
            },
        )
        return RunResult(artifacts=[artifact], message=f"wrote {artifact}")
