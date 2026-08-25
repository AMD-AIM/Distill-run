"""EasyDistill generation followed by Swift student training.

The two stages also exist as independent engines. This combined engine runs:
seed -> teacher API -> SFT JSONL -> student checkpoint, while retaining the
intermediate JSONL under ``--output``.

Both stages are validated up front, so a wrong student path fails before any
tokens are generated.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..context import RunContext
from ..dataset import DatasetSpec, ResolvedDataset
from ..errors import CancelledError
from ..utils import is_nonempty_file, read_jsonl
from .base import Engine, RunResult
from .easydistill import EasyDistillEngine
from .swift import SwiftEngine

logger = logging.getLogger(__name__)

SFT_FILENAME = "sft.jsonl"
CHECKPOINT_DIRNAME = "checkpoint"


class EasyDistillSwiftEngine(Engine):
    name = "easydistill-swift"

    def __init__(self) -> None:
        self.generate = EasyDistillEngine()
        self.train = SwiftEngine()

    def dataset_specs(self, ctx: RunContext) -> dict[str, DatasetSpec]:
        # Only the seed set comes from outside; stage two reads stage one's output.
        return self.generate.dataset_specs(self._generate_ctx(ctx))

    def preflight(self, ctx: RunContext) -> None:
        self.generate.preflight(self._generate_ctx(ctx))
        self.train.preflight(self._train_ctx(ctx, ctx.output_dir / SFT_FILENAME))

    def run(self, ctx: RunContext) -> RunResult:
        sft_path = ctx.output_dir / SFT_FILENAME

        if ctx.resume and is_nonempty_file(sft_path):
            rows = len(read_jsonl(sft_path))
            logger.info("stage 1/2 skipped: reusing %d rows from %s", rows, sft_path)
        else:
            logger.info("stage 1/2: generating SFT data -> %s", sft_path)
            generate_ctx = self._generate_ctx(ctx, output=sft_path)
            generate_ctx.datasets["input"] = ctx.dataset("input")
            first = self.generate.run(generate_ctx)
            logger.info("stage 1/2 done: %s", first.message)

        if ctx.cancellation.requested:
            raise CancelledError("cancelled after generation, before training")

        checkpoint_dir = ctx.output_dir / CHECKPOINT_DIRNAME
        logger.info("stage 2/2: training student -> %s", checkpoint_dir)
        train_ctx = self._train_ctx(ctx, sft_path, output=checkpoint_dir)
        train_ctx.datasets["dataset"] = ResolvedDataset(
            spec=DatasetSpec(uri=str(sft_path)), path=sft_path
        )
        second = self.train.run(train_ctx)
        logger.info("stage 2/2 done: %s", second.message)

        return RunResult(
            artifacts=[sft_path, *second.artifacts],
            message=f"SFT data at {sft_path}, {second.message}",
        )

    def _generate_ctx(self, ctx: RunContext, output: Path | None = None) -> RunContext:
        return ctx.substep(
            engine="easydistill",
            output=output or (ctx.output_dir / SFT_FILENAME),
            config=ctx.section("easydistill"),
        )

    def _train_ctx(self, ctx: RunContext, sft_path: Path, output: Path | None = None) -> RunContext:
        return ctx.substep(
            engine="swift",
            output=output or (ctx.output_dir / CHECKPOINT_DIRNAME),
            config={"swift": ctx.section("swift"), "dataset": str(sft_path)},
        )
