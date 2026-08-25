"""Engine interface.

An engine is a thin adapter: it gets a validated context with local dataset paths
and translates it into one framework call. Argument parsing, URI resolution and
caching all happen outside, so adding an engine touches one module plus the
registry.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path

from .. import params as P
from ..context import RunContext
from ..dataset import DatasetSpec

_DATASET_PURPOSE = {
    "easydistill": "seed",
    "easydistill-swift": "seed",
    "swift": "sft",
    "trl": "prompt",
}


@dataclass
class RunResult:
    artifacts: list[Path] = field(default_factory=list)
    message: str = ""


class Engine(abc.ABC):
    name: str = "engine"

    def dataset_specs(self, ctx: RunContext) -> dict[str, DatasetSpec]:
        """Declare which data locations must be resolved before :meth:`run`.

        Override when a location can also arrive through ``--config``; the specs
        are resolved to local paths and handed back via ``ctx.dataset(dest)``.
        """
        return specs_from_params(ctx)

    @abc.abstractmethod
    def preflight(self, ctx: RunContext) -> None:
        """Validate config and reachability. Must be cheap and CPU-only."""

    @abc.abstractmethod
    def run(self, ctx: RunContext) -> RunResult:
        """Execute the work and return what was produced."""


def specs_from_params(ctx: RunContext) -> dict[str, DatasetSpec]:
    specs: dict[str, DatasetSpec] = {}
    for param in P.dataset_params_for(ctx.engine):
        uri = ctx.param(param.dest)
        if uri:
            specs[param.dest] = make_spec(ctx, str(uri))
    return specs


def make_spec(ctx: RunContext, uri: str) -> DatasetSpec:
    return DatasetSpec(
        uri=uri,
        format=ctx.param("dataset_format"),
        revision=ctx.param("dataset_revision"),
        subset=ctx.param("dataset_subset"),
        split=ctx.param("dataset_split", "train"),
        max_samples=ctx.param("dataset_max_samples"),
        shuffle_seed=ctx.param("dataset_shuffle_seed", 42),
        purpose=_DATASET_PURPOSE.get(ctx.engine),
    )


def config_dataset(ctx: RunContext) -> str | None:
    """Read a dataset location out of the engine's own config section."""
    section = ctx.config.get("dataset")
    if isinstance(section, str):
        return section
    if isinstance(section, dict):
        value = section.get("path") or section.get("input_path")
        return str(value) if value else None
    return None
