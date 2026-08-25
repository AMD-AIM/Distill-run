"""The easydistill-swift engine hands data from generation to training."""

from __future__ import annotations

from pathlib import Path

import pytest

from distill_run.engines.base import RunResult
from distill_run.engines.easydistill_swift import EasyDistillSwiftEngine
from distill_run.errors import CancelledError
from distill_run.utils import write_jsonl
from tests.unit.test_preflight import make_ctx


def _ctx(tmp_path: Path, config: dict | None = None):
    seed = tmp_path / "seed.jsonl"
    write_jsonl(seed, [{"instruction": "hi"}])
    ctx = make_ctx(
        "easydistill-swift",
        tmp_path,
        {
            "output": str(tmp_path / "run"),
            "student": str(tmp_path / "student"),
            "teacher_base_url": "http://teacher/v1",
            "teacher_model_id": "m",
            "input": str(seed),
        },
        config=config or {},
    )
    (tmp_path / "run").mkdir(parents=True, exist_ok=True)
    from distill_run.dataset import DatasetSpec, ResolvedDataset

    ctx.datasets["input"] = ResolvedDataset(spec=DatasetSpec(uri=str(seed)), path=seed)
    return ctx


class FakeStage:
    def __init__(self, engine: EasyDistillSwiftEngine, attr: str, writes: str | None = None):
        self.calls: list[Path] = []
        self.writes = writes
        setattr(engine, attr, self)

    def run(self, ctx):
        self.calls.append(ctx.output)
        if self.writes:
            target = ctx.output
            target.parent.mkdir(parents=True, exist_ok=True)
            if self.writes == "jsonl":
                write_jsonl(target, [{"messages": [{"role": "user", "content": "x"}]}])
            else:
                target.mkdir(parents=True, exist_ok=True)
        return RunResult(artifacts=[ctx.output], message="fake")

    def preflight(self, ctx):
        return None

    def dataset_specs(self, ctx):
        return {}


def test_stage_one_output_feeds_stage_two(tmp_path: Path):
    engine = EasyDistillSwiftEngine()
    generate = FakeStage(engine, "generate", writes="jsonl")
    train = FakeStage(engine, "train", writes="dir")

    result = engine.run(_ctx(tmp_path))

    sft = tmp_path / "run" / "sft.jsonl"
    assert generate.calls == [sft]
    assert train.calls == [tmp_path / "run" / "checkpoint"]
    assert sft in result.artifacts


def test_resume_reuses_existing_sft_data(tmp_path: Path):
    engine = EasyDistillSwiftEngine()
    generate = FakeStage(engine, "generate", writes="jsonl")
    FakeStage(engine, "train", writes="dir")

    ctx = _ctx(tmp_path)
    sft = tmp_path / "run" / "sft.jsonl"
    write_jsonl(sft, [{"messages": [{"role": "user", "content": "cached"}]}])
    ctx.params.values["resume"] = True

    engine.run(ctx)
    assert generate.calls == [], "generation should be skipped when SFT data already exists"


def test_cancellation_between_stages_stops_before_training(tmp_path: Path):
    engine = EasyDistillSwiftEngine()
    FakeStage(engine, "generate", writes="jsonl")
    train = FakeStage(engine, "train", writes="dir")

    ctx = _ctx(tmp_path)
    ctx.cancellation.request()

    with pytest.raises(CancelledError):
        engine.run(ctx)
    assert train.calls == []


def test_substep_output_overrides_the_top_level_one(tmp_path: Path):
    # The raw --output parameter still names the run directory, so an engine that
    # reads it instead of ctx.output would write the wrong place.
    engine = EasyDistillSwiftEngine()
    ctx = _ctx(tmp_path)

    generate_ctx = engine._generate_ctx(ctx)
    assert generate_ctx.output == tmp_path / "run" / "sft.jsonl"
    assert generate_ctx.param("output") == str(tmp_path / "run")

    train_ctx = engine._train_ctx(ctx, tmp_path / "run" / "sft.jsonl")
    assert train_ctx.output == tmp_path / "run" / "checkpoint"


def test_generation_writes_to_the_context_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from distill_run.engines.easydistill import EasyDistillEngine

    engine = EasyDistillSwiftEngine()
    ctx = _ctx(tmp_path)
    generate_ctx = engine._generate_ctx(ctx)
    generate_ctx.datasets["input"] = ctx.datasets["input"]

    written = {}

    def fake_dispatch(_job_type, config_path):
        import yaml

        written.update(yaml.safe_load(config_path.read_text(encoding="utf-8")))
        target = Path(written["dataset"]["output_path"])
        write_jsonl(target, [{"messages": [{"role": "user", "content": "x"}]}])

    monkeypatch.setattr(EasyDistillEngine, "_dispatch", staticmethod(fake_dispatch))
    EasyDistillEngine().run(generate_ctx)

    assert written["dataset"]["output_path"] == str(tmp_path / "run" / "sft.jsonl")
    assert (tmp_path / "run" / "sft.jsonl").is_file()


def test_config_sections_are_routed_to_the_right_stage(tmp_path: Path):
    engine = EasyDistillSwiftEngine()
    config = {
        "easydistill": {"job_type": "instruct_distill", "generation": {"max_tokens": 7}},
        "swift": {"max_steps": 5},
    }
    ctx = _ctx(tmp_path, config=config)

    generate_ctx = engine._generate_ctx(ctx)
    train_ctx = engine._train_ctx(ctx, tmp_path / "run" / "sft.jsonl")

    assert generate_ctx.engine == "easydistill"
    assert generate_ctx.config["generation"]["max_tokens"] == 7
    assert "swift" not in generate_ctx.config

    assert train_ctx.engine == "swift"
    assert train_ctx.config["swift"] == {"max_steps": 5}
    assert train_ctx.config["dataset"].endswith("sft.jsonl")
