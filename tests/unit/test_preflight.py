"""Preflight must catch mistakes on CPU, before weights or tokens are spent."""

from __future__ import annotations

import json
import multiprocessing
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from distill_run import preflight
from distill_run.config import merge_sources
from distill_run.context import RunContext
from distill_run.engines.easydistill import EasyDistillEngine
from distill_run.engines.easydistill_swift import EasyDistillSwiftEngine
from distill_run.engines.swift import (
    SwiftEngine,
    check_distributed_dataset_size,
    latest_checkpoint,
    require_checkpoint,
)
from distill_run.engines.trl import (
    DEFAULT_MODEL_INIT,
    TrlEngine,
    check_model_compatibility,
    has_tied_embeddings,
)
from distill_run.errors import ConfigError, EngineError, PreflightError, TeacherUnavailableError
from distill_run.signals import Cancellation


def make_ctx(engine: str, tmp_path: Path, args: dict, config: dict | None = None) -> RunContext:
    params = merge_sources(engine, {"engine": engine, **args}, {})
    output = Path(args.get("output", tmp_path / "out"))
    output.parent.mkdir(parents=True, exist_ok=True)
    return RunContext(
        engine=engine,
        run_id="t1",
        params=params,
        config=config or {},
        work_dir=tmp_path / "work",
        output=output,
        cache_dir=tmp_path / "cache",
        model_cache_dir=tmp_path / "models",
        cancellation=Cancellation(),
    )


class TeacherUp:
    def __init__(self, served=("m",)):
        self.served = served
        self.url = None

    def get(self, url, **_kwargs):
        self.url = url
        served = self.served

        class R:
            def raise_for_status(self):
                return None

            def json(self):
                return {"data": [{"id": s} for s in served]}

        return R()


def weights(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    return path


def model_config(tmp_path: Path, name: str, vocab_size: int) -> Path:
    path = tmp_path / name
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "vocab_size": vocab_size}),
        encoding="utf-8",
    )
    return path


# --- teacher endpoint --------------------------------------------------------


def test_easydistill_requires_a_teacher_endpoint(tmp_path: Path):
    ctx = make_ctx("easydistill", tmp_path, {"output": str(tmp_path / "sft.jsonl")})
    with pytest.raises(ConfigError, match="teacher endpoint missing"):
        EasyDistillEngine().preflight(ctx)


def test_easydistill_fails_when_teacher_is_unreachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class Down:
        def get(self, *_a, **_k):
            raise OSError("connection refused")

    monkeypatch.setattr(preflight, "http_session", lambda: Down())
    ctx = make_ctx(
        "easydistill",
        tmp_path,
        {
            "output": str(tmp_path / "sft.jsonl"),
            "teacher_base_url": "http://teacher/v1",
            "teacher_model_id": "m",
        },
    )
    with pytest.raises(TeacherUnavailableError, match="not reachable"):
        EasyDistillEngine().preflight(ctx)


def test_easydistill_accepts_a_teacher_declared_only_in_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    up = TeacherUp()
    monkeypatch.setattr(preflight, "http_session", lambda: up)
    ctx = make_ctx(
        "easydistill",
        tmp_path,
        {"output": str(tmp_path / "sft.jsonl")},
        config={"backend": {"base_url": "http://teacher/v1", "model_id": "m"}},
    )
    EasyDistillEngine().preflight(ctx)
    assert up.url == "http://teacher/v1/models"


def test_easydistill_needs_seed_data_from_somewhere(tmp_path: Path):
    ctx = make_ctx("easydistill", tmp_path, {"output": str(tmp_path / "sft.jsonl")})
    with pytest.raises(ConfigError, match="no seed data"):
        EasyDistillEngine().dataset_specs(ctx)


# --- weights -----------------------------------------------------------------


def test_swift_rejects_a_missing_student_path(tmp_path: Path):
    ctx = make_ctx(
        "swift", tmp_path, {"output": str(tmp_path / "ckpt"), "student": str(tmp_path / "absent")}
    )
    with pytest.raises(
        PreflightError,
        match=r"(?s)student weights path.*(available local models|distill-run --models)",
    ):
        SwiftEngine().preflight(ctx)


def test_swift_rejects_an_empty_student_dir(tmp_path: Path):
    empty = tmp_path / "student"
    empty.mkdir()
    ctx = make_ctx("swift", tmp_path, {"output": str(tmp_path / "ckpt"), "student": str(empty)})
    with pytest.raises(PreflightError, match="empty"):
        SwiftEngine().preflight(ctx)


# --- framework hyper-parameter names ----------------------------------------


def test_unknown_hyperparameter_name_is_rejected_with_a_suggestion():
    # ms-swift renamed train_type to tuner_type; the framework itself only notices
    # after loading the model, so the name is checked here instead.
    with pytest.raises(ConfigError, match="tuner_type"):
        preflight.check_arg_names(
            {"train_type": "lora"},
            {"tuner_type", "lora_rank", "max_steps"},
            label="ms-swift sft",
            config_key="swift",
        )


def test_known_hyperparameter_names_pass():
    preflight.check_arg_names(
        {"tuner_type": "lora", "max_steps": 3},
        {"tuner_type", "max_steps"},
        label="ms-swift sft",
        config_key="swift",
    )


def test_name_check_is_skipped_when_the_framework_is_absent():
    # On a machine without ms-swift the field list is empty; skip rather than
    # reject everything the user wrote.
    preflight.check_arg_names({"anything": 1}, set(), label="x", config_key="swift")


def test_swift_preflight_catches_a_bad_hyperparameter_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(preflight, "swift_arg_names", lambda: {"tuner_type", "max_steps"})
    ctx = make_ctx(
        "swift",
        tmp_path,
        {"output": str(tmp_path / "ckpt"), "student": str(weights(tmp_path, "student"))},
        config={"swift": {"train_type": "lora"}},
    )
    with pytest.raises(ConfigError, match="does not accept"):
        SwiftEngine().preflight(ctx)


# --- easydistill-swift validates both stages up front ------------------------


def test_easydistill_swift_checks_the_student_before_generating_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(preflight, "http_session", lambda: TeacherUp())
    monkeypatch.setattr(preflight, "swift_arg_names", lambda: set())
    ctx = make_ctx(
        "easydistill-swift",
        tmp_path,
        {
            "output": str(tmp_path / "run"),
            "student": str(tmp_path / "no-such-student"),
            "teacher_base_url": "http://teacher/v1",
            "teacher_model_id": "m",
            "input": str(tmp_path / "seed.jsonl"),
        },
    )
    with pytest.raises(PreflightError, match="student weights path"):
        EasyDistillSwiftEngine().preflight(ctx)


def test_easydistill_swift_passes_when_both_stages_are_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(preflight, "http_session", lambda: TeacherUp())
    monkeypatch.setattr(preflight, "swift_arg_names", lambda: set())
    ctx = make_ctx(
        "easydistill-swift",
        tmp_path,
        {
            "output": str(tmp_path / "run"),
            "student": str(weights(tmp_path, "student")),
            "teacher_base_url": "http://teacher/v1",
            "teacher_model_id": "m",
            "input": str(tmp_path / "seed.jsonl"),
        },
    )
    EasyDistillSwiftEngine().preflight(ctx)


# --- checkpoint discovery ----------------------------------------------------


def test_latest_checkpoint_picks_the_highest_step(tmp_path: Path):
    for step in (5, 40, 200):
        (tmp_path / f"checkpoint-{step}").mkdir()
    (tmp_path / "final").mkdir()
    assert latest_checkpoint(tmp_path).name == "checkpoint-200"


def test_latest_checkpoint_looks_inside_the_run_dir_swift_creates(tmp_path: Path):
    nested = tmp_path / "v0-20260824-1200"
    (nested / "checkpoint-12").mkdir(parents=True)
    assert latest_checkpoint(tmp_path).name == "checkpoint-12"


def test_latest_checkpoint_returns_none_when_absent(tmp_path: Path):
    assert latest_checkpoint(tmp_path / "nothing") is None


def test_swift_rejects_success_without_a_checkpoint(tmp_path: Path):
    output = tmp_path / "swift-output"
    output.mkdir()
    (output / "args.json").write_text("{}", encoding="utf-8")
    with pytest.raises(EngineError, match="no checkpoint was created"):
        require_checkpoint(output)


def test_swift_accepts_checkpoint_with_steps_and_weights(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint-2"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 2}),
        encoding="utf-8",
    )
    (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
    assert require_checkpoint(tmp_path) == checkpoint


def test_swift_rejects_checkpoint_with_zero_steps(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint-0"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 0}),
        encoding="utf-8",
    )
    (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
    with pytest.raises(EngineError, match="global_step=0"):
        require_checkpoint(tmp_path)


def test_swift_rejects_checkpoint_without_weights(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 1}),
        encoding="utf-8",
    )
    with pytest.raises(EngineError, match="no non-empty model weights"):
        require_checkpoint(tmp_path)


def test_swift_rejects_dataset_smaller_than_worker_count(tmp_path: Path):
    dataset = tmp_path / "one-row.jsonl"
    dataset.write_text('{"messages": []}\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="fewer than the 2 Swift workers"):
        check_distributed_dataset_size(dataset, 2)


def test_trl_does_not_default_to_inference_style_device_map():
    assert "device_map" not in DEFAULT_MODEL_INIT


def test_trl_rejects_mismatched_vocabularies_before_loading_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(path: Path, **_kwargs):
            raw = json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
            return SimpleNamespace(
                get_text_config=lambda: SimpleNamespace(vocab_size=raw["vocab_size"])
            )

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoConfig=FakeAutoConfig),
    )
    student = model_config(tmp_path, "student-config", 100)
    teacher = model_config(tmp_path, "teacher-config", 200)
    with pytest.raises(ConfigError, match="vocabulary mismatch"):
        check_model_compatibility(student, teacher)


def test_trl_rejects_device_map_under_fsdp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    student = weights(tmp_path, "trl-student")
    teacher = weights(tmp_path, "trl-teacher")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(
        "distill_run.engines.trl.check_model_compatibility",
        lambda *_args: None,
    )
    monkeypatch.setattr(preflight, "trl_arg_names", lambda: set())
    ctx = make_ctx(
        "trl",
        tmp_path,
        {
            "output": str(tmp_path / "trl-out"),
            "student": str(student),
            "teacher": str(teacher),
        },
        config={"teacher_model_init": {"device_map": "auto"}},
    )
    with pytest.raises(ConfigError, match="FSDP owns model placement"):
        TrlEngine().preflight(ctx)


def test_tied_embedding_detection_supports_nested_text_config(tmp_path: Path):
    model = tmp_path / "qwen35"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"text_config": {"tie_word_embeddings": True}}),
        encoding="utf-8",
    )
    assert has_tied_embeddings(model)


def test_trl_rejects_tied_embeddings_under_fsdp2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    student = model_config(tmp_path, "tied-student", 100)
    teacher = model_config(tmp_path, "untied-teacher", 100)
    student_config = json.loads((student / "config.json").read_text(encoding="utf-8"))
    student_config["tie_word_embeddings"] = True
    (student / "config.json").write_text(json.dumps(student_config), encoding="utf-8")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(preflight, "trl_arg_names", lambda: set())
    monkeypatch.setattr(
        "distill_run.engines.trl.check_model_compatibility",
        lambda *_args: None,
    )
    ctx = make_ctx(
        "trl",
        tmp_path,
        {
            "output": str(tmp_path / "trl-out"),
            "student": str(student),
            "teacher": str(teacher),
            "distributed_strategy": "fsdp2",
        },
    )
    with pytest.raises(ConfigError, match="tied input/output embeddings"):
        TrlEngine().preflight(ctx)


def _writable_probe_worker(path: str) -> None:
    preflight.check_writable(Path(path), "shared output")


def test_writable_probe_is_safe_across_ranks(tmp_path: Path):
    output = tmp_path / "shared"
    workers = [
        multiprocessing.Process(target=_writable_probe_worker, args=(str(output),))
        for _ in range(4)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=10)
        assert worker.exitcode == 0
    assert list(output.glob(".distill-run-write-probe-*")) == []
