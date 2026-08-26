"""End-to-end CLI behaviour using the framework-free noop engine."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from distill_run.cli import build_context, main
from distill_run.errors import ExitCode, UsageError


def test_successful_run_exits_zero_and_writes_its_artifact(run_dirs):
    work, output = run_dirs
    code = main(
        ["--engine", "noop", "--output", str(output), "--work-dir", str(work), "--run-id", "r1"]
    )
    assert code == ExitCode.OK
    payload = json.loads((output / "noop.json").read_text(encoding="utf-8"))
    assert payload["run_id"] == "r1"
    assert payload["steps"] == 3


def test_missing_required_param_is_a_usage_error(run_dirs):
    work, _ = run_dirs
    assert main(["--engine", "noop", "--work-dir", str(work)]) == ExitCode.USAGE


def test_unknown_engine_is_a_usage_error(run_dirs):
    work, output = run_dirs
    assert (
        main(["--engine", "nope", "--output", str(output), "--work-dir", str(work)])
        == ExitCode.USAGE
    )


def test_foreign_param_for_the_engine_is_a_usage_error(run_dirs):
    work, output = run_dirs
    code = main(
        [
            "--engine",
            "noop",
            "--output",
            str(output),
            "--work-dir",
            str(work),
            "--teacher-base-url",
            "http://x/v1",
        ]
    )
    assert code == ExitCode.USAGE


def test_engine_failure_uses_the_engine_exit_code(run_dirs, tmp_path: Path):
    work, output = run_dirs
    config = tmp_path / "noop.yaml"
    config.write_text(yaml.safe_dump({"steps": 5, "fail_at_step": 2}), encoding="utf-8")
    code = main(
        [
            "--engine",
            "noop",
            "--output",
            str(output),
            "--work-dir",
            str(work),
            "--config",
            str(config),
        ]
    )
    assert code == ExitCode.ENGINE_FAILED


def test_missing_dataset_fails_during_resolve(run_dirs, tmp_path: Path):
    work, output = run_dirs
    student = tmp_path / "student"
    student.mkdir()
    (student / "config.json").write_text("{}", encoding="utf-8")
    code = main(
        [
            "--engine",
            "swift",
            "--model",
            str(student),
            "--dataset",
            str(tmp_path / "absent.jsonl"),
            "--output",
            str(output),
            "--work-dir",
            str(work),
        ]
    )
    assert code == ExitCode.DATASET_FETCH_FAILED


def test_everything_can_come_from_the_environment(run_dirs, monkeypatch: pytest.MonkeyPatch):
    work, output = run_dirs
    monkeypatch.setenv("DISTILL_ENGINE", "noop")
    monkeypatch.setenv("OUTPUT_PATH", str(output))
    monkeypatch.setenv("WORK_DIR", str(work))
    assert main([]) == ExitCode.OK


def test_cancellation_before_the_engine_exits_130(run_dirs, monkeypatch: pytest.MonkeyPatch):
    work, output = run_dirs
    from distill_run import cli
    from distill_run.signals import Cancellation

    already_cancelled = Cancellation()
    already_cancelled.request()
    monkeypatch.setattr(cli, "install_handlers", lambda *a, **k: already_cancelled)

    code = main(["--engine", "noop", "--output", str(output), "--work-dir", str(work)])
    assert code == ExitCode.CANCELLED


def test_command_only_config_uses_defaults_and_overrides(run_dirs):
    work, output = run_dirs
    ctx = build_context(
        {
            "engine": "swift",
            "output": str(output),
            "work_dir": str(work),
            "student": "/models/student",
            "dataset": "/data/sft.jsonl",
            "config_overrides": ["swift.max_steps=6"],
            "max_steps": 7,
            "learning_rate": 2e-5,
            "gradient_checkpointing": False,
        }
    )
    assert ctx.param("num_processes") == 1
    assert ctx.config["swift"]["lora_rank"] == 8
    assert ctx.config["swift"]["max_steps"] == 7
    assert ctx.config["swift"]["learning_rate"] == 2e-5
    assert ctx.config["swift"]["gradient_checkpointing"] is False


def test_default_config_can_be_printed_without_run_arguments(capsys):
    assert main(["--engine", "trl", "--show-default-config"]) == ExitCode.OK
    output = capsys.readouterr().out
    assert "max_steps: 100" in output
    assert "lora_alpha: 32" in output


@pytest.mark.parametrize(
    ("flag", "section"),
    [
        ("--models", "models"),
        ("--datasets", "datasets"),
        ("--engines", "engines"),
    ],
)
def test_capability_catalogs_do_not_require_an_engine(flag, section, capsys):
    assert main([flag]) == ExitCode.OK
    payload = yaml.safe_load(capsys.readouterr().out)
    assert section in payload


def test_engine_catalog_lists_public_frameworks(capsys):
    assert main(["--engines"]) == ExitCode.OK
    engines = yaml.safe_load(capsys.readouterr().out)["engines"]
    assert engines["swift"]["framework"] == "ms-swift"
    assert engines["trl"]["framework"] == "TRL DistillationTrainer"


def test_unknown_engine_points_to_the_engine_catalog():
    with pytest.raises(UsageError, match=r"distill-run --engines"):
        build_context({"engine": "unsupported"})


def test_multi_gpu_command_is_temporarily_disabled():
    assert main(["--engine", "swift", "--num-processes", "2"]) == ExitCode.USAGE
