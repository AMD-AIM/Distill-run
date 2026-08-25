"""Real-framework smoke tests. Need GPUs, weights and (for black box) a teacher.

Skipped by default so ``pytest`` stays green anywhere::

    DISTILL_SMOKE=1 \
      STUDENT=/models/Qwen3-0.6B TEACHER=/models/Qwen3.8-27B \
      TEACHER_BASE_URL=http://127.0.0.1:8000/v1 \
      TEACHER_MODEL_ID=/models/Qwen3.8-27B \
      pytest tests/smoke -m gpu

Prefer ``./run.sh smoke``, which sets all of this from env.sh.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from distill_run.cli import main
from distill_run.errors import ExitCode

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(os.environ.get("DISTILL_SMOKE") != "1", reason="set DISTILL_SMOKE=1"),
]

REPO = Path(__file__).resolve().parents[2]
SEED = REPO / "examples" / "seed_instructions.jsonl"
CONFIGS = REPO / "configs"


def _path_env(name: str) -> str:
    value = os.environ.get(name)
    if not value or not Path(value).exists():
        pytest.skip(f"{name} is not set to an existing path")
    return value


def _teacher_args() -> list[str]:
    base_url = os.environ.get("TEACHER_BASE_URL")
    if not base_url:
        pytest.skip("TEACHER_BASE_URL is not set")
    return [
        "--teacher-base-url",
        base_url,
        "--teacher-model-id",
        os.environ.get("TEACHER_MODEL_ID", "teacher"),
    ]


def test_generation_produces_sft_rows(tmp_path: Path):
    output = tmp_path / "sft.jsonl"
    code = main(
        [
            "--engine",
            "easydistill",
            "--config",
            str(CONFIGS / "easydistill" / "instruct_distill.yaml"),
            *_teacher_args(),
            "--input",
            str(SEED),
            "--output",
            str(output),
            "--work-dir",
            str(tmp_path / "work"),
        ]
    )
    assert code == ExitCode.OK
    assert output.stat().st_size > 0


def test_student_sft_produces_a_checkpoint(tmp_path: Path):
    student = _path_env("STUDENT")
    dataset = os.environ.get("SFT_DATASET")
    if not dataset:
        pytest.skip("SFT_DATASET is not set; run the generation smoke test first")
    output = tmp_path / "sft-run"
    code = main(
        [
            "--engine",
            "swift",
            "--config",
            str(CONFIGS / "smoke" / "swift_sft.yaml"),
            "--model",
            student,
            "--dataset",
            dataset,
            "--output",
            str(output),
            "--work-dir",
            str(tmp_path / "work"),
        ]
    )
    assert code == ExitCode.OK


def test_easydistill_swift_runs_both_stages(tmp_path: Path):
    student = _path_env("STUDENT")
    output = tmp_path / "easydistill-swift-run"
    code = main(
        [
            "--engine",
            "easydistill-swift",
            "--config",
            str(CONFIGS / "easydistill-swift" / "smoke.yaml"),
            *_teacher_args(),
            "--model",
            student,
            "--input",
            str(SEED),
            "--output",
            str(output),
            "--work-dir",
            str(tmp_path / "work"),
        ]
    )
    assert code == ExitCode.OK
    assert (output / "sft.jsonl").stat().st_size > 0
    assert (output / "checkpoint").is_dir()


def test_white_box_distillation_produces_an_adapter(tmp_path: Path):
    student, teacher = _path_env("STUDENT"), _path_env("TEACHER")
    output = tmp_path / "trl-run"
    code = main(
        [
            "--engine",
            "trl",
            "--config",
            str(CONFIGS / "smoke" / "trl_distill.yaml"),
            "--teacher",
            teacher,
            "--student",
            student,
            "--data",
            str(SEED),
            "--output",
            str(output),
            "--work-dir",
            str(tmp_path / "work"),
        ]
    )
    assert code == ExitCode.OK
    assert (output / "final").is_dir()
