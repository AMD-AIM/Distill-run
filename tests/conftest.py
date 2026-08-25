from __future__ import annotations

import os
from pathlib import Path

import pytest

_CONTRACT_ENV = {
    "OUTPUT_PATH",
    "CONFIG_PATH",
    "RUN_ID",
    "WORK_DIR",
    "INPUT_URI",
    "STUDENT_MODEL",
    "TEACHER_MODEL",
    "NPROC_PER_NODE",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "MASTER_PORT",
    "ACCELERATE_CONFIG_FILE",
}


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Keep tests off real paths and away from inherited configuration."""
    for var in list(os.environ):
        if var.startswith(("DISTILL_", "TEACHER_", "DATASET_")) or var in _CONTRACT_ENV:
            monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("WORK_DIR", str(tmp_path / "work"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))


@pytest.fixture
def run_dirs(tmp_path: Path) -> tuple[Path, Path]:
    work = tmp_path / "work"
    output = tmp_path / "output"
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    return work, output
