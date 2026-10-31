"""Static checks on docker/Dockerfile and .dockerignore."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DIGEST = "sha256:4449f856653602317e4101a76fce599c7fcd58ccec2e539951fce5f73083179e"
BASE = f"docker.io/rocm/pytorch@{DIGEST}"


@pytest.fixture(scope="module")
def dockerfile() -> str:
    return (REPO / "docker" / "Dockerfile").read_text(encoding="utf-8")


def test_dockerfile_pins_the_base_by_digest(dockerfile):
    lines = dockerfile.splitlines()
    assert f"ARG BASE_IMAGE={BASE}" in lines
    assert "FROM ${BASE_IMAGE}" in lines


def test_dockerfile_installs_all_five_engines(dockerfile):
    assert "ARG ENGINES=swift,trl,easydistill,yolo,ultralytics" in dockerfile.splitlines()


def test_dockerfile_does_not_bake_host_specific_env(dockerfile):
    baked = [
        text
        for text in (
            "ROCR_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
            "HOME=",
            "WORK_DIR=",
            "OUTPUT_DIR=",
        )
        if text in dockerfile
    ]
    assert baked == []


def test_dockerfile_has_no_secret_inputs(dockerfile):
    secret_arg = re.compile(r"(?i)^\s*ARG\s+\S*(PASSWORD|TOKEN|SECRET)")
    assert not [line for line in dockerfile.splitlines() if secret_arg.search(line)]
    assert "docker login" not in dockerfile


def test_dockerfile_runtime_contract(dockerfile):
    for text in (
        "WORKDIR /opt/distill-run",
        "DISTILL_RUN_ENVS_ROOT=/opt/distill-run/.venvs",
        'ENTRYPOINT ["distill-run"]',
        'CMD ["--help"]',
    ):
        assert text in dockerfile


def test_dockerignore_keeps_submodule_sources():
    lines = (REPO / ".dockerignore").read_text(encoding="utf-8").splitlines()
    for pattern in ("**/.git", ".venv", ".venvs", ".distill", "build", "work", "output"):
        assert pattern in lines
    assert not [line for line in lines if line.startswith("third_party")]
