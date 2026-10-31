"""scripts/build-image.sh against a throwaway git repo and a fake docker."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
IMAGE_REPO = "ccr.ccs.tencentyun.com/radeon/distill-run"
BASE_DIGEST = "sha256:4449f856653602317e4101a76fce599c7fcd58ccec2e539951fce5f73083179e"
BASE = f"docker.io/rocm/pytorch@{BASE_DIGEST}"
DIGEST = f"{IMAGE_REPO}@sha256:{'a' * 64}"
SECRET = "r156i-not-a-real-secret"

DOCKER_STUB = f"""#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$DOCKER_STUB_LOG"
case "$1" in
    login) cat > "$DOCKER_STUB_STDIN" ;;
    inspect) echo "{DIGEST}" ;;
esac
"""

GIT_ENV = {
    "GIT_AUTHOR_NAME": "test",
    "GIT_AUTHOR_EMAIL": "test@example.com",
    "GIT_COMMITTER_NAME": "test",
    "GIT_COMMITTER_EMAIL": "test@example.com",
}


def _git(repo: Path, *args: str) -> str:
    env = {**os.environ, **GIT_ENV}
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, env=env).strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    files = {
        "docker/Dockerfile": "FROM scratch\n",
        "src/distill_run/params.py": "PARAMS = ()\n",
        "third_party/trl/pyproject.toml": "",
        "third_party/easydistill/pyproject.toml": "",
        ".gitignore": "build/\n",
    }
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    script = root / "scripts" / "build-image.sh"
    script.parent.mkdir()
    shutil.copy2(REPO / "scripts" / "build-image.sh", script)
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    return root


@pytest.fixture
def docker(tmp_path: Path) -> dict[str, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "docker"
    stub.write_text(DOCKER_STUB, encoding="utf-8")
    stub.chmod(0o755)
    log = tmp_path / "docker.log"
    log.touch()
    return {"bin": bin_dir, "log": log, "stdin": tmp_path / "docker.stdin"}


def _run(repo: Path, docker: dict[str, Path], *args: str, **env: str):
    environ = {
        key: value
        for key, value in os.environ.items()
        if key not in {"IMAGE_REPO", "BASE_IMAGE", "ENGINES", "CCR_USERNAME", "CCR_PASSWORD"}
    }
    environ.update(
        PATH=f"{docker['bin']}{os.pathsep}{os.environ['PATH']}",
        DOCKER_STUB_LOG=str(docker["log"]),
        DOCKER_STUB_STDIN=str(docker["stdin"]),
        **env,
    )
    return subprocess.run(
        ["bash", str(repo / "scripts" / "build-image.sh"), *args],
        cwd=repo,
        env=environ,
        capture_output=True,
        text=True,
    )


def _calls(docker: dict[str, Path]) -> list[str]:
    return docker["log"].read_text(encoding="utf-8").splitlines()


def _params_sha(repo: Path) -> str:
    return hashlib.sha256((repo / "src/distill_run/params.py").read_bytes()).hexdigest()


def test_refuses_a_worktree_with_untracked_files(repo, docker):
    (repo / "stray.txt").write_text("x\n", encoding="utf-8")
    result = _run(repo, docker)
    assert result.returncode == 1
    assert "uncommitted changes" in result.stderr
    assert _calls(docker) == []


def test_refuses_missing_submodule_sources(repo, docker):
    _git(repo, "rm", "-q", "third_party/trl/pyproject.toml")
    _git(repo, "commit", "-q", "-m", "drop trl")
    result = _run(repo, docker)
    assert result.returncode == 3
    assert _calls(docker) == []


def test_no_push_builds_and_tags_without_pushing(repo, docker):
    result = _run(repo, docker, "--no-push")
    assert result.returncode == 0, result.stderr
    sha = _git(repo, "rev-parse", "HEAD")
    tag = f"{_git(repo, 'rev-parse', '--short=12', 'HEAD')}-gfx1100"
    params = _params_sha(repo)
    builds = [call for call in _calls(docker) if call.startswith("build ")]
    assert len(builds) == 1
    for text in (
        "-f docker/Dockerfile",
        f"-t {IMAGE_REPO}:{tag}",
        f"--build-arg BASE_IMAGE={BASE}",
        f"--build-arg VCS_REF={sha}",
        f"--build-arg PARAMS_SHA256={params}",
    ):
        assert text in builds[0]
    assert not [call for call in _calls(docker) if call.startswith("push")]
    assert result.stdout.splitlines()[-1] == f"PARAMS_SHA256={params}"


def test_push_records_the_digest(repo, docker):
    result = _run(repo, docker)
    assert result.returncode == 0, result.stderr
    tag = f"{_git(repo, 'rev-parse', '--short=12', 'HEAD')}-gfx1100"
    assert f"push {IMAGE_REPO}:{tag}" in _calls(docker)
    assert result.stdout.splitlines()[-3:] == [
        f"IMAGE={IMAGE_REPO}:{tag}",
        f"DIGEST={DIGEST}",
        f"PARAMS_SHA256={_params_sha(repo)}",
    ]
    row = (repo / "build" / "image-digests.tsv").read_text(encoding="utf-8").splitlines()[-1]
    fields = row.split("\t")
    assert len(fields) == 5
    assert fields[1] == tag
    assert fields[3] == DIGEST


def test_password_only_via_stdin(repo, docker):
    result = _run(repo, docker, CCR_PASSWORD=SECRET)
    assert result.returncode == 0, result.stderr
    assert "login ccr.ccs.tencentyun.com --username 100053684421 --password-stdin" in _calls(docker)
    assert docker["stdin"].read_text(encoding="utf-8") == SECRET
    tsv = (repo / "build" / "image-digests.tsv").read_text(encoding="utf-8")
    for text in (docker["log"].read_text(encoding="utf-8"), result.stdout, result.stderr, tsv):
        assert SECRET not in text


def test_never_tags_latest(repo, docker):
    result = _run(repo, docker)
    assert result.returncode == 0, result.stderr
    assert ":latest" not in docker["log"].read_text(encoding="utf-8")
