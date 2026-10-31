from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from distill_run.doctor import main as doctor_main
from distill_run.runtime_env import maybe_run_in_engine_environment


def test_engine_command_replaces_the_process_with_the_engine_python(tmp_path: Path, monkeypatch):
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    monkeypatch.setenv("DISTILL_RUN_SWIFT_PYTHON", str(python))
    monkeypatch.setattr("distill_run.runtime_env.os.name", "posix")
    captured = {}

    class Replaced(Exception):
        pass

    def fake_execve(path, command, env):
        captured.update(path=path, command=command, env=env)
        raise Replaced

    def no_child(*_args, **_kwargs):
        raise AssertionError("a child process would not receive SIGTERM from the pod")

    monkeypatch.setattr("distill_run.runtime_env.os.execve", fake_execve, raising=False)
    monkeypatch.setattr("distill_run.runtime_env.subprocess.call", no_child)
    with pytest.raises(Replaced):
        maybe_run_in_engine_environment("swift", ["--engine", "swift"])
    assert captured["path"] == str(python)
    assert captured["command"] == [str(python), "-m", "distill_run", "--engine", "swift"]
    assert captured["env"]["DISTILL_RUN_ENGINE_WORKER"] == "1"


@pytest.mark.skipif(os.name != "posix", reason="exec and POSIX signals")
def test_sigterm_reaches_the_engine_worker(tmp_path: Path):
    """The relaunched worker, not a waiting parent, must own the pid that gets SIGTERM."""
    envs = tmp_path / "envs"
    venv = envs / "ultralytics"
    (venv / "bin").mkdir(parents=True)
    (venv / ".distill-run-runtime").write_text(str(venv.resolve()), encoding="utf-8")
    worker = venv / "bin" / "python"
    worker.write_text(
        "#!/bin/sh\n"
        "trap 'echo worker-got-term; exit 130' TERM\n"
        "echo worker-ready\n"
        "while :; do sleep 0.1; done\n",
        encoding="utf-8",
    )
    worker.chmod(0o755)
    code = (
        "import sys; from distill_run.runtime_env import maybe_run_in_engine_environment as m; "
        "sys.exit(m('ultralytics', ['--engine', 'ultralytics']) or 0)"
    )
    env = {k: v for k, v in os.environ.items() if k != "DISTILL_RUN_ENGINE_WORKER"}
    env["DISTILL_RUN_ENVS_ROOT"] = str(envs)
    proc = subprocess.Popen(
        [sys.executable, "-c", code], env=env, stdout=subprocess.PIPE, text=True
    )
    assert proc.stdout.readline().strip() == "worker-ready"
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=10) == 130
    assert "worker-got-term" in proc.stdout.read()


def test_engine_worker_does_not_relaunch(monkeypatch):
    monkeypatch.setenv("DISTILL_RUN_ENGINE_WORKER", "1")
    assert maybe_run_in_engine_environment("trl", ["--engine", "trl"]) is None


def test_doctor_reports_a_missing_engine_environment(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("DISTILL_RUN_ENVS_ROOT", str(tmp_path))
    assert doctor_main(["--engine", "swift", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    swift = next(check for check in payload if check["name"] == "swift")
    assert swift["ok"] is False
    assert "environment missing" in swift["detail"]
