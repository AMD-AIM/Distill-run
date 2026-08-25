from __future__ import annotations

import json
from pathlib import Path

from distill_run.doctor import main as doctor_main
from distill_run.runtime_env import maybe_run_in_engine_environment


def test_engine_command_relaunches_with_configured_python(tmp_path: Path, monkeypatch):
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    monkeypatch.setenv("DISTILL_RUN_SWIFT_PYTHON", str(python))
    captured = {}

    def fake_call(command, env):
        captured["command"] = command
        captured["env"] = env
        return 17

    monkeypatch.setattr("distill_run.runtime_env.subprocess.call", fake_call)
    code = maybe_run_in_engine_environment("swift", ["--engine", "swift"])
    assert code == 17
    assert captured["command"][0] == str(python)
    assert captured["env"]["DISTILL_RUN_ENGINE_WORKER"] == "1"


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
