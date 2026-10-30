"""Service-facing contract: integrity checks, URL refresh, events, train_args, report_to."""

from __future__ import annotations

import hashlib
import http.server
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from distill_run import events
from distill_run.cli import build_context, main
from distill_run.dataset import DatasetSpec, resolve
from distill_run.engines.ultralytics import ReportTarget, check_train_args
from distill_run.errors import (
    ConfigError,
    DatasetFetchError,
    ExitCode,
    IntegrityError,
    UsageError,
)
from distill_run.integrity import parse_digest, verify


@pytest.fixture(autouse=True)
def _fresh_logger():
    import logging

    ours = logging.getLogger("distill_run")
    saved = list(ours.handlers)
    ours.handlers = []
    yield
    ours.handlers = saved


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_digest_forms_and_rejections():
    hexd = "ab" * 32
    assert parse_digest(f"sha256:{hexd.upper()}") == hexd
    assert parse_digest(hexd) == hexd
    for bad in ("md5:" + "0" * 32, "sha256:xyz", "sha256:" + "0" * 63):
        with pytest.raises(ConfigError):
            parse_digest(bad)


def test_verify_mismatch_is_exit_5(tmp_path: Path):
    f = tmp_path / "a.ndjson"
    f.write_bytes(b"hello")
    verify(f, "sha256:" + _sha(b"hello"), "dataset")
    with pytest.raises(IntegrityError) as info:
        verify(f, _sha(b"other"), "dataset")
    assert info.value.exit_code == ExitCode.DATASET_FETCH_FAILED == 5


class _Signed(http.server.BaseHTTPRequestHandler):
    """Serves /data.ndjson only with ?sig=<current>; anything else is 403 like an expired URL."""

    body = b'{"type": "dataset"}\n'
    valid = "new"
    hits: list[str] = []

    def do_GET(self):  # noqa: N802
        type(self).hits.append(self.path)
        if self.path.endswith(f"sig={self.valid}"):
            self.send_response(200)
            self.send_header("Content-Length", str(len(self.body)))
            self.end_headers()
            self.wfile.write(self.body)
        else:
            self.send_response(403)
            self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def signed_server():
    _Signed.hits = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Signed)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_expired_url_is_refreshed_and_digest_checked(tmp_path: Path, signed_server):
    calls = []

    def refresh():
        calls.append(1)
        return f"{signed_server}/data.ndjson?sig=new"

    spec = DatasetSpec(uri=f"{signed_server}/data.ndjson?sig=old", sha256=_sha(_Signed.body),
                       refresh=refresh)
    resolved = resolve(spec, tmp_path / "cache")
    assert resolved.path.name == "data.ndjson"
    assert resolved.path.read_bytes() == _Signed.body
    assert calls == [1]


def test_expired_url_without_refresh_fails_with_5(tmp_path: Path, signed_server):
    spec = DatasetSpec(uri=f"{signed_server}/data.ndjson?sig=old")
    with pytest.raises(DatasetFetchError, match="HTTP 403") as info:
        resolve(spec, tmp_path / "cache")
    assert "sig=old" not in info.value.message


def test_digest_mismatch_after_download_drops_the_cache(tmp_path: Path, signed_server):
    spec = DatasetSpec(uri=f"{signed_server}/data.ndjson?sig=new", sha256="0" * 64)
    cache = tmp_path / "cache"
    with pytest.raises(IntegrityError):
        resolve(spec, cache)
    assert not list(cache.glob("*.done"))


def test_cli_exits_5_and_reports_the_stage(run_dirs, tmp_path: Path, capsys):
    work, output = run_dirs
    data = tmp_path / "set.ndjson"
    data.write_text('{"type": "dataset", "task": "detect"}\n', encoding="utf-8")
    code = main(["--engine", "yolo", "--model", "yolo26n", "--data", str(data),
                 "--dataset-sha256", "sha256:" + "0" * 64,
                 "--output", str(output), "--work-dir", str(work)])
    assert code == 5
    lines = [events.parse(line) for line in capsys.readouterr().out.splitlines()]
    lines = [e for e in lines if e]
    assert lines[0] == {**lines[0], "event": "stage", "stage": "downloading"}
    assert lines[-1]["event"] == "exit" and lines[-1]["exit_code"] == 5
    assert lines[-1]["code"] == "IntegrityCheckFailed" and lines[-1]["stage"] == "downloading"


def test_events_survive_a_quiet_log_level(capsys, monkeypatch):
    from distill_run.cli import configure_logging

    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    configure_logging()
    events.reset()
    events.stage("training")
    events.stage("training")
    events.progress(3, 10, 41.6, fitness=0.5)
    parsed = [events.parse(x) for x in capsys.readouterr().out.splitlines()]
    assert [p["event"] for p in parsed] == ["stage", "progress"]
    assert parsed[1]["eta_seconds"] == 42 and parsed[1]["total_epochs"] == 10
    assert parsed[1]["time"].endswith("Z")


def test_train_args_are_accepted_ignored_or_rejected():
    accepted, ignored = check_train_args({
        "epochs": 100, "imgsz": 640, "batch": 0.7, "patience": 20, "freeze": 10, "mixup": 0.1,
        "device": 0, "project": "x", "cfg": "/etc/x.yaml", "dropout": 0.1,
    })
    assert accepted == {"epochs": 100, "imgsz": 640, "batch": 0.7, "patience": 20, "freeze": 10,
                        "mixup": 0.1, "dropout": 0.1}
    assert ignored == ["cfg", "device", "project"]
    with pytest.raises(ConfigError) as info:
        check_train_args({"epochs": 1001, "batch": 512, "box": 7.5, "fraction": 0})
    for field in ("epochs", "batch", "box", "fraction"):
        assert field in info.value.message


def test_train_args_json_merges_with_first_level_flags(run_dirs):
    work, output = run_dirs
    ctx = build_context({"engine": "ultralytics", "output": str(output), "work_dir": str(work),
                         "student": "yolo26n.pt", "dataset": "x.ndjson",
                         "train_args": '{"epochs": 3, "patience": 5}', "epochs": 7,
                         "report_to_model": "acme-vision/warehouse/exp-1"})
    assert ctx.config["train_args"] == {"epochs": 7, "patience": 5}
    assert ctx.config["report_to"]["model"] == "acme-vision/warehouse/exp-1"


def test_train_args_accept_ultralytics_key_value_form(run_dirs):
    work, output = run_dirs
    ctx = build_context({"engine": "ultralytics", "output": str(output), "work_dir": str(work),
                         "student": "yolo26n.pt", "dataset": "x.ndjson",
                         "train_args": "epochs=5 lr0=0.01 cos_lr=True optimizer='AdamW'"})
    assert ctx.config["train_args"] == {"epochs": 5, "lr0": 0.01, "cos_lr": True,
                                        "optimizer": "AdamW"}
    with pytest.raises(UsageError):
        build_context({"engine": "ultralytics", "output": str(output), "work_dir": str(work),
                       "student": "yolo26n.pt", "dataset": "x.ndjson", "train_args": "epochs 5"})


def test_report_target_needs_a_token_and_valid_names(monkeypatch):
    section = {"api_base_url": "https://platform.example/", "model": "acme-vision/proj/exp-1",
               "dataset_slug": "acme-vision/warehouse", "dataset_version": "3"}
    assert ReportTarget.from_config({"model": "x"}) is None
    monkeypatch.delenv("REPORT_TO_TOKEN", raising=False)
    with pytest.raises(ConfigError, match="REPORT_TO_TOKEN"):
        ReportTarget.from_config(section)
    monkeypatch.setenv("REPORT_TO_TOKEN", "secret-token")
    target = ReportTarget.from_config(section)
    assert (target.base_url, target.owner, target.model, target.dataset_version) == (
        "https://platform.example", "acme-vision", "exp-1", 3)
    assert "secret-token" not in repr(target)
    for bad in ("abc/proj/exp", "acme-vision/Proj/exp", "acme-vision/proj"):
        with pytest.raises(ConfigError):
            ReportTarget.from_config({**section, "model": bad})


def _hooks(tmp_path: Path, report=None):
    from distill_run.engines.ultralytics import _Hooks

    ctx = SimpleNamespace(cancellation=SimpleNamespace(requested=False))
    return _Hooks(ctx, report, tmp_path)


def test_progress_skips_the_final_validation_callback(tmp_path: Path, capsys):
    from distill_run.cli import configure_logging

    configure_logging()
    hooks = _hooks(tmp_path)
    trainer = SimpleNamespace(epochs=2, start_epoch=0, train_time_start=time.time(), fitness=0.5)
    for epoch in (0, 1, 2):
        trainer.epoch = epoch
        hooks.fit_epoch_end(trainer)
    progress = [events.parse(x) for x in capsys.readouterr().out.splitlines()]
    assert [p["epoch"] for p in progress if p and p["event"] == "progress"] == [1, 2]


def test_report_problem_names_what_did_not_reach_the_platform(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("REPORT_TO_TOKEN", "t")
    report = ReportTarget.from_config({"api_base_url": "http://p", "model": "acme-vision/p/m"})
    best = tmp_path / "best.pt"
    best.write_bytes(b"w")
    hooks = _hooks(tmp_path, report)
    assert "did not accept the run" in hooks.report_problem(SimpleNamespace(platform=None))
    trainer = SimpleNamespace(platform={"model_id": "m"}, best=str(best), last=str(best))
    assert "uploading best.pt" in hooks.report_problem(trainer)
    hooks.uploaded = {"modelPath": "gs://x"}
    assert "training_complete" in hooks.report_problem(trainer)
    hooks.sent["training_complete"] = {"ok": True}
    assert hooks.report_problem(trainer) is None
    assert _hooks(tmp_path).report_problem(trainer) is None


def test_ultralytics_engine_accepts_its_flags():
    from distill_run import params as P

    accepted = {p.dest for p in P.params_for("ultralytics")}
    assert {"train_args", "epochs", "student", "dataset", "model_sha256", "report_to_url",
            "dataset_slug", "dataset_sha256", "resume", "device"} <= accepted
    assert "max_steps" not in accepted and "logging_steps" not in accepted
    assert json.loads(json.dumps(sorted(accepted)))
