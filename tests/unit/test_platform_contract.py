"""Service-facing contract: integrity checks, URL refresh, events, train_args, report_to."""

from __future__ import annotations

import hashlib
import http.server
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from distill_run import events
from distill_run.cli import build_context, main
from distill_run.dataset import DatasetSpec, resolve
from distill_run.engines.ultralytics import ReportTarget, check_train_args
from distill_run.errors import (
    ConfigError,
    DataError,
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


class _Export(http.server.BaseHTTPRequestHandler):
    """/data.ndjson only with ?sig=new (else 403); /api/datasets/... hands out that URL."""

    body = b'{"type": "dataset"}\n'
    calls: list[tuple[str, str | None]] = []
    export_status = 200

    def do_GET(self):  # noqa: N802
        type(self).calls.append((self.path, self.headers.get("Authorization")))
        if self.path.startswith("/api/datasets/"):
            url = f"http://127.0.0.1:{self.server.server_port}/data.ndjson?sig=new"
            status, data = type(self).export_status, json.dumps({"downloadUrl": url}).encode()
        elif self.path == "/data.ndjson?sig=new":
            status, data = 200, self.body
        else:
            status, data = 403, b""
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def export_server():
    _Export.calls, _Export.export_status = [], 200
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Export)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def _ultralytics_specs(run_dirs, monkeypatch, base: str, **extra):
    from distill_run.engines import ultralytics as ul

    monkeypatch.setattr(ul, "_setup_ultralytics", lambda ctx, report: None)
    monkeypatch.setattr(ul, "_check_types", lambda args: None)
    monkeypatch.setenv("REPORT_TO_TOKEN", "tok-r156i")
    work, output = run_dirs
    ctx = build_context({"engine": "ultralytics", "output": str(output), "work_dir": str(work),
                         "student": "yolo26n.yaml", "dataset": f"{base}/data.ndjson?sig=old",
                         "dataset_sha256": "sha256:" + _sha(_Export.body),
                         "report_to_url": base, "report_to_model": "acme-vision/p/m", **extra})
    return ul.UltralyticsEngine().dataset_specs(ctx)


def test_expired_dataset_url_is_resigned_with_the_report_token(run_dirs, tmp_path: Path,
                                                               monkeypatch, export_server, caplog):
    specs = _ultralytics_specs(run_dirs, monkeypatch, export_server,
                               dataset_slug="acme-vision/warehouse", dataset_version=3)
    assert resolve(specs["dataset"], tmp_path / "cache").path.read_bytes() == _Export.body
    assert _Export.calls == [
        ("/data.ndjson?sig=old", None),
        ("/api/datasets/acme-vision/warehouse/export?v=3", "Bearer tok-r156i"),
        ("/data.ndjson?sig=new", None),
    ]
    assert "tok-r156i" not in caplog.text and "sig=old" not in caplog.text


def test_resign_rejected_exits_5_without_the_token(run_dirs, tmp_path: Path, monkeypatch,
                                                   export_server):
    _Export.export_status = 401
    specs = _ultralytics_specs(run_dirs, monkeypatch, export_server,
                               dataset_slug="acme-vision/warehouse")
    with pytest.raises(DatasetFetchError, match="re-signing dataset acme-vision/warehouse") as info:
        resolve(specs["dataset"], tmp_path / "cache")
    assert info.value.exit_code == 5 and "tok-r156i" not in info.value.message


def test_without_a_dataset_slug_an_expired_url_is_not_resigned(run_dirs, tmp_path: Path,
                                                               monkeypatch, export_server):
    specs = _ultralytics_specs(run_dirs, monkeypatch, export_server)
    with pytest.raises(DatasetFetchError, match="HTTP 403"):
        resolve(specs["dataset"], tmp_path / "cache")
    assert not any(path.startswith("/api/") for path, _ in _Export.calls)


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
    trainer = SimpleNamespace(epochs=10, start_epoch=0, train_time_start=time.time(), fitness=0.5)

    def final_eval():  # as in Ultralytics: one more on_fit_epoch_end at epoch + 1
        trainer.epoch += 1
        hooks.fit_epoch_end(trainer)
        trainer.epoch -= 1

    trainer.final_eval = final_eval
    hooks.pretrain_start(trainer)
    for epoch in (0, 1):  # stopped early, after 2 of 10 epochs
        trainer.epoch = epoch
        hooks.fit_epoch_end(trainer)
    trainer.final_eval()
    progress = [events.parse(x) for x in capsys.readouterr().out.splitlines()]
    assert [p["epoch"] for p in progress if p and p["event"] == "progress"] == [1, 2]


def _fake_platform(monkeypatch):
    """Stand-in for ultralytics.utils.callbacks.platform that records what it would send."""
    import sys
    import types

    calls: list[tuple] = []
    platform = types.ModuleType("ultralytics.utils.callbacks.platform")
    platform.on_pretrain_routine_start = lambda trainer: None
    platform.on_fit_epoch_end = lambda trainer: calls.append(("epoch_end", trainer.epoch + 1))
    platform.on_model_save = lambda trainer: calls.append(
        ("upload", trainer.epoch + 1, trainer.platform["last_upload"]))
    names = ("ultralytics", "ultralytics.utils", "ultralytics.utils.callbacks")
    modules = {name: types.ModuleType(name) for name in names}
    modules["ultralytics"].utils = modules["ultralytics.utils"]
    modules["ultralytics.utils"].callbacks = modules["ultralytics.utils.callbacks"]
    modules["ultralytics.utils.callbacks"].platform = platform
    modules["ultralytics.utils.callbacks.platform"] = platform
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return platform, calls


def _train_webhook(tmp_path: Path, monkeypatch, epochs: int, chain=True, **cadence):
    """Drive _Hooks like Ultralytics: on_model_save, on_fit_epoch_end per epoch, then final_eval."""
    monkeypatch.setenv("REPORT_TO_TOKEN", "t")
    report = ReportTarget.from_config({"api_base_url": "http://p", "model": "acme-vision/p/m",
                                       **cadence})
    platform, calls = _fake_platform(monkeypatch)
    hooks = _hooks(tmp_path, report)
    trainer = SimpleNamespace(
        epochs=epochs, start_epoch=0, epoch=0, train_time_start=time.time(), fitness=0.5,
        save_dir=str(tmp_path), platform={"model_id": "m", "last_upload": 123.0},
        callbacks={"on_pretrain_routine_start": [platform.on_pretrain_routine_start],
                   "on_fit_epoch_end": [platform.on_fit_epoch_end],
                   "on_model_save": [platform.on_model_save]} if chain else {})

    def run(event):
        for fn in trainer.callbacks.get(event, []):
            fn(trainer)

    def final_eval():  # as in Ultralytics: one more on_fit_epoch_end at epoch + 1
        trainer.epoch += 1
        hooks.fit_epoch_end(trainer)
        run("on_fit_epoch_end")
        trainer.epoch -= 1

    trainer.final_eval = final_eval
    hooks.pretrain_start(trainer)
    for epoch in range(epochs):
        trainer.epoch = epoch
        hooks.model_save(trainer)
        run("on_model_save")
        hooks.fit_epoch_end(trainer)
        run("on_fit_epoch_end")
    trainer.final_eval()
    return calls


def test_webhook_reports_every_n_epochs(tmp_path: Path, monkeypatch):
    calls = _train_webhook(tmp_path, monkeypatch, 5, metrics_every_epochs=2,
                           checkpoint_every_epochs=2)
    assert [c[1] for c in calls if c[0] == "epoch_end"] == [2, 4, 6]  # 6 is final_eval
    assert [c for c in calls if c[0] == "upload"] == [("upload", 2, 0.0), ("upload", 4, 0.0)]


def test_webhook_cadence_defaults_and_zero(tmp_path: Path, monkeypatch):
    calls = _train_webhook(tmp_path, monkeypatch, 3)
    assert [c[1] for c in calls if c[0] == "epoch_end"] == [1, 2, 3, 4]
    assert [c for c in calls if c[0] == "upload"] == []
    assert [c[1] for c in _train_webhook(tmp_path, monkeypatch, 12) if c[0] == "upload"] == [10]
    assert [c for c in _train_webhook(tmp_path, monkeypatch, 12, checkpoint_every_epochs=0)
            if c[0] == "upload"] == []


def test_webhook_without_the_platform_callback_still_trains(tmp_path: Path, monkeypatch):
    assert _train_webhook(tmp_path, monkeypatch, 2, chain=False, metrics_every_epochs=2) == []


def test_rest_reports_every_n_epochs(tmp_path: Path):
    from distill_run.platform_rest import RestConfig

    sent = []
    hooks = _hooks(tmp_path)
    hooks.ctx.run_id = "r"
    hooks.rest = SimpleNamespace(
        config=RestConfig(metrics_every_epochs=2, checkpoint_every_epochs=2),
        patch=lambda status, **fields: sent.append(("patch", fields["epochs"])),
        upload=lambda path, name: sent.append(("upload", name)))
    last = tmp_path / "last.pt"
    last.write_bytes(b"w")
    trainer = SimpleNamespace(epochs=5, start_epoch=0, train_time_start=time.time(), fitness=0.5,
                              metrics={}, best_fitness=None, last=str(last),
                              final_eval=lambda: None)
    hooks.pretrain_start(trainer)
    for epoch in range(5):
        trainer.epoch = epoch
        hooks.model_save(trainer)
        hooks.fit_epoch_end(trainer)
    hooks._queue.shutdown(wait=True)
    assert [s for s in sent if s[0] == "patch"] == [("patch", 2), ("patch", 4)]
    assert [s for s in sent if s[0] == "upload"] == [("upload", "last.pt"), ("upload", "last.pt")]


def _rest_hooks(tmp_path: Path, every: int):
    from distill_run.platform_rest import RestConfig

    sent: list[tuple[str, dict]] = []
    hooks = _hooks(tmp_path)
    hooks.ctx.run_id = "r"
    hooks.rest = SimpleNamespace(
        config=RestConfig(metrics_every_epochs=every, checkpoint_every_epochs=0),
        patch=lambda status, **fields: sent.append((status, fields)),
        upload=lambda path, name=None: {"filename": name or path.name})
    return hooks, sent


def test_rest_metrics_go_into_train_results_and_the_final_patch_sends_the_array(tmp_path: Path):
    hooks, sent = _rest_hooks(tmp_path, 2)
    best = tmp_path / "best.pt"
    best.write_bytes(b"w")
    trainer = SimpleNamespace(epochs=5, start_epoch=0, train_time_start=time.time(), fitness=0.5,
                              metrics={"metrics/mAP50(B)": 0.8, "val/box_loss": 1.0},
                              best_fitness=0.5, best=str(best), last=str(best),
                              final_eval=lambda: None)
    hooks.pretrain_start(trainer)
    for epoch in range(5):
        trainer.epoch = epoch
        hooks.fit_epoch_end(trainer)
    hooks._queue.shutdown(wait=True)
    assert all("metrics" not in fields for _, fields in sent)
    assert [[r["epoch"] for r in fields["train_results"]] for _, fields in sent] == [[2], [2, 4]]
    assert sent[0][1]["train_results"][0]["metrics"] == {"mAP50": 0.8}
    hooks._queue = ThreadPoolExecutor(max_workers=1)
    assert hooks._rest_finish(trainer) is None
    status, fields = sent[-1]
    assert status == "completed" and isinstance(fields["train_results"], list)
    assert [r["epoch"] for r in fields["train_results"]] == [2, 4, 5]


def test_rest_failure_sends_code_and_message(tmp_path: Path):
    hooks, sent = _rest_hooks(tmp_path, 1)
    hooks.failed(None, DataError("labels are broken"))
    hooks._queue.shutdown(wait=True)
    status, fields = sent[-1]
    assert status == "failed"
    assert (fields["error_code"], fields["error"]) == ("DataInvalid", "labels are broken")


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


def test_report_cadence_flags_reach_report_to(run_dirs):
    work, output = run_dirs
    base = {"engine": "ultralytics", "output": str(output), "work_dir": str(work),
            "student": "yolo26n.pt", "dataset": "x.ndjson"}
    report_to = build_context(base).config["report_to"]
    assert (report_to["metrics_every_epochs"], report_to["checkpoint_every_epochs"]) == (1, 10)
    report_to = build_context({**base, "report_to_metrics_every_epochs": 5,
                               "report_to_checkpoint_every_epochs": 0}).config["report_to"]
    assert (report_to["metrics_every_epochs"], report_to["checkpoint_every_epochs"]) == (5, 0)


def test_ultralytics_engine_accepts_its_flags():
    from distill_run import params as P

    accepted = {p.dest for p in P.params_for("ultralytics")}
    assert {"train_args", "epochs", "student", "dataset", "model_sha256", "report_to_url",
            "dataset_slug", "dataset_sha256", "resume", "device",
            "report_to_metrics_every_epochs", "report_to_checkpoint_every_epochs"} <= accepted
    assert "max_steps" not in accepted and "logging_steps" not in accepted
    assert json.loads(json.dumps(sorted(accepted)))


def _cancel_hooks(tmp_path: Path, monkeypatch, protocol="webhook"):
    """Webhook hooks with a real Cancellation; REST calls are recorded, not sent."""
    from distill_run.engines import ultralytics as ul
    from distill_run.signals import Cancellation

    monkeypatch.setenv("REPORT_TO_TOKEN", "tok-r156i")
    report = ReportTarget.from_config({"api_base_url": "http://p", "model": "acme-vision/p/m",
                                       "protocol": protocol, "model_id": "m1"})
    sent: list[tuple] = []

    def request(client, method, path, body=None, params=None):
        sent.append((method, path, body, client.config))
        return {}

    monkeypatch.setattr(ul.RestClient, "request", request)
    ctx = SimpleNamespace(cancellation=Cancellation(), run_id="r")
    return ul._Hooks(ctx, report, tmp_path), sent


def test_sigterm_marks_the_webhook_model_cancelled_at_once(tmp_path: Path, monkeypatch):
    hooks, sent = _cancel_hooks(tmp_path, monkeypatch)
    hooks.report = replace(hooks.report, heartbeat_seconds=0)
    trainer = SimpleNamespace(stop=False, platform={
        "model_id": "m1", "model_url": "https://platform.example/acme-vision/p/m-2"})
    hooks.train_start(trainer)
    hooks.ctx.cancellation.request(15)
    deadline = time.time() + 5
    while not sent and time.time() < deadline:
        time.sleep(0.05)
    hooks.batch_end(trainer)
    hooks.train_end(trainer)  # runs before the callback's final upload
    assert trainer.stop
    assert len(sent) == 1
    method, path, body, config = sent[0]
    assert (method, path, body["status"]) == ("PATCH", "/api/models/acme-vision/p/m-2",
                                              "cancelled")
    assert set(body) == {"status", "metadata"}
    assert config.timeout_seconds * config.max_attempts + config.max_retry_wait <= 30


def test_webhook_cancel_patch_falls_back_to_the_report_model(tmp_path: Path, monkeypatch):
    hooks, sent = _cancel_hooks(tmp_path, monkeypatch)
    hooks.train_end(SimpleNamespace(platform={"model_id": "m1"}))
    assert sent == []  # no signal, no PATCH
    hooks.ctx.cancellation.request(15)
    hooks.train_end(SimpleNamespace(platform=None))
    assert sent == []  # the platform never took the run
    hooks._cancel_sent = False
    hooks.train_end(SimpleNamespace(platform={"model_id": "m1"}))
    assert [(m, p) for m, p, _, _ in sent] == [("PATCH", "/api/models/acme-vision/p/m")]


def test_cancel_patch_failure_does_not_stop_the_upload(tmp_path: Path, monkeypatch, caplog):
    import logging

    from distill_run.engines import ultralytics as ul
    from distill_run.platform_rest import PlatformHTTPError

    # configure_logging() in an earlier test may have stopped propagation to caplog's handler.
    monkeypatch.setattr(logging.getLogger("distill_run"), "propagate", True)
    caplog.set_level(logging.WARNING, logger="distill_run")
    hooks, _ = _cancel_hooks(tmp_path, monkeypatch)

    def refuse(client, method, path, body=None, params=None):
        raise PlatformHTTPError(method, path, 409)

    monkeypatch.setattr(ul.RestClient, "request", refuse)
    hooks.ctx.cancellation.request(15)
    hooks.train_end(SimpleNamespace(platform={"model_id": "m1"}))
    assert "reporting the cancellation failed" in caplog.text
    assert "tok-r156i" not in caplog.text


def test_platform_cancel_is_exit_130_with_its_own_code(tmp_path: Path, monkeypatch):
    from distill_run.errors import CancelledError, PlatformCancelledError

    hooks, sent = _cancel_hooks(tmp_path, monkeypatch)
    summary = {"epochs_completed": 2}
    trainer = SimpleNamespace(platform={"model_id": "m1", "cancelled": False})
    assert hooks.cancelled_error(trainer, summary) is None
    trainer.platform["cancelled"] = True  # epoch_end answered cancelled: true
    error = hooks.cancelled_error(trainer, summary)
    assert isinstance(error, PlatformCancelledError)
    assert (error.exit_code, error.code) == (130, "PlatformCancelled")
    assert "epoch 2" in error.message
    hooks.ctx.cancellation.request(15)  # our own SIGTERM wins
    assert type(hooks.cancelled_error(trainer, summary)) is CancelledError
    assert sent == []


def test_platform_cancel_exit_event(run_dirs, monkeypatch, capsys):
    from distill_run.engines import ultralytics as ul
    from distill_run.errors import PlatformCancelledError

    def cancelled(self, ctx):
        raise PlatformCancelledError("cancelled from the platform at epoch 2")

    monkeypatch.setattr(ul, "_setup_ultralytics", lambda ctx, report: None)
    monkeypatch.setattr(ul, "_check_types", lambda args: None)
    monkeypatch.setattr(ul.UltralyticsEngine, "preflight", lambda self, ctx: None)
    monkeypatch.setattr(ul.UltralyticsEngine, "run", cancelled)
    monkeypatch.setattr(ul.UltralyticsEngine, "dataset_specs", lambda self, ctx: {})
    work, output = run_dirs
    code = main(["--engine", "ultralytics", "--model", "yolo26n.yaml", "--data", "x.ndjson",
                 "--output", str(output), "--work-dir", str(work)])
    assert code == 130
    lines = [e for e in (events.parse(x) for x in capsys.readouterr().out.splitlines()) if e]
    assert lines[-1]["event"] == "exit"
    assert (lines[-1]["exit_code"], lines[-1]["code"]) == (130, "PlatformCancelled")


def test_bundled_fonts_land_in_the_ultralytics_config_dir(tmp_path: Path):
    from distill_run.engines.ultralytics import _install_fonts

    fonts, config = tmp_path / "fonts", tmp_path / "config"
    fonts.mkdir()
    (fonts / "Arial.ttf").write_bytes(b"font")
    _install_fonts(str(fonts), config)
    assert (config / "Arial.ttf").read_bytes() == b"font"
    (config / "Arial.ttf").write_bytes(b"newer")
    _install_fonts(str(fonts), config)
    assert (config / "Arial.ttf").read_bytes() == b"newer"
    _install_fonts(None, tmp_path / "none")
    _install_fonts(str(tmp_path / "missing"), tmp_path / "none")
    assert not (tmp_path / "none").exists()
