"""REST reporting (design doc §8.2 / §8.3) against a local fake Platform."""

from __future__ import annotations

import copy
import hashlib
import http.server
import io
import json
import logging
import re
import threading
from pathlib import Path

import pytest

from distill_run import defaults, redact
from distill_run.engines.ultralytics import ReportTarget
from distill_run.errors import ConfigError
from distill_run.platform_rest import (
    PlatformHTTPError,
    RestClient,
    RestConfig,
    RestReporter,
    platform_metrics,
    resign_dataset_url,
    train_result,
)

TOKEN = "tok-SECRET-1"
UTC_Z = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?Z$")


class _Platform(http.server.BaseHTTPRequestHandler):
    calls: list[dict] = []
    # path -> list of (status, headers, body) served before the default answer
    script: dict[str, list[tuple[int, dict, dict]]] = {}
    port = 0

    def _handle(self, method):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        path = self.path.split("?")[0]
        entry = {"method": method, "path": self.path, "auth": self.headers.get("Authorization"),
                 "headers": {k.lower(): v for k, v in self.headers.items()}}
        if method == "PUT":
            entry["bytes"] = len(raw)
        elif raw:
            entry["body"] = json.loads(raw)
        type(self).calls.append(entry)
        queued = type(self).script.get(path)
        if queued:
            status, headers, body = queued.pop(0)
        else:
            status, headers, body = 200, {}, self._default(method, path)
        data = json.dumps(body).encode()
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _default(self, method, path):
        if path == "/api/upload/signed-url":
            n = sum(c["path"] == path for c in type(self).calls) - 1
            return {"sessionId": f"s{n}", "uploadUrl": f"http://127.0.0.1:{self.port}/gcs/{n}?sig=x",
                    "expiresAt": "2026-10-31T00:00:00Z",
                    "headers": {"x-goog-if-generation-match": "0"}}
        if path.startswith("/api/datasets/"):
            return {"downloadUrl": "https://storage.example/data.ndjson?sig=new"}
        return {"ok": True}

    def do_POST(self):  # noqa: N802
        self._handle("POST")

    def do_PUT(self):  # noqa: N802
        self._handle("PUT")

    def do_PATCH(self):  # noqa: N802
        self._handle("PATCH")

    def do_GET(self):  # noqa: N802
        self._handle("GET")

    def log_message(self, *args):
        pass


@pytest.fixture
def platform():
    _Platform.calls, _Platform.script = [], {}
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Platform)
    _Platform.port = server.server_port
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", _Platform
    server.shutdown()


def _reporter(base_url, sleeps=None, **config):
    client = RestClient(base_url, TOKEN, RestConfig(**config),
                        sleep=(sleeps.append if sleeps is not None else lambda s: None))
    return RestReporter(client, owner="acme-vision", project="warehouse", model="exp-1",
                        model_id="65f1", dataset_slug="acme-vision/warehouse")


def test_upload_follows_signed_url_put_complete(platform, tmp_path: Path):
    base, fake = platform
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"x" * 1000)
    result = _reporter(base, upload_md5=True).upload(weights)
    assert result["bytes"] == 1000
    signed, put, complete = fake.calls
    assert signed["body"] == {"assetType": "models", "assetId": "65f1", "filename": "best.pt",
                              "contentType": "application/octet-stream", "totalBytes": 1000}
    assert signed["auth"] == f"Bearer {TOKEN}"
    assert put["method"] == "PUT" and put["bytes"] == 1000 and put["auth"] is None
    assert put["headers"]["x-goog-if-generation-match"] == "0"
    assert put["headers"]["content-type"] == "application/octet-stream"
    assert complete["path"] == "/api/upload/complete"
    assert complete["body"]["sessionId"] == "s0" and complete["body"]["md5"]


def test_upload_412_starts_again_from_the_signed_url(platform, tmp_path: Path):
    base, fake = platform
    fake.script["/gcs/0"] = [(412, {}, {})]
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"w")
    _reporter(base).upload(weights)
    assert [c["path"].split("?")[0] for c in fake.calls] == [
        "/api/upload/signed-url", "/gcs/0", "/api/upload/signed-url", "/gcs/1",
        "/api/upload/complete"]
    assert fake.calls[-1]["body"]["sessionId"] == "s1"


def test_429_waits_for_retry_after(platform):
    base, fake = platform
    path = "/api/models/acme-vision/warehouse/exp-1"
    fake.script[path] = [(429, {"Retry-After": "7"}, {}), (503, {}, {})]
    sleeps: list[float] = []
    _reporter(base, sleeps).patch("running", epochs=3)
    assert sleeps[0] == 7 and len(sleeps) == 2
    assert len([c for c in fake.calls if c["path"] == path]) == 3


def test_client_errors_are_not_retried_and_hide_the_token(platform):
    base, fake = platform
    fake.script["/api/models/acme-vision/warehouse/exp-1"] = [(409, {}, {"error": "renamed"})]
    with pytest.raises(PlatformHTTPError) as info:
        _reporter(base).patch("running")
    assert info.value.status == 409 and "renamed" in str(info.value)
    assert TOKEN not in str(info.value) and TOKEN not in repr(_reporter(base).client)
    assert len(fake.calls) == 1


def test_patch_body_follows_section_8_3(platform):
    base, fake = platform
    results = [train_result(100, 0.71, {"metrics/mAP50(B)": 0.88, "metrics/mAP50-95(B)": 0.64,
                                        "val/box_loss": 1.2})]
    _reporter(base, metadata={"session_id": "sess_7f3a"}).patch(
        "completed", epochs=100, best_epoch=83, best_fitness=0.71, train_results=results,
        metadata={"run_id": "r1"})
    call = fake.calls[0]
    assert call["method"] == "PATCH" and call["path"] == "/api/models/acme-vision/warehouse/exp-1"
    assert call["body"] == {
        "status": "completed", "epochs": 100,
        "bestEpoch": 83, "bestFitness": 0.71, "datasetSlug": "acme-vision/warehouse",
        "trainResults": results, "metadata": {"session_id": "sess_7f3a", "run_id": "r1"},
    }
    assert results[0]["metrics"] == {"mAP50": 0.88, "mAP50-95": 0.64}
    assert "name" not in call["body"] and "projectId" not in call["body"]
    assert "metrics" not in call["body"]


def test_running_status_is_running_in_the_defaults(platform, monkeypatch):
    base, fake = platform
    _reporter(base).patch("running")
    assert fake.calls[0]["body"]["status"] == "running"
    monkeypatch.setenv("REPORT_TO_TOKEN", "t")
    section = {**copy.deepcopy(defaults.ULTRALYTICS["report_to"]), "api_base_url": "http://p",
               "model": "acme-vision/p/m", "protocol": "rest", "model_id": "65f1"}
    assert ReportTarget.from_config(section).rest.statuses["running"] == "running"


def test_training_error_is_an_object_within_the_limits(platform):
    base, fake = platform
    _reporter(base).patch("failed", error="m" * 2000, error_code="C" * 60)
    error = fake.calls[0]["body"]["trainingError"]
    assert set(error) == {"message", "code", "timestamp"}
    assert error["message"] == "m" * 1000 and error["code"] == "C" * 50
    assert UTC_Z.match(error["timestamp"])
    _reporter(base).patch("failed", error="no code")
    assert set(fake.calls[1]["body"]["trainingError"]) == {"message", "timestamp"}


def test_train_result_item_shape():
    item = train_result(3, 0.5, {"metrics/mAP50(B)": 0.8, "val/box_loss": 1.0})
    assert {k: v for k, v in item.items() if k != "timestamp"} == {
        "epoch": 3, "fitness": 0.5, "metrics": {"mAP50": 0.8}, "system": {}}
    assert UTC_Z.match(item["timestamp"])
    assert "fitness" not in train_result(1, None, {}) and "fitness" not in train_result(
        1, float("nan"), {})


def test_upload_md5_is_lowercase_hex(platform, tmp_path: Path):
    base, fake = platform
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"x" * 1000)
    _reporter(base, upload_md5=True).upload(weights)
    md5 = fake.calls[-1]["body"]["md5"]
    assert re.fullmatch(r"[0-9a-f]{32}", md5) and md5 == hashlib.md5(b"x" * 1000).hexdigest()


def test_dataset_slug_longer_than_100_characters_is_a_config_error(monkeypatch):
    monkeypatch.setenv("REPORT_TO_TOKEN", "t")
    section = {"api_base_url": "http://p", "model": "acme-vision/p/m"}
    ok = "acme-vision/" + "d" * 88
    assert ReportTarget.from_config({**section, "dataset_slug": ok}).dataset_slug == ok
    with pytest.raises(ConfigError, match="100"):
        ReportTarget.from_config({**section, "dataset_slug": ok + "d"})


def test_paths_and_statuses_are_configurable(platform):
    base, fake = platform
    reporter = _reporter(base, paths={"model": "/v2/m/{owner}/{project}/{model}"},
                         statuses={"running": "RUNNING"})
    reporter.patch("running")
    assert fake.calls[0]["path"] == "/v2/m/acme-vision/warehouse/exp-1"
    assert fake.calls[0]["body"]["status"] == "RUNNING"


def test_resign_dataset_url(platform):
    base, fake = platform
    client = RestClient(base, TOKEN, RestConfig())
    url = resign_dataset_url(client, "acme-vision", "warehouse", 3)
    assert url.endswith("sig=new")
    assert fake.calls[0]["path"] == "/api/datasets/acme-vision/warehouse/export?v=3"


def test_platform_metrics_keeps_the_four_names():
    assert platform_metrics({"metrics/precision(B)": 0.5, "metrics/recall(B)": 0.4,
                             "fitness": 0.3, "mAP50": 0.9}) == {
        "precision": 0.5, "recall": 0.4, "mAP50": 0.9}


def test_rest_protocol_needs_a_model_id(monkeypatch):
    monkeypatch.setenv("REPORT_TO_TOKEN", "t")
    section = {"api_base_url": "http://p", "model": "acme-vision/p/m", "protocol": "rest"}
    with pytest.raises(ConfigError, match="model-id"):
        ReportTarget.from_config(section)
    target = ReportTarget.from_config({**section, "model_id": "65f1",
                                       "metrics_every_epochs": 2,
                                       "checkpoint_every_epochs": 5})
    assert target.protocol == "rest"
    assert (target.rest.metrics_every_epochs, target.rest.checkpoint_every_epochs) == (2, 5)
    with pytest.raises(ConfigError, match="protocol"):
        ReportTarget.from_config({**section, "protocol": "ftp"})


def test_token_from_a_file_with_a_trailing_newline_is_stripped(monkeypatch):
    monkeypatch.setenv("REPORT_TO_TOKEN", "ul_test_token_123\n")
    target = ReportTarget.from_config({"api_base_url": "http://p", "model": "acme-vision/p/m"})
    assert target.token == "ul_test_token_123"


@pytest.mark.parametrize("token", ["ul_a\nul_b", "ul a", "ul_\x1ba"])
def test_token_with_inner_whitespace_or_control_characters_is_rejected(monkeypatch, token):
    monkeypatch.setenv("REPORT_TO_TOKEN", token)
    with pytest.raises(ConfigError, match="REPORT_TO_TOKEN") as raised:
        ReportTarget.from_config({"api_base_url": "http://p", "model": "acme-vision/p/m"})
    assert token not in str(raised.value)


def test_log_filter_masks_the_token_in_messages_and_tracebacks(monkeypatch):
    monkeypatch.setenv("REPORT_TO_TOKEN", "ul_secret_tok_9")
    ReportTarget.from_config({"api_base_url": "http://p", "model": "acme-vision/p/m"})
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger("ultralytics-test-leak")
    logger.handlers = [handler]
    logger.propagate = False
    redact.protect(logger)
    logger.warning("Failed to send training_started: Illegal header value %r",
                   b"Bearer ul_secret_tok_9")
    try:
        raise ValueError("Invalid header value b'Bearer ul_secret_tok_9'")
    except ValueError:
        logger.exception("unexpected failure")
    text = stream.getvalue()
    assert "ul_secret_tok_9" not in text
    assert "Illegal header value b'Bearer ***'" in text
    assert "ValueError: Invalid header value b'Bearer ***'" in text


def test_report_cadence_defaults_zero_and_bad_values(monkeypatch):
    monkeypatch.setenv("REPORT_TO_TOKEN", "t")
    section = {"api_base_url": "http://p", "model": "acme-vision/p/m"}
    target = ReportTarget.from_config(section)
    assert (target.rest.metrics_every_epochs, target.rest.checkpoint_every_epochs) == (1, 10)
    off = ReportTarget.from_config({**section, "checkpoint_every_epochs": 0})
    assert off.rest.checkpoint_every_epochs == 0
    for bad in ({"metrics_every_epochs": 0}, {"checkpoint_every_epochs": -1},
                {"metrics_every_epochs": "abc"}):
        with pytest.raises(ConfigError):
            ReportTarget.from_config({**section, **bad})
