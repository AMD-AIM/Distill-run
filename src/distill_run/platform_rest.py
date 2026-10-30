"""Result reporting through the Platform's public REST API (design doc §8.2 / §8.3).

Used by the ``ultralytics`` engine when ``report_to.protocol`` is ``rest``:

- weights: ``POST signed-url`` -> ``PUT uploadUrl`` with every returned header ->
  ``POST complete``; a 412 on the PUT starts again from the signed URL,
- results: ``PATCH /api/models/{owner}/{project}/{model}``,
- dataset re-signing: ``GET /api/datasets/{owner}/{dataset}/export?v=N``.

Every call retries 429 after ``Retry-After`` (or ``X-RateLimit-Reset``) and
5xx/network errors with exponential backoff. The token only goes into the
``Authorization`` header; messages carry the path and status code, never URLs
with signatures.
"""

from __future__ import annotations

import base64
import email.utils
import hashlib
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

DEFAULT_PATHS = {
    "signed_url": "/api/upload/signed-url",
    "complete": "/api/upload/complete",
    "model": "/api/models/{owner}/{project}/{model}",
    "dataset_export": "/api/datasets/{owner}/{dataset}/export",
}
DEFAULT_STATUSES = {
    "running": "training",
    "completed": "completed",
    "failed": "failed",
    "cancelled": "cancelled",
}
METRIC_NAMES = {
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50-95",
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
}
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class PlatformHTTPError(Exception):
    def __init__(self, method: str, path: str, status: int | None, detail: str = ""):
        self.status = status
        where = f"HTTP {status}" if status else "no response"
        super().__init__(f"{method} {path}: {where}{f' ({detail})' if detail else ''}")


@dataclass
class RestConfig:
    paths: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_PATHS))
    statuses: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_STATUSES))
    metrics_every_epochs: int = 1
    checkpoint_every_epochs: int = 10
    train_results_field: str = "trainResults"
    upload_md5: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 60.0
    max_attempts: int = 5
    max_retry_wait: float = 300.0

    @classmethod
    def from_section(cls, section: dict[str, Any]) -> RestConfig:
        rest = section.get("rest") or {}
        return cls(
            paths={**DEFAULT_PATHS, **(rest.get("paths") or {})},
            statuses={**DEFAULT_STATUSES, **(rest.get("statuses") or {})},
            metrics_every_epochs=int(section.get("metrics_every_epochs") or 1),
            checkpoint_every_epochs=int(section.get("checkpoint_every_epochs") or 0),
            train_results_field=str(rest.get("train_results_field", "trainResults") or ""),
            upload_md5=bool(rest.get("upload_md5", False)),
            metadata=dict(section.get("metadata") or {}),
            timeout_seconds=float(rest.get("timeout_seconds", 60)),
            max_attempts=max(1, int(rest.get("max_attempts", 5))),
            max_retry_wait=float(rest.get("max_retry_wait", 300)),
        )


class RestClient:
    def __init__(self, base_url: str, token: str, config: RestConfig,
                 sleep: Callable[[float], None] = time.sleep):
        self.base_url = base_url.rstrip("/")
        self._token = token
        self.config = config
        self._sleep = sleep

    def __repr__(self) -> str:
        return f"RestClient({self.base_url!r})"

    def request(self, method: str, path: str, body: dict[str, Any] | None = None,
                params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = self.base_url + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Authorization": f"Bearer {self._token}", "Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        raw = self._send(method, path, lambda: urllib.request.Request(
            url, data=data, headers=headers, method=method))
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except ValueError as exc:
            raise PlatformHTTPError(method, path, 200, "response is not JSON") from exc
        return parsed if isinstance(parsed, dict) else {"data": parsed}

    def put_file(self, url: str, path: Path, headers: dict[str, str]) -> None:
        """PUT to a signed URL (no Authorization header); 412 is raised, not retried."""
        size = path.stat().st_size

        def build():
            stream = path.open("rb")
            request = urllib.request.Request(url, data=stream, method="PUT",
                                             headers={**headers, "Content-Length": str(size)})
            request._stream = stream  # closed in _send
            return request

        self._send("PUT", "uploadUrl", build, no_retry={412})

    def _send(self, method: str, label: str, build: Callable[[], urllib.request.Request],
              no_retry: frozenset[int] | set[int] = frozenset()) -> bytes:
        attempts = self.config.max_attempts
        for attempt in range(1, attempts + 1):
            request = build()
            wait: float | None = None
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as resp:
                    return resp.read()
            except urllib.error.HTTPError as exc:
                if exc.code in no_retry or exc.code not in RETRY_STATUSES or attempt == attempts:
                    raise PlatformHTTPError(method, label, exc.code, _error_detail(exc)) from exc
                if exc.code == 429:
                    wait = _retry_after(exc.headers)
                logger.warning("%s %s: HTTP %d, retrying (%d/%d)", method, label, exc.code,
                               attempt, attempts)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                if attempt == attempts:
                    reason = getattr(exc, "reason", exc)
                    raise PlatformHTTPError(method, label, None, str(reason)) from exc
                logger.warning("%s %s: %s, retrying (%d/%d)", method, label,
                               type(exc).__name__, attempt, attempts)
            finally:
                stream = getattr(request, "_stream", None)
                if stream is not None:
                    stream.close()
            if wait is None:
                wait = min(2.0 ** attempt, 60.0)
            self._sleep(min(max(wait, 0.0), self.config.max_retry_wait))
        raise AssertionError("unreachable")


def _retry_after(headers) -> float | None:
    value = headers.get("Retry-After") if headers else None
    if value:
        value = value.strip()
        if value.isdigit():
            return float(value)
        try:
            return email.utils.parsedate_to_datetime(value).timestamp() - time.time()
        except (TypeError, ValueError):
            pass
    reset = headers.get("X-RateLimit-Reset") if headers else None
    if reset and reset.strip().replace(".", "", 1).isdigit():
        number = float(reset)
        # Either an epoch timestamp or a delta in seconds.
        return number - time.time() if number > 1e9 else number
    return None


def _error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read(500).decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return ""
    try:
        parsed = json.loads(body)
    except ValueError:
        return ""
    if isinstance(parsed, dict):
        message = parsed.get("error") or parsed.get("message") or ""
        return str(message)[:200]
    return ""


def resign_dataset_url(client: RestClient, owner: str, dataset: str,
                       version: int | None) -> str:
    path = client.config.paths["dataset_export"].format(owner=owner, dataset=dataset)
    response = client.request("GET", path, params={"v": version} if version is not None else None)
    url = response.get("downloadUrl") or response.get("url")
    if not url:
        raise PlatformHTTPError("GET", path, 200, "no downloadUrl in the response")
    return str(url)


class RestReporter:
    """Weights upload and model PATCH for one run; methods raise PlatformHTTPError."""

    def __init__(self, client: RestClient, *, owner: str, project: str, model: str,
                 model_id: str, dataset_slug: str | None = None):
        self.client = client
        self.config = client.config
        self.model_id = model_id
        self.dataset_slug = dataset_slug
        self.model_path = self.config.paths["model"].format(owner=owner, project=project,
                                                            model=model)

    def upload(self, path: Path, filename: str | None = None) -> dict[str, Any]:
        paths = self.config.paths
        name = filename or path.name
        size = path.stat().st_size
        content_type = "application/octet-stream"
        for attempt in range(1, 4):
            signed = self.client.request("POST", paths["signed_url"], {
                "assetType": "models", "assetId": self.model_id, "filename": name[:256],
                "contentType": content_type, "totalBytes": size,
            })
            if not signed.get("uploadUrl") or not signed.get("sessionId"):
                raise PlatformHTTPError("POST", paths["signed_url"], 200,
                                        "response lacks uploadUrl or sessionId")
            headers = {str(k): str(v) for k, v in (signed.get("headers") or {}).items()}
            headers.setdefault("Content-Type", content_type)
            try:
                self.client.put_file(signed["uploadUrl"], path, headers)
            except PlatformHTTPError as exc:
                if exc.status == 412 and attempt < 3:
                    logger.warning("upload of %s got 412; requesting a new signed URL", name)
                    continue
                raise
            body: dict[str, Any] = {"sessionId": signed["sessionId"]}
            if self.config.upload_md5:
                body["md5"] = _md5_base64(path)
            self.client.request("POST", paths["complete"], body)
            logger.info("uploaded %s (%d bytes)", name, size)
            return {"filename": name, "bytes": size, "sessionId": signed["sessionId"]}
        raise AssertionError("unreachable")

    def patch(self, status_key: str, *, metrics: dict[str, Any] | None = None,
              epochs: int | None = None, best_epoch: int | None = None,
              best_fitness: float | None = None, error: str | None = None,
              train_results: dict[str, Any] | None = None,
              metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"status": self.config.statuses[status_key]}
        if metrics:
            body["metrics"] = platform_metrics(metrics)
        if epochs is not None:
            body["epochs"] = epochs
        if best_epoch is not None:
            body["bestEpoch"] = best_epoch
        if best_fitness is not None:
            body["bestFitness"] = best_fitness
        if self.dataset_slug:
            body["datasetSlug"] = self.dataset_slug
        if error:
            body["trainingError"] = error[:2000]
        if train_results is not None and self.config.train_results_field:
            body[self.config.train_results_field] = train_results
        body["metadata"] = {**self.config.metadata, **(metadata or {})}
        return self.client.request("PATCH", self.model_path, body)


def platform_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """``metrics/mAP50(B)`` -> ``mAP50`` and friends; other keys are dropped."""
    out = {}
    for key, value in metrics.items():
        name = METRIC_NAMES.get(key, key if key in METRIC_NAMES.values() else None)
        if name and isinstance(value, (int, float)):
            out[name] = float(value)
    return out


def _md5_base64(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode()
