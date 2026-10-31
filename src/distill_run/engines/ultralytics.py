"""YOLO training through the ``ultralytics`` package, reporting to Ultralytics Platform.

``train_args`` are Ultralytics ``train()`` arguments, filtered by three lists
(:data:`ACCEPTED`, :data:`IGNORED`, everything else rejected) before any download.
The engine owns ``data``, ``model``, ``save_dir``, ``device``, ``project`` and
``name``.

``report_to.protocol: rest`` reports through the public REST API instead
(:mod:`distill_run.platform_rest`). The default, ``webhook``, uses Ultralytics'
own Platform callback
(``ultralytics/utils/callbacks/platform.py``): it is switched on by
``ULTRALYTICS_API_KEY`` plus ``project=owner/project`` and ``name=model``, and
sends ``training_started``, ``epoch_end`` (each epoch), console output, a
checkpoint upload at most every 15 minutes, the final ``best.pt`` and
``training_complete``. This engine adds what the callback lacks: binding to an
existing ``model_id``, a ``training_progress`` heartbeat, ``training_failed``, and
exit code 10 when the final upload or ``training_complete`` did not go through.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import re
import shutil
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any

from .. import events, redact
from ..context import RunContext
from ..errors import (
    CancelledError,
    ConfigError,
    DataError,
    DatasetFetchError,
    DistillRunError,
    PreflightError,
    ReportError,
    classify,
)
from ..fetchers import redact_url
from ..platform_rest import (
    PlatformHTTPError,
    RestClient,
    RestConfig,
    RestReporter,
)
from ..platform_rest import resign_dataset_url as _resign
from ..utils import atomic_write_json, ensure_dir
from .base import Engine, RunResult, config_dataset, make_spec, specs_from_params
from .yolo import download_verified, fetch_weights, weights_base_url

logger = logging.getLogger(__name__)

AUGMENT = frozenset({
    "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective",
    "flipud", "fliplr", "bgr", "mosaic", "mixup", "cutmix", "copy_paste", "copy_paste_mode",
    "auto_augment", "erasing",
})
ACCEPTED = frozenset({
    "epochs", "imgsz", "batch", "lr0", "lrf", "optimizer", "momentum", "weight_decay",
    "warmup_epochs", "cos_lr", "patience", "seed", "workers", "freeze", "dropout", "rect",
    "multi_scale", "single_cls", "fraction", "close_mosaic", "val", "save_period", "amp",
    *AUGMENT,
})
IGNORED = frozenset({"device", "project", "name", "save_dir", "data", "model", "resume"})

PROTOCOLS = ("webhook", "rest")

_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_LOCAL_PATH = re.compile(r"^(?:/|\./|\.\./|~|file:|[A-Za-z]:[\\/])")


def check_train_args(raw: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Split ``train_args`` into (accepted, ignored names); raise ConfigError on the rest."""
    accepted: dict[str, Any] = {}
    ignored: list[str] = []
    errors: dict[str, str] = {}
    for key, value in raw.items():
        if key in IGNORED or (isinstance(value, str) and _LOCAL_PATH.match(value)):
            ignored.append(key)
        elif key not in ACCEPTED:
            errors[key] = "not an accepted training argument"
        elif value is not None:
            problem = _range_error(key, value)
            if problem:
                errors[key] = problem
            else:
                accepted[key] = value
    if errors:
        detail = "; ".join(f"{k}: {v}" for k, v in sorted(errors.items()))
        raise ConfigError(f"invalid train_args: {detail}")
    return accepted, sorted(ignored)


def _range_error(key: str, value: Any) -> str | None:
    number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if key == "epochs" and not (isinstance(value, int) and not isinstance(value, bool)
                                and 1 <= value <= 1000):
        return "must be an integer in 1..1000"
    if key == "batch" and not (number and (value == -1 or 0 < value < 1
                                           or (float(value).is_integer() and 1 <= value <= 256))):
        return "must be -1, a memory fraction in (0, 1), or an integer in 1..256"
    if key == "imgsz" and not (isinstance(value, int) and 32 <= value <= 4096):
        return "must be an integer in 32..4096"
    if key == "workers" and not (isinstance(value, int) and 0 <= value <= 64):
        return "must be an integer in 0..64"
    if key == "fraction" and not (number and 0 < value <= 1):
        return "must be in (0, 1]"
    return None


@dataclass(frozen=True)
class ReportTarget:
    base_url: str
    owner: str
    project: str
    model: str
    token: str = field(repr=False)
    model_id: str | None = None
    dataset_slug: str | None = None
    dataset_version: int | None = None
    heartbeat_seconds: float = 600.0
    protocol: str = "webhook"
    rest: RestConfig | None = None

    def client(self) -> RestClient:
        return RestClient(self.base_url, self.token, self.rest or RestConfig())

    @classmethod
    def from_config(cls, section: dict[str, Any]) -> ReportTarget | None:
        base_url = str(section.get("api_base_url") or "").rstrip("/")
        if not base_url:
            return None
        token_env = section.get("token_env") or "REPORT_TO_TOKEN"
        token = os.environ.get(token_env, "").strip()
        if not token:
            raise ConfigError(f"report_to.api_base_url is set but ${token_env} is empty")
        if any(ch.isspace() or not ch.isprintable() for ch in token):
            raise ConfigError(f"${token_env} contains whitespace or control characters")
        redact.register(token)
        parts = str(section.get("model") or "").strip("/").split("/")
        if len(parts) != 3:
            raise ConfigError("report_to.model (--report-to-model) must be owner/project/model")
        owner, project, model = parts
        if not (4 <= len(owner) <= 32 and _SLUG.match(owner)):
            raise ConfigError(f"report_to owner {owner!r}: 4-32 lowercase letters, digits, hyphens")
        for label, value in (("project", project), ("model", model)):
            if not (len(value) <= 128 and _SLUG.match(value)):
                raise ConfigError(f"report_to {label} {value!r}: lowercase letters, digits, hyphens")
        slug = section.get("dataset_slug")
        if slug and len(str(slug).strip("/").split("/")) != 2:
            raise ConfigError("--dataset-slug must be owner/dataset")
        version = section.get("dataset_version")
        protocol = str(section.get("protocol") or "webhook").lower()
        if protocol not in PROTOCOLS:
            raise ConfigError(f"report_to.protocol must be one of {', '.join(PROTOCOLS)}")
        model_id = str(section["model_id"]) if section.get("model_id") else None
        if protocol == "rest" and not model_id:
            raise ConfigError("report_to.protocol=rest needs --report-to-model-id "
                              "(the assetId weights are uploaded to)")
        rest = RestConfig.from_section(section)
        if rest.metrics_every_epochs < 1 or rest.checkpoint_every_epochs < 0:
            raise ConfigError("report_to.metrics_every_epochs must be >= 1 and "
                              "report_to.checkpoint_every_epochs >= 0")
        return cls(
            base_url=base_url, owner=owner, project=project, model=model, token=token,
            model_id=model_id,
            dataset_slug=str(slug).strip("/") if slug else None,
            dataset_version=int(version) if version not in (None, "") else None,
            heartbeat_seconds=float(section.get("heartbeat_seconds", 600)),
            protocol=protocol, rest=rest,
        )


class UltralyticsEngine(Engine):
    name = "ultralytics"

    def __init__(self) -> None:
        self._args: dict[str, Any] = {}
        self._report: ReportTarget | None = None
        self._weights: str = ""
        self._data: str = ""

    def dataset_specs(self, ctx: RunContext):
        # Runs before any download, so a bad train_args or report_to exits 3 without fetching.
        raw = ctx.config.get("train_args") or {}
        if not isinstance(raw, dict):
            raise ConfigError("train_args must be a JSON object")
        self._args, ignored = check_train_args(raw)
        if ignored:
            logger.warning("ignored train_args: %s", ", ".join(ignored))
            events.emit("ignored_args", args=ignored)
        self._report = ReportTarget.from_config(ctx.section("report_to"))
        _setup_ultralytics(ctx, self._report)
        _check_types(self._args)

        specs = specs_from_params(ctx)
        if "dataset" not in specs:
            from_config = config_dataset(ctx)
            if not from_config:
                raise ConfigError("no dataset: pass --data/--dataset (or DATASET_URI)")
            specs["dataset"] = make_spec(ctx, from_config)
        report = self._report
        if report and report.dataset_slug and specs["dataset"].scheme in {"http", "https"}:
            specs["dataset"] = replace(specs["dataset"], refresh=partial(resign_dataset_url, report))
        return specs

    def preflight(self, ctx: RunContext) -> None:
        self._weights = resolve_weights(ctx)
        events.stage("converting")
        self._data = prepare_data(ctx.dataset("dataset").path, ctx.cache_dir / "ultralytics",
                                  self._args.get("fraction", 1.0))

    def run(self, ctx: RunContext) -> RunResult:
        os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
        import torch
        from ultralytics import YOLO

        output = ensure_dir(ctx.output)
        last = output / "weights" / "last.pt"
        report = self._report
        if ctx.resume and last.is_file():
            model, kwargs = YOLO(str(last)), {"resume": True}
            logger.info("resuming from %s", last)
        else:
            if ctx.resume:
                logger.warning("--resume requested but %s does not exist; starting fresh", last)
            model = YOLO(self._weights)
            kwargs = {
                **self._args, "data": self._data, "save_dir": str(output), "exist_ok": True,
                "device": "0" if torch.cuda.is_available() else "cpu",
            }
            if report and report.protocol == "webhook":
                kwargs.update(project=f"{report.owner}/{report.project}", name=report.model)
        hooks = _Hooks(ctx, report, output)
        for event, fn in hooks.callbacks().items():
            model.add_callback(event, fn)
        try:
            with hooks.watch_platform():
                model.train(**kwargs)
        except BaseException as exc:
            hooks.failed(getattr(model, "trainer", None), exc)
            if isinstance(exc, KeyboardInterrupt):
                raise
            if isinstance(exc, DistillRunError):
                raise
            logger.exception("ultralytics training failed")
            raise classify(exc) from exc
        finally:
            hooks.stop_heartbeat()
        trainer = model.trainer
        summary = hooks.summary(trainer)
        problem = hooks.finish(trainer, summary)
        if ctx.cancellation.requested:
            raise CancelledError(f"cancelled at epoch {summary['epochs_completed']}; "
                                 f"weights so far are in {output / 'weights'}")
        if problem:
            raise ReportError(f"{problem}; local results are in {output}")
        best = Path(trainer.best)
        artifacts = [p for p in (best, Path(trainer.last), output / "results.csv",
                                 output / "summary.json") if p.exists()]
        metrics = summary.get("metrics") or {}
        return RunResult(
            artifacts=artifacts,
            message=f"best epoch {summary['best_epoch']} "
            f"mAP50-95={metrics.get('metrics/mAP50-95(B)', float('nan')):.4f} -> {output}",
        )


def _setup_ultralytics(ctx: RunContext, report: ReportTarget | None) -> None:
    """Environment the ``ultralytics`` import reads once; call before importing it."""
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ensure_dir(ctx.work_dir / "ultralytics")))
    os.environ["YOLO_AUTOINSTALL"] = "False"
    webhook = report is not None and report.protocol == "webhook"
    if webhook:
        os.environ["ULTRALYTICS_PLATFORM_URL"] = report.base_url
        os.environ["ULTRALYTICS_API_KEY"] = report.token
    else:
        os.environ.pop("ULTRALYTICS_API_KEY", None)
    try:
        from ultralytics.utils import LOGGER, SETTINGS
    except ImportError as exc:
        raise PreflightError(
            f"ultralytics engine needs the ultralytics package: {exc}; "
            "run scripts/setup_envs.py --engines ultralytics"
        ) from exc
    if webhook:
        try:
            import ultralytics_platform  # noqa: F401
        except ImportError as exc:
            raise PreflightError(f"reporting needs ultralytics-platform and httpx: {exc}") from exc
    redact.protect(LOGGER, logging.getLogger("distill_run"))
    weights = ensure_dir(Path(ctx.model_cache_dir) / "yolo")
    SETTINGS.update({
        "weights_dir": str(weights),
        "datasets_dir": str(ensure_dir(ctx.cache_dir / "ultralytics")),
        "runs_dir": str(ensure_dir(ctx.work_dir / "ultralytics-runs")),
        "sync": False, "api_key": "",
        **{k: False for k in ("clearml", "comet", "dvc", "mlflow", "raytune", "tensorboard",
                              "wandb")},
    })


def _check_types(args: dict[str, Any]) -> None:
    from ultralytics.cfg import get_cfg

    try:
        get_cfg(overrides=dict(args))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"invalid train_args: {exc}") from exc


def resign_dataset_url(report: ReportTarget) -> str:
    """A fresh signed NDJSON URL from ``GET /api/datasets/{owner}/{dataset}/export?v=N``."""
    owner, dataset = report.dataset_slug.split("/")
    try:
        url = _resign(report.client(), owner, dataset, report.dataset_version)
    except PlatformHTTPError as exc:
        raise DatasetFetchError(f"re-signing dataset {report.dataset_slug}: {exc}") from exc
    logger.info("re-signed dataset URL: %s", redact_url(url))
    return url


def resolve_weights(ctx: RunContext) -> str:
    """Local weights path, or a spec Ultralytics resolves itself (``*.yaml``, ``ul://``)."""
    spec = str(ctx.require("student")).strip()
    sha256 = ctx.param("model_sha256")
    cache = ensure_dir(Path(ctx.model_cache_dir) / "yolo")
    lower = spec.lower()
    if lower.startswith("ul://") or lower.endswith((".yaml", ".yml")):
        if sha256:
            raise ConfigError(f"--model-sha256 cannot be checked for {spec}")
        return spec
    if "/" not in spec and "\\" not in spec and lower.endswith(".pt") and not Path(spec).exists():
        url = f"{(ctx.section('yolo').get('weights_url') or weights_base_url()).rstrip('/')}/{spec}"
        return str(download_verified(url, cache / spec, sha256))
    return str(fetch_weights(spec, cache, sha256))


def prepare_data(path: Path, out_dir: Path, fraction: float) -> str:
    """Convert an NDJSON export to YOLO txt and check the dataset; return what train() takes."""
    from ultralytics.data.utils import check_det_dataset

    try:
        if str(path).lower().endswith(".ndjson"):
            from ultralytics.data.converter import convert_ndjson_to_yolo

            path = asyncio.run(convert_ndjson_to_yolo(path, ensure_dir(out_dir), fraction,
                                                      split="val"))
        check_det_dataset(str(path))
    except ImportError as exc:
        raise PreflightError(f"converting {Path(path).name} needs {exc.name or exc}; "
                             "run scripts/setup_envs.py --engines ultralytics") from exc
    except (ConnectionError, TimeoutError, OSError) as exc:
        raise DatasetFetchError(f"dataset {Path(path).name}: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - Ultralytics raises bare Exceptions for bad layouts
        raise DataError(f"dataset {Path(path).name}: {exc}") from exc
    return str(path)


class _Hooks:
    """Distill-run callbacks next to Ultralytics' own (ours run first for each event)."""

    def __init__(self, ctx: RunContext, report: ReportTarget | None, output: Path):
        self.ctx, self.report, self.output = ctx, report, output
        self.webhook = report is not None and report.protocol == "webhook"
        self.rest: RestReporter | None = None
        if report is not None and report.protocol == "rest":
            self.rest = RestReporter(report.client(), owner=report.owner,
                                     project=report.project, model=report.model,
                                     model_id=report.model_id, dataset_slug=report.dataset_slug)
        self._beat_stop = threading.Event()
        self._beat: threading.Thread | None = None
        self.sent: dict[str, Any] = {}
        self.uploaded: dict[str, Any] | None = None
        self._last_epoch = 0
        self._final_eval = False
        self._queue = ThreadPoolExecutor(max_workers=1, thread_name_prefix="report")
        self._checkpoint: Future | None = None

    @contextlib.contextmanager
    def watch_platform(self):
        """Record what the Platform callbacks sent; they only log a warning when a call fails."""
        if not self.webhook:
            yield
            return
        from ultralytics.utils.callbacks import platform

        send, upload = platform._send, platform._upload_model

        def recorded_send(event, *args, **kwargs):
            response = send(event, *args, **kwargs)
            self.sent[event] = response
            return response

        def recorded_upload(*args, **kwargs):
            self.uploaded = upload(*args, **kwargs)
            return self.uploaded

        platform._send, platform._upload_model = recorded_send, recorded_upload
        try:
            yield
        finally:
            platform._send, platform._upload_model = send, upload

    def finish(self, trainer, summary: dict[str, Any]) -> str | None:
        """Send the final results; return what did not reach the platform, if anything."""
        if self.rest:
            return self._rest_finish(trainer, summary)
        return self.report_problem(trainer)

    def report_problem(self, trainer) -> str | None:
        if not self.webhook:
            return None
        if not getattr(trainer, "platform", None):
            return "Ultralytics Platform did not accept the run"
        weights = Path(trainer.best if trainer.best and Path(trainer.best).exists() else trainer.last)
        if weights.exists() and not self.uploaded:
            return f"uploading {weights.name} to the platform failed"
        if self.sent.get("training_complete") is None:
            return "the platform did not accept training_complete"
        return None

    def callbacks(self) -> dict[str, Any]:
        return {
            "on_pretrain_routine_start": self.pretrain_start,
            "on_train_start": self.train_start,
            "on_train_batch_end": self.batch_end,
            "on_fit_epoch_end": self.fit_epoch_end,
            "on_model_save": self.model_save,
            "on_train_end": self.train_end,
        }

    def pretrain_start(self, trainer) -> None:
        # final_eval() re-validates best.pt and fires on_fit_epoch_end once more with
        # epoch + 1; that call is not an epoch.
        final_eval = trainer.final_eval

        def flagged_final_eval():
            self._final_eval = True
            final_eval()

        trainer.final_eval = flagged_final_eval
        report = self.report
        if not self.webhook or not report.model_id:
            return
        from ultralytics.utils.callbacks import platform

        # The Platform callback reuses platform_model_id.txt only when resuming, so present
        # the run as a resume for that one call to bind events to report_to.model_id.
        record = ensure_dir(Path(trainer.save_dir)) / "platform_model_id.txt"
        record.write_text(report.model_id, encoding="utf-8")
        chain = trainer.callbacks["on_pretrain_routine_start"]
        if platform.on_pretrain_routine_start in chain:
            chain[chain.index(platform.on_pretrain_routine_start)] = partial(_bound_start, platform)

    def train_start(self, trainer) -> None:
        if self.rest:
            try:
                self.rest.patch("running", epochs=0, metadata=self._metadata())
            except PlatformHTTPError as exc:
                if exc.status in (400, 401, 403, 404, 409):
                    raise ReportError(f"the platform rejected the run: {exc}") from exc
                logger.warning("reporting the start of training failed: %s", exc)
        ctx = getattr(trainer, "platform", None)
        if self.webhook and not ctx:
            logger.warning("Ultralytics Platform did not accept the run; metrics are not reported")
        if (ctx or self.rest) and self.report.heartbeat_seconds > 0:
            self._beat = threading.Thread(target=self._heartbeat, args=(trainer,), daemon=True)
            self._beat.start()

    def batch_end(self, trainer) -> None:
        if self.ctx.cancellation.requested and not trainer.stop:
            logger.warning("cancellation requested; stopping after this batch")
            trainer.stop = True

    def fit_epoch_end(self, trainer) -> None:
        if self._final_eval or trainer.epoch + 1 <= self._last_epoch:
            return
        self._last_epoch = trainer.epoch + 1
        done = trainer.epoch + 1 - trainer.start_epoch
        per_epoch = (time.time() - trainer.train_time_start) / max(done, 1)
        fitness = trainer.fitness
        events.progress(trainer.epoch + 1, trainer.epochs, per_epoch * (trainer.epochs - trainer.epoch - 1),
                        fitness=None if fitness is None or not math.isfinite(fitness) else round(fitness, 5))
        if self.rest and self._last_epoch % self.rest.config.metrics_every_epochs == 0:
            fields = self._results(trainer)
            self._queue.submit(self._quietly, "metrics", self.rest.patch, "running", **fields)

    def model_save(self, trainer) -> None:
        every = self.rest.config.checkpoint_every_epochs if self.rest else 0
        epoch = trainer.epoch + 1
        if not every or epoch % every or epoch >= trainer.epochs:
            return
        if self._checkpoint and not self._checkpoint.done():
            logger.info("previous checkpoint upload still running; skipping epoch %d", epoch)
            return
        last = Path(trainer.last)
        if not last.is_file():
            return
        # last.pt is rewritten every epoch, so upload a copy.
        copy = last.with_name("checkpoint-upload.pt")
        shutil.copyfile(last, copy)
        self._checkpoint = self._queue.submit(self._quietly, f"checkpoint epoch {epoch}",
                                              self.rest.upload, copy, "last.pt")

    def train_end(self, trainer) -> None:
        self.stop_heartbeat()
        if getattr(trainer, "platform", None) or self.rest:
            events.stage("uploading")

    def _rest_finish(self, trainer, summary: dict[str, Any]) -> str | None:
        self._queue.shutdown(wait=True)
        (self.output / "weights" / "checkpoint-upload.pt").unlink(missing_ok=True)
        problems = []
        weights = Path(trainer.best if trainer.best and Path(trainer.best).exists() else trainer.last)
        if weights.exists():
            try:
                self.uploaded = self.rest.upload(weights)
            except PlatformHTTPError as exc:
                logger.error("uploading %s failed: %s", weights.name, exc)
                problems.append(f"uploading {weights.name} to the platform failed")
        status = "cancelled" if self.ctx.cancellation.requested else "completed"
        try:
            self.rest.patch(status, **self._results(trainer), train_results=summary)
        except PlatformHTTPError as exc:
            logger.error("writing the results failed: %s", exc)
            problems.append("the platform did not accept the results")
        return "; ".join(problems) or None

    def _results(self, trainer) -> dict[str, Any]:
        stopper = getattr(trainer, "stopper", None)
        best = trainer.best_fitness
        return {
            "metrics": {k: float(v) for k, v in (trainer.metrics or {}).items()
                        if isinstance(v, (int, float)) and math.isfinite(float(v))},
            "epochs": min(trainer.epoch + 1, trainer.epochs),
            "best_epoch": getattr(stopper, "best_epoch", None),
            "best_fitness": float(best) if best is not None and math.isfinite(float(best)) else None,
            "metadata": self._metadata(),
        }

    def _metadata(self) -> dict[str, Any]:
        return {"run_id": self.ctx.run_id, "updated_at": events.utc_now()}

    @staticmethod
    def _quietly(what: str, fn, *args, **kwargs) -> None:
        try:
            fn(*args, **kwargs)
        except PlatformHTTPError as exc:
            logger.warning("reporting %s failed: %s", what, exc)

    def _heartbeat(self, trainer) -> None:
        if self.rest:
            # Status and metadata only: a delayed retry must not move epochs backwards.
            while not self._beat_stop.wait(self.report.heartbeat_seconds):
                self._quietly("heartbeat", self.rest.patch, "running", metadata=self._metadata())
            return
        from ultralytics.utils.callbacks import platform

        while not self._beat_stop.wait(self.report.heartbeat_seconds):
            ctx = getattr(trainer, "platform", None)
            if not ctx:
                return
            project, name = platform._get_project_name(trainer)
            platform._send("training_progress", {"epoch": trainer.epoch, "epochs": trainer.epochs,
                                                 "runId": ctx.get("run_id")},
                           project, name, ctx.get("model_id"), retry=1)

    def stop_heartbeat(self) -> None:
        self._beat_stop.set()

    def failed(self, trainer, exc: BaseException) -> None:
        if self.rest:
            self._queue.shutdown(wait=True, cancel_futures=True)
            if isinstance(exc, ReportError):
                return
            cancelled = isinstance(exc, KeyboardInterrupt) or self.ctx.cancellation.requested
            error = exc if isinstance(exc, DistillRunError) else classify(exc)
            self._quietly("the failure", self.rest.patch, "cancelled" if cancelled else "failed",
                          error=None if cancelled else f"{error.code}: {error.message}",
                          metadata=self._metadata())
            return
        ctx = getattr(trainer, "platform", None) if trainer is not None else None
        if not ctx or not ctx.get("model_id"):
            return
        from ultralytics.utils.callbacks import platform

        error = exc if isinstance(exc, DistillRunError) else classify(exc)
        project, name = platform._get_project_name(trainer)
        platform._send("training_failed", {"error": error.message[:2000], "code": error.code,
                                           "runId": ctx.get("run_id")},
                       project, name, ctx["model_id"], retry=3)

    def summary(self, trainer) -> dict[str, Any]:
        stopper = getattr(trainer, "stopper", None)
        ctx = getattr(trainer, "platform", None) or {}
        metrics = {k: float(v) for k, v in (trainer.metrics or {}).items()
                   if isinstance(v, (int, float)) and math.isfinite(float(v))}
        summary = {
            "epochs_completed": trainer.epoch + 1 - trainer.start_epoch,
            "total_epochs": trainer.epochs,
            "best_epoch": getattr(stopper, "best_epoch", None),
            "best_fitness": float(trainer.best_fitness) if trainer.best_fitness is not None else None,
            "metrics": metrics,
            "cancelled": self.ctx.cancellation.requested,
            "seconds": round(time.time() - trainer.train_time_start, 1),
            "weights": {"best": str(trainer.best), "last": str(trainer.last)},
            "platform": {"model_id": ctx.get("model_id"), "run_id": ctx.get("run_id"),
                         "model_url": ctx.get("model_url")} if ctx else (
                {"protocol": "rest", "model_id": self.report.model_id} if self.rest else None),
        }
        atomic_write_json(self.output / "summary.json", summary)
        logger.info("summary: %s", json.dumps(summary, default=str))
        return summary


def _bound_start(platform, trainer) -> None:
    resume = trainer.args.resume
    trainer.args.resume = True
    try:
        platform.on_pretrain_routine_start(trainer)
    finally:
        trainer.args.resume = resume
