"""YOLO detection training on the built-in implementation in :mod:`distill_run.yolo`.

``--model`` accepts:

* an architecture name (``yolo26n``, ``yolo11s``, ``yolov8m``, ``yolov5lu`` ...):
  random init, or official weights with ``--pretrained``;
* the same name with ``.pt`` (``yolo26n.pt``): official weights;
* a local ``.pt`` (official or written by this engine), or a directory holding
  ``config.json`` + ``best.safetensors`` from an earlier run;
* ``http(s)://.../file.pt``, ``hf://org/repo/file.pt`` or ``ms://org/repo/file.pt``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from ..context import RunContext
from ..errors import ConfigError, DataError, ModelFetchError, PreflightError, classify
from ..utils import HTTP_TIMEOUT, ensure_dir, http_session
from .base import Engine, RunResult, config_dataset, make_spec, specs_from_params

logger = logging.getLogger(__name__)

OFFICIAL_WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0"


@dataclass
class ModelChoice:
    family: str
    scale: str
    weights: Path | None


class YoloEngine(Engine):
    name = "yolo"

    def __init__(self) -> None:
        self._data: Any = None
        self._model: ModelChoice | None = None

    def dataset_specs(self, ctx: RunContext):
        specs = specs_from_params(ctx)
        if "dataset" not in specs:
            from_config = config_dataset(ctx)
            if not from_config:
                raise ConfigError(
                    "no detection data: pass --data/--dataset (or DATASET_URI), "
                    "or set dataset.path in --config"
                )
            specs["dataset"] = make_spec(ctx, from_config)
        return specs

    def preflight(self, ctx: RunContext) -> None:
        try:
            import cv2  # noqa: F401
            import torch  # noqa: F401
            import torchvision  # noqa: F401
        except ImportError as exc:
            raise PreflightError(
                f"yolo engine needs torch, torchvision and opencv: {exc}; "
                "run scripts/setup_envs.py --engines yolo"
            ) from exc
        from ..yolo.data import DataFormatError, load_detection_data
        from ..yolo.train import TrainConfig

        section = ctx.section("yolo")
        try:
            TrainConfig.from_mapping(_train_options(section))
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"yolo config: {exc}") from exc
        try:
            data = load_detection_data(
                ctx.dataset("dataset").path,
                names=section.get("names"),
                val_fraction=float(section.get("val_fraction", 0.0)),
                seed=int(section.get("seed", 0)),
            )
        except (DataFormatError, OSError, ValueError, KeyError) as exc:
            raise DataError(f"detection dataset: {exc}") from exc
        logger.info("dataset: %s", json.dumps(data.summary()))
        for note in data.notes:
            logger.warning("dataset: %s", note)
        self._data = data
        self._model = resolve_model(ctx)

    def run(self, ctx: RunContext) -> RunResult:
        import torch

        from ..yolo.model import DetectionModel
        from ..yolo.train import TrainConfig, Trainer
        from ..yolo.weights import apply_state_dict, load_checkpoint

        data, choice = self._data, self._model
        cfg = TrainConfig.from_mapping(_train_options(ctx.section("yolo")))
        output = ctx.output
        output.mkdir(parents=True, exist_ok=True)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu":
            logger.warning("no GPU visible; training on CPU")
        else:
            logger.info("device: %s (%s)", torch.cuda.get_device_name(0),
                        os.environ.get("HIP_VISIBLE_DEVICES", "all"))
        try:
            model = DetectionModel(choice.family, choice.scale, nc=data.nc)
            info = model.describe()
            logger.info("model %s%s: %.2fM parameters, nc=%d", choice.family, choice.scale,
                        info["parameters"] / 1e6, data.nc)
            if choice.weights is not None:
                ckpt = load_checkpoint(choice.weights)
                report = apply_state_dict(model, ckpt["state_dict"])
                if report["loaded"] < 0.5 * report["total"]:
                    raise ConfigError(
                        f"{choice.weights}: only {report['loaded']}/{report['total']} tensors "
                        f"match {choice.family}{choice.scale}"
                    )
                if report["shape_mismatch"]:
                    logger.info("re-initialised %d class-dependent tensors (nc %s -> %d)",
                                len(report["shape_mismatch"]), ckpt.get("nc"), data.nc)
            trainer = Trainer(
                model, data, cfg, output, device=device,
                should_stop=lambda: ctx.cancellation.requested, resume=ctx.resume,
            )
            summary = trainer.train()
        except Exception as exc:  # noqa: BLE001 - normalise OOM and friends
            logger.exception("yolo training failed")
            raise classify(exc) from exc
        artifacts = [Path(p) for p in summary["artifacts"]]
        metrics = summary.get("final_metrics") or {}
        message = (
            f"{summary['model']} trained {summary['global_step']} steps, "
            f"mAP50-95={metrics.get('mAP50-95', float('nan')):.4f} -> {output}"
        )
        return RunResult(artifacts=artifacts, message=message)


_ENGINE_ONLY_KEYS = {"names", "val_fraction", "pretrained", "weights_url"}


def _train_options(section: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in section.items() if k not in _ENGINE_ONLY_KEYS}


def resolve_model(ctx: RunContext) -> ModelChoice:
    from ..yolo.model import parse_name, weights_filename

    spec = str(ctx.require("student")).strip()
    section = ctx.section("yolo")
    pretrained = bool(section.get("pretrained", False))
    cache = ensure_dir(Path(ctx.model_cache_dir) / "yolo")
    base_url = section.get("weights_url") or os.environ.get("YOLO_WEIGHTS_URL") or OFFICIAL_WEIGHTS_URL

    bare = spec[:-3] if spec.lower().endswith(".pt") and "/" not in spec else spec
    parsed = parse_name(bare)
    if parsed and "/" not in spec and not Path(spec).exists():
        family, scale = parsed
        if spec.lower().endswith(".pt") or pretrained:
            name = weights_filename(family, scale)
            weights = _download(f"{base_url.rstrip('/')}/{name}", cache / name)
            return ModelChoice(family, scale, weights)
        return ModelChoice(family, scale, None)

    scheme = urlparse(spec).scheme.lower()
    if scheme in {"http", "https"}:
        name = Path(unquote(urlparse(spec).path)).name or "weights.pt"
        path = _download(spec, cache / name)
    elif scheme in {"hf", "ms", "modelscope"}:
        path = _hub_file(spec, cache)
    else:
        path = Path(spec[len("file://"):] if spec.startswith("file://") else spec).expanduser()
        if not path.exists():
            raise ModelFetchError(
                f"model not found: {spec}; use a name such as yolo26n / yolo11s.pt / yolov8m, "
                "a local .pt, or an http(s)/hf/ms URI"
            )
    return _from_weights(path)


def _from_weights(path: Path) -> ModelChoice:
    from ..yolo.model import parse_name

    if path.is_dir():
        config = path / "config.json"
        weights = next(iter(sorted(path.glob("*.safetensors"))), None) or next(
            iter(sorted(path.glob("*.pt"))), None
        )
        if not config.is_file() or weights is None:
            raise ModelFetchError(f"{path}: expected config.json plus a .safetensors or .pt file")
        meta = json.loads(config.read_text(encoding="utf-8"))
        return ModelChoice(meta["family"], meta["scale"], _as_checkpoint(weights, meta))

    from ..yolo.weights import load_checkpoint

    try:
        ckpt = load_checkpoint(path)
    except Exception as exc:  # noqa: BLE001 - corrupt or foreign pickle
        raise ModelFetchError(f"cannot read checkpoint {path}: {exc}") from exc
    if ckpt.get("family") and ckpt.get("scale"):
        return ModelChoice(ckpt["family"], ckpt["scale"], path)
    for hint in (ckpt.get("yaml_file"), path.stem):
        parsed = parse_name(Path(str(hint or "")).stem)
        if parsed:
            family, _ = parsed
            scale = ckpt.get("scale") or parsed[1]
            return ModelChoice(family, scale, path)
    raise ModelFetchError(f"{path}: cannot tell the architecture; rename it like yolo11s.pt")


def _as_checkpoint(weights: Path, meta: dict[str, Any]) -> Path:
    if weights.suffix != ".safetensors":
        return weights
    import torch
    from safetensors.torch import load_file

    target = weights.with_suffix(".converted.pt")
    if not target.exists():
        torch.save({**meta, "state_dict": load_file(str(weights))}, target)
    return target


def _download(url: str, target: Path) -> Path:
    if target.is_file() and target.stat().st_size > 0:
        logger.info("weights cache hit %s", target)
        return target
    tmp = target.with_suffix(target.suffix + f".tmp-{os.getpid()}")
    logger.info("downloading %s -> %s", url, target)
    try:
        with http_session().get(url, stream=True, timeout=HTTP_TIMEOUT) as resp:
            resp.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in resp.iter_content(1 << 20):
                    f.write(chunk)
    except Exception as exc:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        raise ModelFetchError(f"failed to download {url}: {exc}") from exc
    if tmp.stat().st_size < 100_000:
        tmp.unlink(missing_ok=True)
        raise ModelFetchError(f"{url}: download too small to be a checkpoint")
    tmp.replace(target)
    return target


def _hub_file(spec: str, cache: Path) -> Path:
    scheme, rest = spec.split("://", 1)
    parts = rest.strip("/").split("/")
    if len(parts) < 3:
        raise ModelFetchError(f"{spec}: expected {scheme}://org/repo/<file>.pt")
    repo, filename = "/".join(parts[:2]), "/".join(parts[2:])
    try:
        if scheme == "hf":
            from huggingface_hub import hf_hub_download

            return Path(hf_hub_download(repo_id=repo, filename=filename, cache_dir=str(cache)))
        from modelscope.hub.file_download import model_file_download

        return Path(model_file_download(repo, filename, cache_dir=str(cache)))
    except ImportError as exc:
        raise ModelFetchError(f"{scheme}:// weights need the hub SDK: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise ModelFetchError(f"failed to fetch {spec}: {exc}") from exc
