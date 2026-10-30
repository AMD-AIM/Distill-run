"""Single-GPU detection training loop.

Outputs under the run directory::

    weights/last.pt   resumable state (model, EMA, optimizer, scaler, counters)
    weights/best.pt   EMA weights with the best fitness so far
    results.csv       one row per epoch (Ultralytics column names)
    metrics.jsonl     the same rows as JSON, plus step and fitness
    results.png       loss and metric curves (needs matplotlib)
    best.safetensors  final EMA weights of the best epoch, plus config.json
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .data import DetectionData, DetectionDataset, collate
from .loss import build_loss, xywh2xyxy
from .metrics import DetectionMetrics, end2end_filter, non_max_suppression
from .model import DetectionModel
from .report import LOSSES, TrainingLog, epoch_row

logger = logging.getLogger("distill_run.yolo")

AUG_KEYS = ("hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "fliplr", "mosaic")


@dataclass
class TrainConfig:
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    workers: int = 8
    optimizer: str = "auto"
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    cos_lr: bool = False
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5
    nbs: int = 64
    amp: bool = True
    close_mosaic: int = 10
    max_steps: int | None = None
    val: bool = True
    conf: float = 0.001
    iou: float = 0.7
    max_det: int = 300
    seed: int = 0
    log_interval: int = 10
    autobatch_fraction: float = 0.6
    augment: dict[str, float] = field(
        default_factory=lambda: {
            "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4, "degrees": 0.0, "translate": 0.1,
            "scale": 0.5, "shear": 0.0, "fliplr": 0.5, "mosaic": 1.0,
        }
    )

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> TrainConfig:
        known = {f.name for f in fields(cls)}
        unknown = sorted(k for k in raw if k not in known and k not in AUG_KEYS)
        if unknown:
            raise ValueError(f"unknown yolo options: {', '.join(unknown)}")
        values = {k: v for k, v in raw.items() if k in known and v is not None}
        cfg = cls(**values)
        cfg.augment = {**cls().augment, **(values.get("augment") or {}),
                       **{k: raw[k] for k in AUG_KEYS if raw.get(k) is not None}}
        return cfg


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999, tau: int = 2000, updates: int = 0):
        self.ema = deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay, self.tau, self.updates = decay, tau, updates

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.tau))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)


def _seed_worker(worker_id: int) -> None:  # noqa: ARG001
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def autobatch(model: DetectionModel, imgsz: int, device: torch.device, fraction: float,
              amp: bool) -> int:
    """Largest batch expected to use ``fraction`` of GPU memory, from a linear fit."""
    if device.type != "cuda":
        return 16
    free, total = torch.cuda.mem_get_info(device)
    budget = min(total * fraction, free * 0.9)
    probe = deepcopy(model).to(device).train()
    sizes, used = [], []
    for b in (1, 2, 4, 8, 16):
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            base = torch.cuda.memory_allocated(device)
            x = torch.rand(b, 3, imgsz, imgsz, device=device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                out = probe(x)
            branches = [out["one2many"], out["one2one"]] if "one2many" in out else [out]
            loss = sum(br["boxes"].float().mean() + br["scores"].float().mean() for br in branches)
            loss.backward()
            used.append(torch.cuda.max_memory_allocated(device) - base)
            sizes.append(b)
            probe.zero_grad(set_to_none=True)
            del x, out, loss
        except torch.OutOfMemoryError:
            break
    del probe
    torch.cuda.empty_cache()
    if len(sizes) < 2:
        return max(sizes or [1])
    slope, intercept = np.polyfit(sizes, used, 1)
    fixed = torch.cuda.memory_allocated(device)
    batch = int((budget - fixed - intercept) / max(slope, 1))
    batch = max(1, min(batch, 1024))
    logger.info(
        "autobatch: %.1f/%.1f GiB free/total, %.2f GiB per image -> batch %d (%.0f%% target)",
        free / 2**30, total / 2**30, slope / 2**30, batch, fraction * 100,
    )
    return batch


def build_optimizer(model: nn.Module, cfg: TrainConfig, iterations: int, nc: int):
    biases, decay, no_decay = [], [], []
    for module in model.modules():
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            if name == "bias":
                biases.append(p)
            elif isinstance(module, nn.BatchNorm2d):
                no_decay.append(p)
            else:
                decay.append(p)
    name, lr, momentum = cfg.optimizer.lower(), cfg.lr0, cfg.momentum
    if name == "auto":
        if iterations > 10000:
            name, lr, momentum = "sgd", 0.01, 0.9
        else:
            name, lr, momentum = "adamw", round(0.002 * 5 / (4 + nc), 6), 0.9
            cfg.warmup_bias_lr = 0.0
        logger.info("optimizer=auto -> %s lr=%g momentum=%g", name, lr, momentum)
    groups = [{"params": biases, "weight_decay": 0.0}, {"params": decay, "weight_decay": cfg.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    if name == "sgd":
        opt = torch.optim.SGD(groups, lr=lr, momentum=momentum, nesterov=True)
    elif name in {"adam", "adamw"}:
        cls = torch.optim.AdamW if name == "adamw" else torch.optim.Adam
        opt = cls(groups, lr=lr, betas=(momentum, 0.999))
    else:
        raise ValueError(f"unsupported optimizer {cfg.optimizer}; use auto, SGD, Adam or AdamW")
    cfg.lr0, cfg.momentum = lr, momentum
    return opt


@torch.no_grad()
def evaluate(model: DetectionModel, loader, device: torch.device, cfg: TrainConfig,
             nc: int, amp: bool, loss_fn=None) -> tuple[dict[str, float], dict[str, float]]:
    """Return ``(metrics, val_losses)``; losses are empty without ``loss_fn``."""
    model.eval()
    metrics = DetectionMetrics(nc)
    loss_sum, batches = torch.zeros(3, device=device), 0
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True).float() / 255
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            decoded, raw = model(images)
        if loss_fn is not None:
            loss_sum += loss_fn(raw, labels.to(device))[1]
            batches += 1
        decoded = decoded.float()
        if model.head.end2end:
            dets = end2end_filter(decoded, cfg.conf)
        else:
            dets = non_max_suppression(decoded, cfg.conf, cfg.iou, cfg.max_det)
        size = images.shape[-1]
        for i, det in enumerate(dets):
            rows = labels[labels[:, 0] == i]
            gt_boxes = xywh2xyxy(rows[:, 2:6] * size).to(device)
            metrics.update(det, rows[:, 1].to(device), gt_boxes)
    losses = dict(zip(LOSSES, (loss_sum / batches).tolist())) if batches else {}
    return metrics.compute(), losses


class Trainer:
    def __init__(
        self,
        model: DetectionModel,
        data: DetectionData,
        cfg: TrainConfig,
        save_dir: Path,
        *,
        device: torch.device,
        should_stop: Callable[[], bool] = lambda: False,
        resume: bool = False,
    ):
        self.model, self.data, self.cfg = model, data, cfg
        self.save_dir = Path(save_dir)
        self.weights_dir = self.save_dir / "weights"
        self.device = device
        self.should_stop = should_stop
        self.resume = resume
        self.amp = bool(cfg.amp) and device.type == "cuda"
        self._best_epoch = 0

    def _loaders(self, batch: int):
        cfg = self.cfg
        train_set = DetectionDataset(self.data.train, cfg.imgsz, augment=True, hyp=cfg.augment)
        val_set = DetectionDataset(self.data.val, cfg.imgsz, augment=False)
        workers = max(0, min(cfg.workers, batch))
        common = {"collate_fn": collate, "pin_memory": self.device.type == "cuda",
                  "worker_init_fn": _seed_worker}
        g = torch.Generator()
        g.manual_seed(cfg.seed)
        train = torch.utils.data.DataLoader(
            train_set, batch_size=batch, shuffle=True, num_workers=workers, generator=g, **common
        )
        val = torch.utils.data.DataLoader(
            val_set, batch_size=max(1, batch * 2), shuffle=False, num_workers=workers, **common
        )
        return train_set, train, val

    def train(self) -> dict[str, Any]:
        cfg, device = self.cfg, self.device
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        model = self.model.to(device)

        batch = cfg.batch if cfg.batch > 0 else autobatch(
            model, cfg.imgsz, device, cfg.autobatch_fraction, self.amp
        )
        train_set, train_loader, val_loader = self._loaders(batch)
        nb = len(train_loader)
        accumulate = max(round(cfg.nbs / batch), 1)
        cfg.weight_decay *= batch * accumulate / cfg.nbs
        iterations = math.ceil(len(train_set) / max(batch, cfg.nbs)) * cfg.epochs
        optimizer = build_optimizer(model, cfg, iterations, self.data.nc)
        if cfg.cos_lr:
            def lf(x):
                return ((1 - math.cos(x * math.pi / cfg.epochs)) / 2) * (cfg.lrf - 1) + 1
        else:
            def lf(x):
                return max(1 - x / cfg.epochs, 0) * (1.0 - cfg.lrf) + cfg.lrf
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        ema = ModelEMA(model)
        loss_fn = build_loss(model, {"box": cfg.box, "cls": cfg.cls, "dfl": cfg.dfl}, cfg.epochs)

        start_epoch, best_fitness, global_step = 0, -1.0, 0
        last = self.weights_dir / "last.pt"
        if self.resume and last.is_file():
            state = torch.load(last, map_location=device, weights_only=True)
            model.load_state_dict(state["model"])
            ema.ema.load_state_dict(state["state_dict"])
            ema.updates = state["ema_updates"]
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
            start_epoch, best_fitness = state["epoch"] + 1, state["best_fitness"]
            global_step = state.get("global_step", 0)
            self._best_epoch = state.get("best_epoch", 0)
            for _ in range(start_epoch):
                scheduler.step()
                if hasattr(loss_fn, "update"):
                    loss_fn.update()
            logger.info("resumed from %s at epoch %d", last, start_epoch)
        elif self.resume:
            logger.warning("--resume requested but %s does not exist; starting fresh", last)

        nw = max(round(cfg.warmup_epochs * nb), 100) if cfg.warmup_epochs > 0 else -1
        last_opt_step = -1
        log = TrainingLog(self.save_dir, start_epoch)
        elapsed_before = log.elapsed()
        best_epoch = self._best_epoch
        logger.info(
            "train: %d images, val: %d images, batch=%d accumulate=%d imgsz=%d epochs=%d amp=%s",
            len(self.data.train), len(self.data.val), batch, accumulate, cfg.imgsz, cfg.epochs,
            self.amp,
        )
        stopped, final_metrics, epoch = None, {}, start_epoch
        t_start = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(start_epoch, cfg.epochs):
            if cfg.close_mosaic and epoch == cfg.epochs - cfg.close_mosaic:
                train_set.mosaic = False
                logger.info("closing mosaic for the last %d epochs", cfg.close_mosaic)
            model.train()
            mean_items = torch.zeros(3, device=device)
            seen, t_epoch = 0, time.time()
            optimizer.zero_grad(set_to_none=True)
            for i, (images, labels, _) in enumerate(train_loader):
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]
                    accumulate = max(1, int(np.interp(ni, xi, [1, cfg.nbs / batch]).round()))
                    for j, group in enumerate(optimizer.param_groups):
                        group["lr"] = np.interp(
                            ni, xi, [cfg.warmup_bias_lr if j == 0 else 0.0,
                                     group["initial_lr"] * lf(epoch)]
                        )
                        if "momentum" in group:
                            group["momentum"] = np.interp(ni, xi, [cfg.warmup_momentum, cfg.momentum])
                images = images.to(device, non_blocking=True).float() / 255
                with torch.autocast(device.type, dtype=torch.float16, enabled=self.amp):
                    preds = model(images)
                loss, items = loss_fn(preds, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss {loss.item()} at step {global_step}")
                scaler.scale(loss).backward()
                if ni - last_opt_step >= accumulate:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    ema.update(model)
                    last_opt_step = ni
                global_step += 1
                seen += images.shape[0]
                mean_items = (mean_items * i + items) / (i + 1)
                if global_step % cfg.log_interval == 0 or i == nb - 1:
                    box, cls_, dfl = mean_items.tolist()
                    mem = torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else 0
                    logger.info(
                        "epoch %d/%d iter %d/%d step %d box=%.4f cls=%.4f dfl=%.4f lr=%.6f "
                        "%.1f img/s mem=%.2fGiB",
                        epoch + 1, cfg.epochs, i + 1, nb, global_step, box, cls_, dfl,
                        optimizer.param_groups[1]["lr"], seen / max(time.time() - t_epoch, 1e-6), mem,
                    )
                if self.should_stop():
                    stopped = "cancelled"
                    break
                if cfg.max_steps and global_step >= cfg.max_steps:
                    stopped = "max_steps"
                    break
            scheduler.step()
            if hasattr(loss_fn, "update"):
                loss_fn.update()

            metrics, val_losses = (
                evaluate(ema.ema, val_loader, device, cfg, self.data.nc, self.amp, loss_fn)
                if cfg.val and stopped != "cancelled" else ({}, {})
            )
            fitness = 0.1 * metrics.get("mAP50", 0) + 0.9 * metrics.get("mAP50-95", 0)
            if metrics:
                final_metrics = metrics
                logger.info("val epoch %d: %s", epoch + 1,
                            " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            improved = fitness > best_fitness
            best_fitness = max(best_fitness, fitness)
            if improved:
                best_epoch = epoch + 1
            log.append(
                epoch_row(epoch + 1, elapsed_before + time.time() - t_start,
                          dict(zip(LOSSES, mean_items.tolist())),
                          metrics, val_losses, [g["lr"] for g in optimizer.param_groups]),
                step=global_step, fitness=fitness if metrics else None,
                best_epoch=best_epoch, total_epochs=cfg.epochs,
            )
            self._save(last, model, ema, optimizer, scaler, epoch, best_fitness, global_step,
                       best_epoch)
            if improved:
                self._save_weights(self.weights_dir / "best.pt", ema.ema)
            if stopped:
                logger.warning("stopping early (%s) at epoch %d step %d", stopped, epoch + 1,
                               global_step)
                break

        if not (self.weights_dir / "best.pt").exists():
            self._save_weights(self.weights_dir / "best.pt", ema.ema)
        exported = self._export()
        try:
            plot = log.plot()
        except Exception as exc:  # a broken plotting backend must not fail a finished run
            logger.warning("results.png not written: %s", exc)
            plot = None
        records = [log.csv, log.jsonl, *([plot] if plot else [])]
        peak = torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else 0.0
        summary = {
            "model": f"{self.model.family}{self.model.scale}",
            "epochs_completed": epoch + 1 - start_epoch if stopped != "cancelled" else epoch - start_epoch,
            "global_step": global_step,
            "batch": batch,
            "stopped": stopped,
            "best_fitness": best_fitness,
            "best_epoch": best_epoch,
            "total_epochs": cfg.epochs,
            "final_metrics": final_metrics,
            "seconds": round(time.time() - t_start, 1),
            "peak_mem_gib": round(peak, 2),
            "artifacts": [str(p) for p in exported],
            "records": [str(p) for p in records],
        }
        (self.save_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def _meta(self) -> dict[str, Any]:
        m = self.model
        return {"family": m.family, "scale": m.scale, "nc": m.nc, "names": self.data.names,
                "imgsz": self.cfg.imgsz, "stride": m.stride.tolist(),
                "train_config": asdict(self.cfg)}

    def _save(self, path, model, ema, optimizer, scaler, epoch, best_fitness, global_step,
              best_epoch):
        state = {
            **self._meta(),
            "model": model.state_dict(),
            "state_dict": ema.ema.state_dict(),
            "ema_updates": ema.updates,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_fitness": best_fitness,
            "global_step": global_step,
            "best_epoch": best_epoch,
        }
        tmp = path.with_suffix(".tmp")
        torch.save(state, tmp)
        tmp.replace(path)

    def _save_weights(self, path: Path, model: nn.Module) -> None:
        tmp = path.with_suffix(".tmp")
        torch.save({**self._meta(), "state_dict": model.state_dict()}, tmp)
        tmp.replace(path)

    def _export(self) -> list[Path]:
        best = torch.load(self.weights_dir / "best.pt", map_location="cpu", weights_only=True)
        meta = {k: v for k, v in best.items() if k not in {"state_dict", "train_config"}}
        config = self.save_dir / "config.json"
        config.write_text(json.dumps({**meta, "format": "distill-run-yolo"}, indent=2),
                          encoding="utf-8")
        artifacts = [self.weights_dir / "best.pt", self.weights_dir / "last.pt", config]
        try:
            from safetensors.torch import save_file
        except ImportError:
            logger.warning("safetensors not installed; skipping best.safetensors")
            return artifacts
        target = self.save_dir / "best.safetensors"
        save_file({k: v.contiguous() for k, v in best["state_dict"].items()}, str(target),
                  metadata={"family": meta["family"], "scale": meta["scale"], "nc": str(meta["nc"])})
        return [target, *artifacts]
