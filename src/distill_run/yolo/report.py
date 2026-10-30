"""Per-epoch training records: ``results.csv``, ``metrics.jsonl`` and ``results.png``.

``results.csv`` uses the Ultralytics column names, so its plotting tools and the
Ultralytics Platform charts read it unchanged. ``metrics.jsonl`` carries the same
values plus the step and fitness, one JSON object per epoch, for streaming.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger("distill_run.yolo")

LOSSES = ("box_loss", "cls_loss", "dfl_loss")
METRICS = ("precision", "recall", "mAP50", "mAP50-95")
COLUMNS = (
    "epoch", "time",
    *(f"train/{k}" for k in LOSSES),
    *(f"metrics/{k}(B)" for k in METRICS),
    *(f"val/{k}" for k in LOSSES),
    "lr/pg0", "lr/pg1", "lr/pg2",
)
PLOTS = (
    "train/box_loss", "train/cls_loss", "train/dfl_loss", "metrics/precision(B)",
    "metrics/recall(B)", "val/box_loss", "val/cls_loss", "val/dfl_loss",
    "metrics/mAP50(B)", "metrics/mAP50-95(B)",
)


def epoch_row(epoch: int, seconds: float, train: dict[str, float], metrics: dict[str, float],
              val: dict[str, float], lrs: list[float]) -> dict[str, Any]:
    row: dict[str, Any] = {"epoch": epoch, "time": round(seconds, 3)}
    row.update({f"train/{k}": train.get(k) for k in LOSSES})
    row.update({f"metrics/{k}(B)": metrics.get(k) for k in METRICS})
    row.update({f"val/{k}": val.get(k) for k in LOSSES})
    row.update({f"lr/pg{i}": lrs[i] if i < len(lrs) else None for i in range(3)})
    return row


class TrainingLog:
    def __init__(self, save_dir: Path, start_epoch: int):
        self.csv = save_dir / "results.csv"
        self.jsonl = save_dir / "metrics.jsonl"
        self.png = save_dir / "results.png"
        self._truncate(start_epoch)

    def _truncate(self, start_epoch: int) -> None:
        """Drop rows a resumed run will write again (or every row on a fresh run)."""
        rows = self.rows() if start_epoch else []
        rows = [r for r in rows if int(float(r["epoch"])) <= start_epoch]
        with self.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows({k: r.get(k, "") for k in COLUMNS} for r in rows)
        records = []
        if start_epoch and self.jsonl.is_file():
            for line in self.jsonl.read_text(encoding="utf-8").splitlines():
                if line.strip() and json.loads(line).get("epoch", 0) <= start_epoch:
                    records.append(line)
        self.jsonl.write_text("".join(f"{r}\n" for r in records), encoding="utf-8")

    def append(self, row: dict[str, Any], **extra: Any) -> None:
        with self.csv.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writerow(
                {k: "" if _finite(row.get(k)) is None else _finite(row[k]) for k in COLUMNS}
            )
        with self.jsonl.open("a", encoding="utf-8") as f:
            record = {k: _finite(v) for k, v in {**row, **extra}.items()}
            f.write(json.dumps(record, allow_nan=False) + "\n")

    def elapsed(self) -> float:
        """Training seconds already recorded, so a resumed run keeps ``time`` cumulative."""
        rows = self.rows()
        return float(rows[-1]["time"] or 0) if rows else 0.0

    def rows(self) -> list[dict[str, str]]:
        if not self.csv.is_file():
            return []
        with self.csv.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def plot(self) -> Path | None:
        rows = self.rows()
        if not rows:
            return None
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.info("matplotlib not installed; skipping results.png")
            return None
        fig, axes = plt.subplots(2, 5, figsize=(15, 6), tight_layout=True)
        for ax, key in zip(axes.ravel(), PLOTS):
            points = [(float(r["epoch"]), float(r[key])) for r in rows if r.get(key) not in ("", None)]
            if points:
                ax.plot(*zip(*points), marker=".", linewidth=1.5)
            ax.set_title(key, fontsize=10)
        fig.savefig(self.png, dpi=120)
        plt.close(fig)
        return self.png


def _finite(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
