#!/usr/bin/env python3
"""In-container smoke for the distill-run image.

Runs, in order: the shell tools the RC wrapper needs, ``distill-run doctor`` for all
five engines, ROCm Torch, the GPU (unless ``--device cpu``) and an importable
distill-run CLI in every venv, a
2-epoch ``--engine ultralytics`` run on coco8 served as Platform NDJSON over local
HTTP, and a 2-step ``--engine yolo`` run on the coco8 directory. Each check prints
``IMAGE_SMOKE <name> exit=<n> seconds=<s>``; the last line is
``IMAGE_SMOKE result=PASS`` or ``IMAGE_SMOKE result=FAIL``.

    python scripts/image_smoke.py --coco8 coco8.zip --device 0 --out /tmp/image-smoke
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import threading
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ENGINES = ("swift", "trl", "easydistill", "yolo", "ultralytics")
VENVS = ("core", *ENGINES)
SHELL_TOOLS = ("setsid", "find", "head")
SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
ULTRALYTICS_EPOCHS = 2


@dataclass
class Check:
    name: str
    argv: list[str]
    expect: Callable[[], bool] | None = None


def coco_to_ndjson(root: Path, base_url: str, out: Path) -> Path:
    """Write a YOLO txt tree (``images/<split>``, ``labels/<split>``) as Platform NDJSON."""
    from distill_run.yolo.data import COCO80

    records: list[dict] = [
        {
            "type": "dataset",
            "task": "detect",
            "class_names": {str(i): name for i, name in enumerate(COCO80)},
        }
    ]
    for split in SPLITS:
        images = root / "images" / split
        if not images.is_dir():
            continue
        for image in sorted(p for p in images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
            label = root / "labels" / split / f"{image.stem}.txt"
            boxes = []
            if label.is_file():
                for row in label.read_text(encoding="utf-8").splitlines():
                    fields = row.split()
                    if len(fields) == 5:
                        boxes.append([int(fields[0]), *(float(v) for v in fields[1:])])
            records.append(
                {
                    "type": "image",
                    "file": image.name,
                    "split": split,
                    "url": f"{base_url}/images/{split}/{image.name}",
                    "annotations": {"boxes": boxes},
                }
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
    return out


def _torch_code(venv: str, device: str) -> str:
    lines = ["import torch", "import distill_run.cli"]
    if venv in ("yolo", "ultralytics"):
        lines.append("import cv2")
    lines += [
        "assert torch.version.hip, f'not a ROCm build: {torch.__version__}'",
        "ok = torch.cuda.is_available()",
        "name = torch.cuda.get_device_name(0) if ok else '-'",
        f"print({venv!r}, torch.__version__, torch.version.hip, ok, name)",
    ]
    if device != "cpu":
        lines += [
            "assert torch.cuda.is_available(), 'no GPU visible'",
            "assert torch.ones(4, device='cuda').sum().item() == 4",
        ]
    return "\n".join(lines)


def _csv_rows(path: Path) -> int:
    if not path.is_file():
        return -1
    return max(len([row for row in path.read_text(encoding="utf-8").splitlines() if row]) - 1, 0)


def build_checks(
    coco8: Path, base_url: str, device: str, out: Path, envs_root: Path
) -> list[Check]:
    device_args = [] if device == "cpu" else ["--device", device]
    work = ["--work-dir", str(out / "work")]
    doctor = ["distill-run", "doctor"]
    for engine in ENGINES:
        doctor += ["--engine", engine]
    checks = [
        Check(
            "shell-tools",
            ["sh", "-c", f'for t in {" ".join(SHELL_TOOLS)}; do command -v "$t" || exit 1; done'],
        ),
        Check("doctor", doctor),
    ]
    for venv in VENVS:
        python = envs_root / venv / "bin" / "python"
        checks.append(Check(f"torch:{venv}", [str(python), "-c", _torch_code(venv, device)]))

    ultralytics_out = out / "ultralytics"
    checks.append(
        Check(
            "ultralytics",
            [
                "distill-run",
                "--engine",
                "ultralytics",
                "--model",
                "yolo26n.yaml",
                "--data",
                f"{base_url}.ndjson",
                "--train-args",
                f"epochs={ULTRALYTICS_EPOCHS} imgsz=320 batch=4 workers=0 amp=False",
                *device_args,
                "--output",
                str(ultralytics_out),
                *work,
            ],
            expect=lambda: _csv_rows(ultralytics_out / "results.csv") == ULTRALYTICS_EPOCHS,
        )
    )
    yolo_out = out / "yolo"
    checks.append(
        Check(
            "yolo",
            [
                "distill-run",
                "--engine",
                "yolo",
                "--model",
                "yolo26n",
                "--data",
                str(coco8),
                "--epochs",
                "1",
                "--max-steps",
                "2",
                "--imgsz",
                "320",
                "--batch",
                "4",
                "--workers",
                "0",
                "--no-amp",
                *device_args,
                "--output",
                str(yolo_out),
                *work,
            ],
            expect=lambda: (yolo_out / "weights" / "last.pt").is_file(),
        )
    )
    return checks


def run_checks(checks: list[Check]) -> int:
    failed = False
    for check in checks:
        print("+", " ".join(check.argv), flush=True)
        started = time.monotonic()
        try:
            code = subprocess.run(check.argv).returncode  # noqa: S603
        except OSError as exc:
            print(f"IMAGE_SMOKE {check.name} {exc}", flush=True)
            code = 127
        seconds = f"seconds={time.monotonic() - started:.1f}"
        if code == 0 and check.expect is not None and not check.expect():
            print(f"IMAGE_SMOKE {check.name} exit=0 artifact=missing {seconds}", flush=True)
            failed = True
            continue
        print(f"IMAGE_SMOKE {check.name} exit={code} {seconds}", flush=True)
        failed = failed or code != 0
    print(f"IMAGE_SMOKE result={'FAIL' if failed else 'PASS'}", flush=True)
    return 1 if failed else 0


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def _unpack(source: Path, data: Path) -> Path:
    if source.is_dir():
        target = data / "coco8"
        if not target.exists():
            shutil.copytree(source, target)
        return target
    with zipfile.ZipFile(source) as archive:
        archive.extractall(data)
    return data / "coco8"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--coco8", type=Path, required=True, help="coco8.zip or its directory")
    parser.add_argument("--device", default="0", help="GPU index, or cpu")
    parser.add_argument("--out", type=Path, default=Path("/tmp/image-smoke"))
    args = parser.parse_args(argv)

    data = args.out / "data"
    data.mkdir(parents=True, exist_ok=True)
    coco8 = _unpack(args.coco8, data)
    if not (coco8 / "images").is_dir():
        raise SystemExit(f"{args.coco8}: no coco8/images inside")

    handler = partial(_QuietHandler, directory=str(data))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_address[1]}/coco8"
        coco_to_ndjson(coco8, base_url, data / "coco8.ndjson")
        envs_root = Path(os.environ.get("DISTILL_RUN_ENVS_ROOT", "/opt/distill-run/.venvs"))
        return run_checks(build_checks(coco8, base_url, args.device, args.out, envs_root))
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
