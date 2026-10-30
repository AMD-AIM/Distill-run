"""Detection datasets: discovery, label loading, augmentation and batching.

Two label formats share one in-memory index of ``(image, labels)`` pairs, where
``labels`` is ``(n, 5)`` float32 ``[cls, cx, cy, w, h]`` normalised to the image:

* YOLO txt: a ``data.yaml`` (``path``/``train``/``val``/``names``) or a plain
  ``images/<split>`` + ``labels/<split>`` tree.
* COCO json: ``annotations/instances_<split>.json`` next to ``images/<split>``
  (or ``<split>/``), or split entries in ``data.yaml`` pointing at ``.json`` files.

Archives (zip/tar) are unpacked once next to the downloaded file.
"""

from __future__ import annotations

import json
import logging
import math
import random
import shutil
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)

IMG_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.xz", ".tar.bz2")

COCO80 = (
    "person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant "
    "stop_sign parking_meter bench bird cat dog horse sheep cow elephant bear zebra giraffe "
    "backpack umbrella handbag tie suitcase frisbee skis snowboard sports_ball kite "
    "baseball_bat baseball_glove skateboard surfboard tennis_racket bottle wine_glass cup fork "
    "knife spoon bowl banana apple sandwich orange broccoli carrot hot_dog pizza donut cake "
    "chair couch potted_plant bed dining_table toilet tv laptop mouse remote keyboard "
    "cell_phone microwave oven toaster sink refrigerator book clock vase scissors teddy_bear "
    "hair_drier toothbrush"
).split()


class DataFormatError(ValueError):
    """Raised for unusable dataset layouts; the engine maps it to DataError."""


@dataclass
class DetectionData:
    root: Path
    names: dict[int, str]
    train: list[tuple[Path, np.ndarray]]
    val: list[tuple[Path, np.ndarray]]
    format: str
    notes: list[str] = field(default_factory=list)

    @property
    def nc(self) -> int:
        return len(self.names)

    def summary(self) -> dict[str, Any]:
        def count(items):
            return int(sum(len(lbl) for _, lbl in items))

        return {
            "root": str(self.root),
            "format": self.format,
            "nc": self.nc,
            "train_images": len(self.train),
            "train_boxes": count(self.train),
            "val_images": len(self.val),
            "val_boxes": count(self.val),
        }


# --- discovery -----------------------------------------------------------------


def is_archive(path: Path) -> bool:
    return path.is_file() and path.name.lower().endswith(ARCHIVE_SUFFIXES)


def extract(path: Path) -> Path:
    """Unpack ``path`` into ``<path>.extracted`` once; return the content root."""
    target = path.with_name(path.name + ".extracted")
    marker = target / ".complete"
    if not marker.is_file():
        staging = path.with_name(path.name + ".extracting")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        logger.info("extracting %s", path)
        if path.name.lower().endswith(".zip"):
            with zipfile.ZipFile(path) as z:
                for member in z.namelist():
                    _check_member(staging, member)
                z.extractall(staging)
        else:
            with tarfile.open(path) as t:
                for member in t.getmembers():
                    _check_member(staging, member.name)
                    if member.issym() or member.islnk():
                        raise DataFormatError(f"{path}: links are not allowed ({member.name})")
                t.extractall(staging)
        shutil.rmtree(target, ignore_errors=True)
        staging.rename(target)
        marker.write_text(str(path), encoding="utf-8")
    entries = [p for p in target.iterdir() if not p.name.startswith((".", "__MACOSX"))]
    return entries[0] if len(entries) == 1 and entries[0].is_dir() else target


def _check_member(root: Path, name: str) -> None:
    resolved = (root / name).resolve()
    if root.resolve() not in resolved.parents and resolved != root.resolve():
        raise DataFormatError(f"archive member escapes the target directory: {name}")


def load_detection_data(
    location: Path, *, names: Any = None, val_fraction: float = 0.0, seed: int = 0
) -> DetectionData:
    path = Path(location)
    if is_archive(path):
        path = extract(path)
    if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}:
        data = _from_yaml(path)
    elif path.is_file() and path.suffix.lower() == ".json":
        data = _from_coco_files(path.parent, {"train": path}, None)
    elif path.is_dir():
        data = _from_directory(path)
    else:
        raise DataFormatError(f"unsupported dataset location: {path}")
    if names is not None:
        data.names = _normalise_names(names)
    if not data.train:
        raise DataFormatError(f"{data.root}: no training images found")
    if not data.val:
        data.val = _split_off_val(data, val_fraction, seed)
    _check_class_ids(data)
    return data


def _split_off_val(data: DetectionData, fraction: float, seed: int) -> list:
    if fraction <= 0:
        data.notes.append("no validation split; validating on the training images")
        return list(data.train)
    items = list(data.train)
    random.Random(seed).shuffle(items)
    k = max(1, int(len(items) * fraction))
    data.train = items[k:] or items
    data.notes.append(f"held out {k} training images for validation")
    return items[:k]


def _check_class_ids(data: DetectionData) -> None:
    top = max((int(lbl[:, 0].max()) for _, lbl in data.train + data.val if len(lbl)), default=-1)
    if top >= data.nc:
        raise DataFormatError(
            f"{data.root}: label class id {top} out of range for {data.nc} class names"
        )


def _normalise_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in sorted(names.items(), key=lambda kv: int(kv[0]))}
    if isinstance(names, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names)}
    if isinstance(names, int):
        return {i: f"class{i}" for i in range(names)}
    raise DataFormatError(f"names must be a list or mapping, got {type(names).__name__}")


def _from_yaml(file: Path) -> DetectionData:
    raw = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict) or "train" not in raw:
        raise DataFormatError(f"{file}: data yaml needs at least a 'train' entry")
    root = Path(raw.get("path") or file.parent)
    if not root.is_absolute():
        root = (file.parent / root).resolve()
        if not root.exists() and (file.parent / root.name).exists():
            root = file.parent / root.name
    names = raw.get("names")
    if names is None and raw.get("nc"):
        names = int(raw["nc"])
    splits = {k: raw[k] for k in ("train", "val") if raw.get(k)}
    if any(_is_json_entry(v) for v in splits.values()):
        return _from_coco_files(root, splits, names)
    if names is None:
        raise DataFormatError(f"{file}: 'names' (or 'nc') is required")
    named = _normalise_names(names)
    return DetectionData(
        root=root,
        names=named,
        train=_yolo_split(root, splits["train"]),
        val=_yolo_split(root, splits["val"]) if "val" in splits else [],
        format="yolo",
    )


def _is_json_entry(value: Any) -> bool:
    if isinstance(value, dict):
        return "json" in value
    return isinstance(value, str) and value.lower().endswith(".json")


def _from_directory(root: Path) -> DetectionData:
    for name in ("data.yaml", "dataset.yaml", f"{root.name}.yaml"):
        if (root / name).is_file():
            return _from_yaml(root / name)
    yamls = [p for p in root.glob("*.y*ml") if "train" in p.read_text(encoding="utf-8")]
    if len(yamls) == 1:
        return _from_yaml(yamls[0])

    jsons = sorted((root / "annotations").glob("*.json")) if (root / "annotations").is_dir() else []
    if jsons:
        splits: dict[str, Path] = {}
        for j in jsons:
            stem = j.stem.lower()
            if "train" in stem and "train" not in splits:
                splits["train"] = j
            elif "val" in stem and "val" not in splits:
                splits["val"] = j
        if "train" not in splits:
            splits["train"] = jsons[0]
        return _from_coco_files(root, splits, None)

    images = root / "images"
    labels = root / "labels"
    if images.is_dir() and labels.is_dir():
        split_dirs = sorted(p.name for p in images.iterdir() if p.is_dir())
        train = next((s for s in split_dirs if s.startswith("train")), None)
        val = next((s for s in split_dirs if s.startswith("val")), None)
        if train is None and not split_dirs:
            train_items = _yolo_split(root, "images")
        else:
            train_items = _yolo_split(root, f"images/{train or split_dirs[0]}")
        val_items = _yolo_split(root, f"images/{val}") if val else []
        names_file = next((p for p in (root / "classes.txt", root / "names.txt") if p.is_file()), None)
        if names_file:
            names = _normalise_names(names_file.read_text(encoding="utf-8").split())
        elif root.name.lower().startswith("coco"):
            names = _normalise_names(list(COCO80))
        else:
            top = max((int(lbl[:, 0].max()) for _, lbl in train_items + val_items if len(lbl)), default=0)
            names = _normalise_names(top + 1)
        return DetectionData(root=root, names=names, train=train_items, val=val_items, format="yolo")
    raise DataFormatError(
        f"{root}: expected data.yaml, annotations/*.json (COCO) or images/ + labels/ (YOLO)"
    )


def _image_files(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if path.is_file() and path.suffix.lower() == ".txt":
        files = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                p = Path(line)
                files.append(p if p.is_absolute() else (path.parent / p).resolve())
        return files
    if path.is_file() and path.suffix.lower() in IMG_EXTS:
        return [path]
    raise DataFormatError(f"image source not found: {path}")


def _label_path(image: Path) -> Path:
    parts = list(image.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def _yolo_split(root: Path, entry: Any) -> list[tuple[Path, np.ndarray]]:
    entries = entry if isinstance(entry, list) else [entry]
    items: list[tuple[Path, np.ndarray]] = []
    for value in entries:
        p = Path(value)
        for image in _image_files(p if p.is_absolute() else root / p):
            items.append((image, _read_yolo_labels(_label_path(image))))
    return items


def _read_yolo_labels(path: Path) -> np.ndarray:
    if not path.is_file():
        return np.zeros((0, 5), np.float32)
    rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r[:5] for r in rows if len(r) >= 5]
    if not rows:
        return np.zeros((0, 5), np.float32)
    labels = np.array(rows, dtype=np.float32)
    labels[:, 1:] = labels[:, 1:].clip(0, 1)
    return labels[(labels[:, 3] > 0) & (labels[:, 4] > 0)]


def _from_coco_files(root: Path, splits: dict[str, Any], names: Any) -> DetectionData:
    categories: dict[int, str] | None = None
    result: dict[str, list] = {}
    for split, entry in splits.items():
        if isinstance(entry, dict):
            json_path, image_dir = Path(entry["json"]), Path(entry.get("images", ""))
        else:
            json_path, image_dir = Path(entry), None
        if not json_path.is_absolute():
            json_path = root / json_path
        raw = json.loads(json_path.read_text(encoding="utf-8"))
        cats = {int(c["id"]): str(c["name"]) for c in raw.get("categories", [])}
        if categories is None:
            categories = cats
        elif cats and cats != categories:
            raise DataFormatError(f"{json_path}: categories differ between splits")
        if image_dir is None or not str(image_dir):
            image_dir = _coco_image_dir(root, json_path)
        elif not image_dir.is_absolute():
            image_dir = root / image_dir
        result[split] = _coco_items(raw, image_dir, sorted(categories))
    named = _normalise_names(names) if names is not None else {
        i: categories[cid] for i, cid in enumerate(sorted(categories or {}))
    }
    return DetectionData(
        root=root, names=named, train=result.get("train", []), val=result.get("val", []),
        format="coco",
    )


def _coco_image_dir(root: Path, json_path: Path) -> Path:
    stem = json_path.stem
    split = stem.split("_", 1)[1] if stem.startswith("instances_") else stem
    for candidate in (root / "images" / split, root / split, root / "images", json_path.parent):
        if candidate.is_dir():
            return candidate
    raise DataFormatError(f"{json_path}: cannot find image directory for split '{split}'")


def _coco_items(raw: dict, image_dir: Path, category_ids: list[int]) -> list:
    index = {cid: i for i, cid in enumerate(category_ids)}
    boxes: dict[int, list] = {}
    for ann in raw.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        boxes.setdefault(int(ann["image_id"]), []).append(ann)
    items = []
    for img in raw.get("images", []):
        w, h = float(img["width"]), float(img["height"])
        rows = []
        for ann in boxes.get(int(img["id"]), []):
            x, y, bw, bh = (float(v) for v in ann["bbox"])
            x1, y1 = max(x, 0.0), max(y, 0.0)
            x2, y2 = min(x + bw, w), min(y + bh, h)
            if x2 <= x1 or y2 <= y1:
                continue
            rows.append([index[int(ann["category_id"])], (x1 + x2) / 2 / w, (y1 + y2) / 2 / h,
                         (x2 - x1) / w, (y2 - y1) / h])
        labels = np.array(rows, np.float32) if rows else np.zeros((0, 5), np.float32)
        items.append((image_dir / img["file_name"], labels))
    return items


# --- augmentation -----------------------------------------------------------------


def _cv2():
    import cv2

    cv2.setNumThreads(0)
    return cv2


def letterbox(img: np.ndarray, size: int, color=(114, 114, 114)):
    cv2 = _cv2()
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    if (nh, nw) != (h, w):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), color, dtype=np.uint8)
    out[top : top + nh, left : left + nw] = img
    return out, r, (left, top)


def augment_hsv(img: np.ndarray, hgain: float, sgain: float, vgain: float) -> None:
    if not (hgain or sgain or vgain):
        return
    cv2 = _cv2()
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(np.uint8)
    lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
    lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)
    merged = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    cv2.cvtColor(merged, cv2.COLOR_HSV2BGR, dst=img)


def random_affine(img, boxes, cls, size, border, degrees, translate, scale, shear):
    """Rotate/scale/shear/translate ``img`` into ``size`` x ``size``; boxes are pixel xyxy."""
    cv2 = _cv2()
    h, w = img.shape[:2]
    c = np.eye(3)
    c[0, 2], c[1, 2] = -w / 2, -h / 2
    r = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    r[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
    sh = np.eye(3)
    sh[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    sh[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    t = np.eye(3)
    out = size
    t[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * out
    t[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * out
    m = t @ sh @ r @ c
    img = cv2.warpAffine(img, m[:2], dsize=(out, out), borderValue=(114, 114, 114))
    if len(boxes):
        n = len(boxes)
        xy = np.ones((n * 4, 3))
        xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
        xy = (xy @ m.T)[:, :2].reshape(n, 8)
        x, y = xy[:, [0, 2, 4, 6]], xy[:, [1, 3, 5, 7]]
        new = np.stack((x.min(1), y.min(1), x.max(1), y.max(1)), 1)
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, out)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, out)
        keep = _box_candidates(boxes.T * s, new.T)
        boxes, cls = new[keep], cls[keep]
    return img, boxes, cls


def _box_candidates(b1, b2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):
    w1, h1 = b1[2] - b1[0], b1[3] - b1[1]
    w2, h2 = b2[2] - b2[0], b2[3] - b2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)


# --- torch dataset -----------------------------------------------------------------


class DetectionDataset:
    """Map-style dataset returning ``(CHW uint8 RGB tensor, (n, 5) labels, path)``.

    Labels come out normalised ``[cls, cx, cy, w, h]`` relative to the
    ``imgsz`` x ``imgsz`` output. Set ``mosaic = False`` to stop mosaic, e.g.
    for the final ``close_mosaic`` epochs.
    """

    def __init__(self, items, imgsz: int, *, augment: bool, hyp: dict[str, Any] | None = None):
        self.items = items
        self.imgsz = imgsz
        self.augment = augment
        self.hyp = dict(hyp or {})
        self.mosaic = augment and self.hyp.get("mosaic", 1.0) > 0

    def __len__(self) -> int:
        return len(self.items)

    def _load(self, index: int):
        cv2 = _cv2()
        path, labels = self.items[index]
        img = cv2.imread(str(path))
        if img is None:
            raise DataFormatError(f"cannot read image {path}")
        h0, w0 = img.shape[:2]
        r = self.imgsz / max(h0, w0)
        if r != 1:
            img = cv2.resize(img, (max(1, round(w0 * r)), max(1, round(h0 * r))),
                             interpolation=cv2.INTER_LINEAR if r > 1 or self.augment else cv2.INTER_AREA)
        return img, labels

    @staticmethod
    def _to_xyxy(labels: np.ndarray, w: float, h: float, padw: float, padh: float):
        boxes = labels[:, 1:].copy()
        xyxy = np.empty_like(boxes)
        xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * w + padw
        xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * h + padh
        xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * w + padw
        xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * h + padh
        return xyxy, labels[:, 0].copy()

    def _mosaic(self, index: int):
        s = self.imgsz
        yc, xc = (int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2))
        canvas = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        all_boxes, all_cls = [], []
        indices = [index] + random.choices(range(len(self.items)), k=3)
        for i, idx in enumerate(indices):
            img, labels = self._load(idx)
            h, w = img.shape[:2]
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            canvas[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            if len(labels):
                boxes, cls = self._to_xyxy(labels, w, h, x1a - x1b, y1a - y1b)
                all_boxes.append(boxes)
                all_cls.append(cls)
        boxes = np.concatenate(all_boxes) if all_boxes else np.zeros((0, 4), np.float32)
        cls = np.concatenate(all_cls) if all_cls else np.zeros((0,), np.float32)
        boxes = boxes.clip(0, 2 * s)
        return canvas, boxes, cls

    def __getitem__(self, index: int):
        import torch

        s = self.imgsz
        hyp = self.hyp
        if self.augment and self.mosaic and random.random() < hyp.get("mosaic", 1.0):
            img, boxes, cls = self._mosaic(index)
            img, boxes, cls = random_affine(
                img, boxes, cls, s, -s // 2, hyp.get("degrees", 0.0), hyp.get("translate", 0.1),
                hyp.get("scale", 0.5), hyp.get("shear", 0.0),
            )
        else:
            raw, labels = self._load(index)
            img, ratio, (padw, padh) = letterbox(raw, s)
            h, w = raw.shape[:2]
            boxes, cls = (
                self._to_xyxy(labels, w, h, padw, padh) if len(labels)
                else (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32))
            )
            if self.augment:
                img, boxes, cls = random_affine(
                    img, boxes, cls, s, 0, hyp.get("degrees", 0.0), hyp.get("translate", 0.1),
                    hyp.get("scale", 0.5), hyp.get("shear", 0.0),
                )
        if self.augment:
            augment_hsv(img, hyp.get("hsv_h", 0.015), hyp.get("hsv_s", 0.7), hyp.get("hsv_v", 0.4))
            if random.random() < hyp.get("fliplr", 0.5):
                img = np.ascontiguousarray(img[:, ::-1])
                if len(boxes):
                    boxes = boxes.copy()
                    boxes[:, [0, 2]] = s - boxes[:, [2, 0]]
        out = np.zeros((len(boxes), 5), np.float32)
        if len(boxes):
            out[:, 0] = cls
            out[:, 1] = (boxes[:, 0] + boxes[:, 2]) / 2 / s
            out[:, 2] = (boxes[:, 1] + boxes[:, 3]) / 2 / s
            out[:, 3] = (boxes[:, 2] - boxes[:, 0]) / s
            out[:, 4] = (boxes[:, 3] - boxes[:, 1]) / s
        tensor = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1)))
        return tensor, torch.from_numpy(out), str(self.items[index][0])


def collate(batch):
    import torch

    images, labels, paths = zip(*batch)
    rows = [torch.cat((torch.full((len(lbl), 1), i, dtype=lbl.dtype), lbl), 1)
            for i, lbl in enumerate(labels)]
    return torch.stack(images), torch.cat(rows), list(paths)
