"""Built-in YOLO engine: naming, parameters, dataset discovery, and (with torch) model/loss."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from distill_run import params as P
from distill_run.config import merge_sources

np = pytest.importorskip("numpy")

from distill_run.yolo.data import DataFormatError, load_detection_data  # noqa: E402
from distill_run.yolo.model import parse_name, weights_filename  # noqa: E402


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("yolo26n", ("yolo26", "n")),
        ("YOLO11x", ("yolo11", "x")),
        ("yolov8m", ("yolov8", "m")),
        ("yolov5su", ("yolov5", "s")),
        ("yolo5l", ("yolov5", "l")),
        ("yolo99n", None),
        ("yolo11q", None),
    ],
)
def test_parse_name(name, expected):
    assert parse_name(name) == expected


def test_official_weight_names():
    assert weights_filename("yolov5", "n") == "yolov5nu.pt"
    assert weights_filename("yolo26", "x") == "yolo26x.pt"


def test_yolo_params_route_into_the_yolo_config_section():
    accepted = {p.dest for p in P.params_for("yolo")}
    assert {"student", "dataset", "epochs", "imgsz", "batch", "device", "resume"} <= accepted
    assert "teacher" not in accepted
    resolved = merge_sources("yolo", {"epochs": 3}, {"YOLO_BATCH": "-1"})
    assert resolved.values["epochs"] == 3
    assert resolved.values["batch"] == -1
    with pytest.raises(Exception, match="does not accept"):
        merge_sources("yolo", {"teacher": "x"}, {})


def _touch_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_yolo_tree_with_data_yaml(tmp_path: Path):
    for split in ("train", "val"):
        _touch_image(tmp_path / "images" / split / "a.jpg")
        label = tmp_path / "labels" / split / "a.txt"
        label.parent.mkdir(parents=True)
        label.write_text("1 0.5 0.5 0.2 0.2\n0 0.1 0.1 0.05 0.05\n", encoding="utf-8")
    (tmp_path / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames: [cat, dog]\n", encoding="utf-8"
    )
    data = load_detection_data(tmp_path)
    assert data.format == "yolo"
    assert data.names == {0: "cat", 1: "dog"}
    assert len(data.train) == 1 and data.train[0][1].shape == (2, 5)


def test_class_id_out_of_range_is_rejected(tmp_path: Path):
    _touch_image(tmp_path / "images" / "train" / "a.jpg")
    (tmp_path / "labels" / "train").mkdir(parents=True)
    (tmp_path / "labels" / "train" / "a.txt").write_text("5 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    (tmp_path / "data.yaml").write_text("train: images/train\nnames: [a, b]\n", encoding="utf-8")
    with pytest.raises(DataFormatError, match="out of range"):
        load_detection_data(tmp_path / "data.yaml")


def test_coco_json_layout(tmp_path: Path):
    _touch_image(tmp_path / "images" / "train2017" / "x.jpg")
    ann = {
        "images": [{"id": 7, "file_name": "x.jpg", "width": 100, "height": 50}],
        "annotations": [
            {"id": 1, "image_id": 7, "category_id": 18, "bbox": [10, 10, 20, 10], "iscrowd": 0},
            {"id": 2, "image_id": 7, "category_id": 3, "bbox": [0, 0, 5, 5], "iscrowd": 1},
        ],
        "categories": [{"id": 3, "name": "car"}, {"id": 18, "name": "dog"}],
    }
    (tmp_path / "annotations").mkdir()
    (tmp_path / "annotations" / "instances_train2017.json").write_text(json.dumps(ann))
    data = load_detection_data(tmp_path)
    assert data.format == "coco"
    assert data.names == {0: "car", 1: "dog"}
    labels = data.train[0][1]
    assert labels.shape == (1, 5)
    assert labels[0].tolist() == pytest.approx([1, 0.2, 0.3, 0.2, 0.2])
    assert data.val == data.train  # no val split: validate on train, with a note
    assert data.notes


def test_archive_is_extracted_once(tmp_path: Path):
    import zipfile

    archive = tmp_path / "set.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("set/images/train/a.jpg", b"")
        z.writestr("set/labels/train/a.txt", "0 0.5 0.5 0.5 0.5\n")
        z.writestr("set/classes.txt", "thing\n")
    data = load_detection_data(archive)
    assert data.names == {0: "thing"}
    assert (tmp_path / "set.zip.extracted" / ".complete").is_file()


torch = pytest.importorskip("torch")


@pytest.mark.parametrize(("family", "params"), [("yolov8", 3157200), ("yolo11", 2624080),
                                                ("yolo26", 2572280)])
def test_nano_parameter_counts_match_published(family, params):
    from distill_run.yolo.model import DetectionModel

    model = DetectionModel(family, "n", 80)
    assert sum(p.numel() for p in model.parameters()) == params
    assert model.stride.tolist() == [8.0, 16.0, 32.0]


@pytest.mark.parametrize("family", ["yolov5", "yolo26"])
def test_loss_is_finite_and_backpropagates(family):
    from distill_run.yolo.loss import build_loss
    from distill_run.yolo.model import DetectionModel

    model = DetectionModel(family, "n", 3).train()
    loss_fn = build_loss(model, {"box": 7.5, "cls": 0.5, "dfl": 1.5}, epochs=2)
    labels = torch.tensor([[0, 1, 0.5, 0.5, 0.3, 0.3], [1, 2, 0.2, 0.3, 0.1, 0.2]])
    loss, items = loss_fn(model(torch.rand(2, 3, 128, 128)), labels)
    assert torch.isfinite(loss) and items.shape == (3,)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_own_checkpoint_round_trip(tmp_path: Path):
    from distill_run.yolo.model import DetectionModel
    from distill_run.yolo.weights import apply_state_dict, load_checkpoint

    src = DetectionModel("yolov8", "n", 4)
    torch.save({"family": "yolov8", "scale": "n", "nc": 4, "state_dict": src.state_dict()},
               tmp_path / "w.pt")
    ckpt = load_checkpoint(tmp_path / "w.pt")
    dst = DetectionModel("yolov8", "n", 4)
    report = apply_state_dict(dst, ckpt["state_dict"])
    assert report["loaded"] == report["total"] and not report["missing"]
    other = DetectionModel("yolov8", "n", 7)
    assert apply_state_dict(other, ckpt["state_dict"])["shape_mismatch"]
