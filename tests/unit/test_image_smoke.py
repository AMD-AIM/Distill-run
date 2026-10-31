"""scripts/image_smoke.py: NDJSON from coco8, the check list and the summary."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")

from distill_run.yolo.data import COCO80  # noqa: E402
from distill_run.yolo.ndjson import looks_like_ndjson  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
BASE_URL = "http://127.0.0.1:8000/coco8"
ENGINES = ("swift", "trl", "easydistill", "yolo", "ultralytics")


@pytest.fixture(scope="module")
def smoke():
    path = REPO / "scripts" / "image_smoke.py"
    spec = importlib.util.spec_from_file_location("image_smoke", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


def _checks(smoke, tmp_path: Path, device: str = "0"):
    checks = smoke.build_checks(tmp_path / "coco8", BASE_URL, device, tmp_path / "out", tmp_path)
    return {check.name: check for check in checks}, checks


def test_coco_to_ndjson_writes_header_and_one_line_per_image(smoke, tmp_path):
    root = tmp_path / "coco8"
    for name in ("images/train/a.jpg", "images/val/b.jpg"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(b"\xff\xd8\xff")
    (root / "labels" / "train").mkdir(parents=True)
    (root / "labels" / "train" / "a.txt").write_text("45 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    out = smoke.coco_to_ndjson(root, BASE_URL, tmp_path / "coco8.ndjson")
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert lines[0] == {
        "type": "dataset",
        "task": "detect",
        "class_names": {str(i): name for i, name in enumerate(COCO80)},
    }
    assert lines[1] == {
        "type": "image",
        "file": "a.jpg",
        "split": "train",
        "url": "http://127.0.0.1:8000/coco8/images/train/a.jpg",
        "annotations": {"boxes": [[45, 0.5, 0.5, 0.2, 0.2]]},
    }
    assert lines[2]["file"] == "b.jpg"
    assert lines[2]["split"] == "val"
    assert lines[2]["annotations"] == {"boxes": []}
    assert len(lines) == 3
    assert looks_like_ndjson(out)


def test_build_checks_uses_the_required_values(smoke, tmp_path):
    by_name, checks = _checks(smoke, tmp_path)
    assert [check.name for check in checks] == [
        "shell-tools",
        "doctor",
        "torch:core",
        "torch:swift",
        "torch:trl",
        "torch:easydistill",
        "torch:yolo",
        "torch:ultralytics",
        "ultralytics",
        "yolo",
    ]
    doctor = ["distill-run", "doctor"]
    for engine in ENGINES:
        doctor += ["--engine", engine]
    assert by_name["doctor"].argv == doctor

    def pairs(argv):
        return list(zip(argv, argv[1:], strict=False))

    ultralytics = pairs(by_name["ultralytics"].argv)
    for pair in (
        ("--engine", "ultralytics"),
        ("--model", "yolo26n.yaml"),
        ("--data", f"{BASE_URL}.ndjson"),
        ("--train-args", "epochs=2 imgsz=320 batch=4 workers=0 amp=False"),
    ):
        assert pair in ultralytics
    yolo_argv = by_name["yolo"].argv
    yolo = pairs(yolo_argv)
    for pair in (
        ("--engine", "yolo"),
        ("--model", "yolo26n"),
        ("--epochs", "1"),
        ("--max-steps", "2"),
        ("--imgsz", "320"),
        ("--batch", "4"),
        ("--workers", "0"),
        ("--device", "0"),
    ):
        assert pair in yolo
    assert "--no-amp" in yolo_argv

    for name in ("torch:yolo", "torch:ultralytics"):
        assert "import cv2" in by_name[name].argv[-1]
    for name in ("torch:core", "torch:swift"):
        assert "assert torch.cuda.is_available()" in by_name[name].argv[-1]
    for name in (f"torch:{venv}" for venv in ("core", *ENGINES)):
        assert "import distill_run.cli" in by_name[name].argv[-1]
    assert by_name["torch:trl"].argv[0] == str(tmp_path / "trl" / "bin" / "python")


def test_cpu_checks_do_not_require_a_gpu(smoke, tmp_path):
    by_name, _ = _checks(smoke, tmp_path, device="cpu")
    assert "assert torch.cuda.is_available()" not in by_name["torch:core"].argv[-1]
    assert "--device" not in by_name["yolo"].argv


def test_run_checks_reports_each_check_and_fails_if_any_fails(smoke, capsys):
    result = smoke.run_checks([smoke.Check("a", ["true"]), smoke.Check("b", ["false"])])
    out = capsys.readouterr().out
    assert result == 1
    assert "IMAGE_SMOKE a exit=0" in out
    assert "IMAGE_SMOKE b exit=1" in out
    assert "IMAGE_SMOKE result=FAIL" in out


def test_missing_artifact_fails_a_zero_exit(smoke, capsys):
    result = smoke.run_checks([smoke.Check("c", ["true"], expect=lambda: False)])
    out = capsys.readouterr().out
    assert result == 1
    assert "IMAGE_SMOKE c exit=0 artifact=missing" in out
    assert out.splitlines()[-1] == "IMAGE_SMOKE result=FAIL"


def test_all_passing_checks_pass(smoke, capsys):
    assert smoke.run_checks([smoke.Check("d", ["true"], expect=lambda: True)]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "IMAGE_SMOKE result=PASS"
