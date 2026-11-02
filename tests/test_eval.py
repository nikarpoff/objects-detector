"""
Detector evaluation (DET-7) on a toy COCO markup with known answers.
"""

import io
import json

from pathlib import Path

import pytest

from PIL import Image

from app.core.backends.fake import scene_index
from app.core.detection.geometry import Box
from bench.eval import Prediction, Truth, average_precision, iou, main, match

DATA = Path(__file__).resolve().parents[1] / "tests" / "data" / "eval"


def run_eval(capsys, *argv: str) -> dict:
    assert main(list(argv)) == 0
    return json.loads(capsys.readouterr().out)


def test_eval_toy_markup(capsys):
    result = run_eval(
        capsys, "--annotations", str(DATA / "annotations.json"), "--predictions", str(DATA / "predictions.json"),
    )

    # person: TP 0.9, FP 0.8, TP 0.3 -> precision 1 up to recall 0.5, then 2/3: (51 + 50 * 2/3) / 101
    assert result["ap50"]["person"] == pytest.approx((51 + 50 * 2 / 3) / 101, abs=1e-4)
    # door: one exact box; chair: IoU 1/3 is a miss
    assert result["ap50"]["door"] == 1.0
    assert result["ap50"]["chair"] == 0.0
    # Context classes do not count, key classes without markup are skipped
    assert result["ap50"].keys() == {"person", "door", "chair"}
    assert result["map50"] == pytest.approx((0.835 + 1.0 + 0.0) / 3, abs=1e-4)

    # At conf 0.35 the person of 0.3 is lost
    assert result["recall"] == {"person": 0.5, "door": 1.0, "chair": 0.0}

    # Found: the person on the left is right; the door box is on the right, marked ahead
    assert result["matched"] == 2
    assert result["direction_accuracy"] == 0.5
    assert result["distance_accuracy"] == 1.0
    assert (result["images"], result["annotations"], result["vocabulary_hash"]) == (2, 5, "8778f20c71d0")


def test_eval_conf_changes_recall(capsys):
    result = run_eval(
        capsys, "--annotations", str(DATA / "annotations.json"), "--predictions", str(DATA / "predictions.json"),
        "--conf", "0.2",
    )
    assert result["recall"]["person"] == 1.0
    assert result["matched"] == 3
    # The person in the middle of the second snapshot is ahead and near, as marked
    assert result["direction_accuracy"] == pytest.approx(2 / 3, abs=1e-4)
    # AP does not depend on the working conf
    assert result["ap50"]["person"] == pytest.approx(0.835, abs=1e-4)


def test_eval_fake_backend(capsys, tmp_path):
    # Snapshots the fake answers with the stairs scene, marked as the scene is
    (tmp_path / "images").mkdir()
    images, annotations = [], []
    shade = 0
    while len(images) < 2:
        buffer = io.BytesIO()
        Image.new("RGB", (640, 480), (shade, shade, shade)).save(buffer, format="PNG")
        shade += 1
        if scene_index(buffer.getvalue()) != 0:
            continue
        image_id = len(images) + 1
        (tmp_path / "images" / f"{image_id}.png").write_bytes(buffer.getvalue())
        images.append({"id": image_id, "file_name": f"images/{image_id}.png", "width": 640, "height": 480})
        annotations += [
            {"id": 2 * image_id - 1, "image_id": image_id, "category_id": 1, "bbox": [192, 264, 268.8, 216],
             "attributes": {"direction": "ahead", "distance": "near"}},
            {"id": 2 * image_id, "image_id": image_id, "category_id": 2, "bbox": [12.8, 192, 115.2, 168],
             "attributes": {"direction": "left", "distance": "far"}},
        ]
    markup = {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "stairs"}, {"id": 2, "name": "chair"}]}
    (tmp_path / "annotations.json").write_text(json.dumps(markup))

    result = run_eval(capsys, "--annotations", str(tmp_path / "annotations.json"), "--backend", "fake")
    assert result["backend"] == "fake"
    assert result["ap50"] == {"stairs": 1.0, "chair": 1.0}
    assert result["map50"] == 1.0
    assert result["recall"] == {"stairs": 1.0, "chair": 1.0}
    assert (result["direction_accuracy"], result["distance_accuracy"]) == (1.0, 1.0)


def test_eval_unknown_category_fails(capsys, tmp_path):
    markup = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "unicorn"}]}
    (tmp_path / "annotations.json").write_text(json.dumps(markup))

    assert main(["--annotations", str(tmp_path / "annotations.json"), "--predictions", str(DATA / "predictions.json")]) == 1
    assert "Категории вне словаря: unicorn" in capsys.readouterr().err


def test_eval_matching_rules():
    truth = Truth(1, "chair", Box(0.0, 0.0, 0.5, 0.5), None, None)
    assert iou(truth.box, Box(0.0, 0.0, 0.5, 0.5)) == 1.0
    assert iou(truth.box, Box(0.5, 0.5, 1.0, 1.0)) == 0.0

    # Two predictions of one object: the more confident matches, the other is a false positive
    predictions = [Prediction(1, "chair", 0.4, truth.box), Prediction(1, "chair", 0.9, truth.box)]
    matches = match([truth], predictions)
    assert [(prediction.score, found is not None) for prediction, found in matches] == [(0.9, True), (0.4, False)]
    assert average_precision(matches, 1) == 1.0
    assert average_precision([], 1) == 0.0
