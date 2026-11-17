"""
Real models on the CPU (AC-54): a nano model finds a person and furniture on the snapshots of tests/data/smoke.

The `model` tests download the weights on the first run into the user cache (ARTIFACTS_PATH
unset), out of git. The onnx backend runs always: onnxruntime is in the dev group, the model
is Apache-2.0 and needs no torch. The torch backends run when their extra is installed.
The letterbox tests are fast and need no weights.
"""

import gc

from importlib.util import find_spec
from pathlib import Path
from typing import FrozenSet, Iterator, Set

import numpy as np
import pytest

from app.core.backends import Backend, build_backend
from app.core.backends.registry import BACKENDS
from app.core.config import Settings, DETECTION_SETTINGS
from app.core.detection.image import Snapshot
from app.core.detection.letterbox import letterbox

DATA = Path(__file__).parent / "data" / "smoke"
PERSON_SNAPSHOT = DATA / "person_path.jpg"          # a woman on a path, near and ahead
FURNITURE_SNAPSHOT = DATA / "room_chairs.jpg"       # a meeting room: a table and chairs, no people
FURNITURE: FrozenSet[str] = frozenset({"chair", "table"})

# Nano models of the backends: (MODEL_BACKEND, MODEL_NAME)
NANO_MODELS = [
    ("onnx", "rfdetr-nano"),
    ("rfdetr", "rfdetr-nano"),
    ("ultralytics", "yolo11n"),
]


def installed(backend: str) -> bool:
    return all(find_spec(package) is not None for package in BACKENDS[backend].packages)


def labels_on(backend: Backend, snapshot: Path) -> Set[str]:
    result = backend.find_objects(snapshot.read_bytes(), DETECTION_SETTINGS.CONF, None, DETECTION_SETTINGS.MAX_OBJECTS)
    return {found.label for found in result.objects}


@pytest.fixture(params=NANO_MODELS, ids=lambda case: f"{case[0]}-{case[1]}")
def nano_backend(request) -> Iterator[Backend]:
    name, model = request.param
    if name != "onnx" and not installed(name):
        pytest.skip(f"the '{BACKENDS[name].extra}' extra is not installed")

    # No .env: the test sees the defaults and its own values only
    backend = build_backend(Settings(_env_file=None, MODEL_BACKEND=name, MODEL_NAME=model, TARGET_PLATFORM="cpu"))
    try:
        yield backend
    finally:
        # One model at a time: the next one loads after this one is freed
        backend.close()
        del backend
        gc.collect()


@pytest.mark.model
def test_model_finds_person_and_furniture(nano_backend):
    nano_backend.warmup()

    assert "person" in labels_on(nano_backend, PERSON_SNAPSHOT)

    furniture = labels_on(nano_backend, FURNITURE_SNAPSHOT)
    assert furniture & FURNITURE, f"no chair or table, found: {sorted(furniture)}"
    # No people in the meeting room
    assert "person" not in furniture


@pytest.mark.model
def test_model_reports_unsupported_classes(nano_backend):
    # COCO models cannot find the classes out of COCO (DET-3)
    unsupported = nano_backend.params()["unsupported_classes"]
    assert {"door", "stairs", "crosswalk"} <= set(unsupported)
    assert "person" not in unsupported


def make_snapshot(width: int, height: int) -> Snapshot:
    return Snapshot(data=b"", pixels=np.zeros((height, width, 3), dtype=np.uint8))


@pytest.mark.parametrize("width, height", [(640, 480), (480, 640), (1000, 333), (640, 640), (100, 50)])
def test_letterbox_keeps_proportions(width, height):
    frame = letterbox(make_snapshot(width, height), 640)

    assert frame.pixels.shape == (640, 640, 3)
    # The frame fills one side and is centred on the other
    assert round(width * frame.scale_x) == 640 or round(height * frame.scale_y) == 640
    assert abs(frame.scale_x - frame.scale_y) < 0.01
    assert abs(frame.left - (640 - round(width * frame.scale_x)) // 2) <= 1
    assert abs(frame.top - (640 - round(height * frame.scale_y)) // 2) <= 1


def test_letterbox_box_back_to_frame():
    frame = letterbox(make_snapshot(640, 480), 320)

    # 640x480 to 320: scale 0.5, bars of 40 pixels at the top and the bottom
    assert (frame.left, frame.top) == (0, 40)

    whole = frame.to_frame(0, 40, 320, 280)
    assert (whole.x1, whole.y1, whole.x2, whole.y2) == (0.0, 0.0, 1.0, 1.0)

    # A box on the bars is clipped to the frame
    clipped = frame.to_frame(-10, 0, 160, 160)
    assert (clipped.x1, clipped.y1, clipped.x2, clipped.y2) == (0.0, 0.0, 0.5, 0.5)

    # The DETR family gives shares of the input
    normalized = frame.to_frame_normalized(0.25, 0.5, 0.75, 0.875)
    assert (normalized.x1, normalized.y1, normalized.x2, normalized.y2) == (0.25, 0.5, 0.75, 1.0)
