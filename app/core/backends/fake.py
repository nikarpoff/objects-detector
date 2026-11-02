import hashlib
import io
import time

from typing import Any, Dict, List, Sequence, Tuple

from loguru import logger
from PIL import Image

from app.core.backends.base import Backend
from app.core.config import Settings, DETECTION_SETTINGS
from app.core.detection.geometry import Box
from app.core.detection.image import Snapshot
from app.core.detection.mapping import RawDetection

# A label as a model gives it, the confidence and the box (x1, y1, x2, y2) in shares of the frame
FakeDetection = Tuple[str, float, Tuple[float, float, float, float]]

# Fixed scenes; SHA-256 of the snapshot bytes picks one (README, «Бэкенды и модели»).
# The labels are COCO and prompt names, as a real model gives them: the mapping turns
# them into vocabulary ids and drops «potted plant», which is out of the vocabulary
SCENES: Tuple[Tuple[FakeDetection, ...], ...] = (
    # 0. Stairs ahead and near: the passive mode warns about them (the DET-1 example)
    (
        ("stairs", 0.81, (0.3, 0.55, 0.72, 1.0)),
        ("chair", 0.64, (0.02, 0.4, 0.2, 0.75)),
    ),
    # 1. A room with nothing dangerous: the passive mode answers passive.clear
    (
        ("dining table", 0.77, (0.36, 0.5, 0.7, 0.8)),
        ("couch", 0.7, (0.0, 0.35, 0.3, 0.7)),
        ("chair", 0.58, (0.72, 0.45, 0.9, 0.85)),
        ("potted plant", 0.6, (0.92, 0.3, 0.99, 0.5)),
    ),
    # 2. A street: a person near on the left, a car far on the right, a traffic light ahead
    (
        ("person", 0.88, (0.05, 0.2, 0.25, 0.95)),
        ("car", 0.79, (0.7, 0.45, 0.98, 0.7)),
        ("traffic light", 0.66, (0.45, 0.05, 0.52, 0.25)),
        ("dog", 0.31, (0.3, 0.7, 0.4, 0.85)),
    ),
)

# Labels the fake model knows besides the vocabulary prompts: it imitates an open-vocabulary model
_OUT_OF_VOCABULARY = ("potted plant", )

WARMUP_SIZE = (640, 480)


def scene_index(data: bytes) -> int:
    """Scene of the snapshot: the first 8 bytes of its SHA-256, big-endian, modulo the scene count."""
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big") % len(SCENES)


class FakeBackend(Backend):
    """Deterministic stand-in for the model: no weights, no GPU; FAKE_LATENCY_MS imitates the inference time"""
    name = "fake"
    default_model = "fake"
    models = frozenset({"fake"})

    def __init__(self, settings: Settings, model: str) -> None:
        super().__init__(settings, model)
        self.latency_ms = settings.FAKE_LATENCY_MS
        self._warmup_image = self._make_warmup_image()
        logger.warning("Fake backend is on: the answers are synthetic, do not measure or deploy them")
        logger.info("Fake model ready: latency_ms={} scenes={}", self.latency_ms, len(SCENES))

    def class_names(self) -> Sequence[str]:
        prompts = [prompt for item in self.vocabulary.classes for prompt in item.prompts]
        return prompts + list(_OUT_OF_VOCABULARY)

    def detect(self, snapshot: Snapshot, conf: float) -> List[RawDetection]:
        """Blocking: call it from a worker thread."""
        self._imitate_inference()
        return [
            RawDetection(label=label, confidence=confidence, box=Box(*box))
            for label, confidence, box in SCENES[scene_index(snapshot.data)]
            if confidence >= conf
        ]

    def warmup(self) -> None:
        """Blocking: call it from a worker thread."""
        self.find_objects(self._warmup_image, DETECTION_SETTINGS.CONF, None, DETECTION_SETTINGS.MAX_OBJECTS)

    def params(self) -> Dict[str, Any]:
        return {**super().params(), "latency_ms": self.latency_ms}

    def _imitate_inference(self) -> None:
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000)

    @staticmethod
    def _make_warmup_image() -> bytes:
        buffer = io.BytesIO()
        Image.new("RGB", WARMUP_SIZE, (128, 128, 128)).save(buffer, format="JPEG")
        return buffer.getvalue()
