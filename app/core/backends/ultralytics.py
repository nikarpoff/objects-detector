"""
Ultralytics backend: YOLO11, YOLO26, RT-DETR, YOLO-World and YOLOE (DET-5).

The library and its weights are AGPL-3.0, so the backend is an optional extra (NFR-13).
Closed-vocabulary models name the COCO classes; open-vocabulary ones (YOLO-World, YOLOE)
take the vocabulary prompts as their classes once, at startup.
"""

import contextlib
import os
import threading

from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from loguru import logger

# The library pip-installs what it misses at run time: the service image is fixed, so it must not.
# The variable is read at the import of the library
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset
from ultralytics.utils.events import events

from app.core.backends.base import Backend, weights_dir
from app.core.config import Settings, DETECTION_SETTINGS as CFG
from app.core.detection.image import Snapshot
from app.core.detection.letterbox import letterbox, warmup_snapshot
from app.core.detection.mapping import RawDetection
from app.core.exceptions import BackendError

_CLOSED_MODELS = frozenset(
    [f"yolo11{size}" for size in "nsmlx"]
    + [f"yolo26{size}" for size in "nsmlx"]
    + [f"rtdetr-{size}" for size in "lx"]
)
_WORLD_MODELS = frozenset(
    [f"yolov8{size}-world" for size in "smlx"]
    + [f"yolov8{size}-worldv2" for size in "smlx"]
)
# YOLOE checkpoints are segmentation ones; the «-pf» (prompt-free) variants take no prompts
_YOLOE_MODELS = frozenset(
    [f"yoloe-v8{size}-seg" for size in "sml"]
    + [f"yoloe-11{size}-seg" for size in "sml"]
    + [f"yoloe-26{size}-seg" for size in "nsmlx"]
)

# The tokenizer of the text encoders: a package of the extra, installed from the Ultralytics fork
_CLIP_PACKAGE = "clip"


class UltralyticsBackend(Backend):
    """YOLO and RT-DETR models of the ultralytics library"""
    name = "ultralytics"
    default_model = "yolo11n"
    models = _CLOSED_MODELS | _WORLD_MODELS | _YOLOE_MODELS

    def __init__(self, settings: Settings, model: str) -> None:
        super().__init__(settings, model)
        self.open_vocabulary = model in _WORLD_MODELS | _YOLOE_MODELS
        self._device = "cuda:0" if self.device == "cuda" else "cpu"
        self._half = self.device == "cuda"                      # FP16 on CUDA (DET-6)
        # The library keeps one predictor per model: calls from several workers go one by one
        self._lock = threading.Lock()

        # Usage analytics of the library stay off: the service sends nothing about its requests
        events.enabled = False

        # The library downloads what it misses to the working directory: the checkpoint by
        # its path, the text encoder of an open-vocabulary model by a bare name
        with contextlib.chdir(self.weights_dir):
            self._model = YOLO(str(self.weights_dir / f"{model}.pt"))
            if self.open_vocabulary:
                self._set_prompts()

        self._names: Dict[int, str] = dict(self._model.names)
        logger.info(
            "Ultralytics model ready: model={} device={} half={} imgsz={} classes={} open_vocabulary={}",
            model, self._device, self._half, self.imgsz, len(self._names), self.open_vocabulary,
        )

    @classmethod
    def fetch(cls, settings: Settings, model: str) -> Optional[Path]:
        root = weights_dir(settings)
        if model in _WORLD_MODELS | _YOLOE_MODELS:
            # The text encoder is chosen by the checkpoint: the model is built once to fetch it too
            cls(settings, model).close()
        else:
            attempt_download_asset(str(root / f"{model}.pt"))
        return root

    def class_names(self) -> Sequence[str]:
        return list(self._names.values())

    def detect(self, snapshot: Snapshot, conf: float) -> List[RawDetection]:
        """Blocking: call it from a worker thread."""
        # 1. Letterbox here, not in the library: it would fit the input to the frame proportions,
        # and every new input shape costs a graph compilation
        frame = letterbox(snapshot, self.imgsz)

        # 2. The library takes BGR; boxes come back in the input pixels
        with self._lock:
            result = self._model.predict(
                frame.pixels[:, :, ::-1],
                imgsz=self.imgsz,
                conf=conf,
                iou=CFG.NMS_IOU,
                max_det=CFG.MAX_DETECTIONS,
                quantize=16 if self._half else 32,
                device=self._device,
                verbose=False,
            )[0]

        boxes = result.boxes
        return [
            RawDetection(label=self._names[int(class_index)], confidence=float(confidence), box=frame.to_frame(*xyxy))
            for xyxy, confidence, class_index in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist())
        ]

    def warmup(self) -> None:
        """Blocking: call it from a worker thread."""
        self.detect(warmup_snapshot(), CFG.CONF)

    def params(self) -> Dict[str, Any]:
        return {**super().params(), "half": self._half, "open_vocabulary": self.open_vocabulary}

    def _set_prompts(self) -> None:
        """The vocabulary prompts as the classes of an open-vocabulary model: the text encoder runs once."""
        if find_spec(_CLIP_PACKAGE) is None:
            raise BackendError(f"Model '{self.model}' of backend '{self.name}' requires the '{_CLIP_PACKAGE}' package of the 'ultralytics' extra")

        # CLIP weights of YOLO-World go next to the other weights, not to the library settings directory
        from ultralytics.nn import text_model
        text_model.WEIGHTS_DIR = self.weights_dir

        prompts = [prompt for item in self.vocabulary.classes for prompt in item.prompts]
        self._model.set_classes(prompts)
        logger.info("Prompts set: model={} prompts={}", self.model, len(prompts))
