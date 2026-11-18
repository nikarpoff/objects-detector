"""
RF-DETR backend: a real-time detection transformer under Apache-2.0 (DET-5).

Only the sizes whose weights are Apache-2.0 are listed; the XL and 2XL ones come with
`rfdetr[plus]` under the Roboflow platform license and are left out. The models are
trained on COCO, so the vocabulary classes out of COCO are unsupported.
"""

import threading

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import rfdetr
import torch

from loguru import logger
from rfdetr.assets.model_weights import download_pretrain_weights

from app.core.backends.base import Backend, weights_dir
from app.core.config import Settings, DETECTION_SETTINGS as CFG
from app.core.detection.image import Snapshot
from app.core.detection.letterbox import letterbox, warmup_snapshot
from app.core.detection.mapping import RawDetection

# MODEL_NAME to the class of the library
_VARIANTS: Dict[str, str] = {
    "rfdetr-nano": "RFDETRNano",
    "rfdetr-small": "RFDETRSmall",
    "rfdetr-medium": "RFDETRMedium",
    "rfdetr-large": "RFDETRLarge",
}


class RFDETRBackend(Backend):
    """RF-DETR models of the rfdetr library"""
    name = "rfdetr"
    default_model = "rfdetr-nano"
    models = frozenset(_VARIANTS)

    def __init__(self, settings: Settings, model: str) -> None:
        super().__init__(settings, model)
        self._half = self.device == "cuda"                      # FP16 on CUDA (DET-6)
        # One model object: calls from several workers go one by one
        self._lock = threading.Lock()

        # The library downloads the checkpoint to the given path when it is not there
        source = getattr(rfdetr, _VARIANTS[model])(pretrain_weights=str(_checkpoint(self.weights_dir, model)), device=self.device)

        # The model is trained at its own resolution: the input is that square, not IMGSZ
        self.imgsz = int(source.model_config.resolution)
        self._names: List[str] = list(source.class_names)
        # An inference snapshot of the weights serves the requests, in half precision on CUDA; the
        # training-ready source is dropped, so the weights are held once
        dtype = torch.float16 if self._half else torch.float32
        self._predictor = source.inference(compile=False, dtype=dtype)
        del source

        logger.info(
            "RF-DETR model ready: model={} device={} half={} imgsz={} classes={}",
            model, self.device, self._half, self.imgsz, len(self._names),
        )

    @classmethod
    def fetch(cls, settings: Settings, model: str) -> Optional[Path]:
        path = _checkpoint(weights_dir(settings), model)
        download_pretrain_weights(str(path))
        return path

    def class_names(self) -> Sequence[str]:
        return self._names

    def detect(self, snapshot: Snapshot, conf: float) -> List[RawDetection]:
        """Blocking: call it from a worker thread."""
        # The library stretches a frame to the model square; a letterboxed input keeps the proportions
        frame = letterbox(snapshot, self.imgsz)

        with self._lock:
            detections = self._predictor.predict(frame.pixels, threshold=conf, include_source_image=False)

        if len(detections) == 0:
            return []
        # COCO checkpoints number the classes by the sparse COCO ids: the library names them itself
        labels = detections.data["class_name"]
        return [
            RawDetection(label=str(label), confidence=float(confidence), box=frame.to_frame(*xyxy))
            for xyxy, confidence, label in zip(detections.xyxy.tolist(), detections.confidence.tolist(), labels)
        ][:CFG.MAX_DETECTIONS]

    def warmup(self) -> None:
        """Blocking: call it from a worker thread."""
        self.detect(warmup_snapshot(), CFG.CONF)

    def params(self) -> Dict[str, Any]:
        return {**super().params(), "half": self._half}


def _checkpoint(root: Path, model: str) -> Path:
    """The checkpoint of a size: the file name the library knows it by."""
    variant = getattr(rfdetr, _VARIANTS[model])
    return root / variant._model_config_class.model_fields["pretrain_weights"].default
