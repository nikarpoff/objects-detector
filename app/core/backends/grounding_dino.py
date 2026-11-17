"""
Grounding DINO backend: the open-vocabulary accuracy reference, slow (DET-5).

The model reads the classes as a text: the vocabulary prompts joined into one query
at startup. It keeps its own preprocessing - the frame proportions with the short side
of the processor - because the reference must work as its authors measured it.
"""

import threading

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from huggingface_hub import snapshot_download
from loguru import logger
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from app.core.backends.base import Backend, weights_dir
from app.core.config import Settings, DETECTION_SETTINGS as CFG
from app.core.detection.geometry import Box
from app.core.detection.image import Snapshot
from app.core.detection.letterbox import warmup_snapshot
from app.core.detection.mapping import RawDetection

# MODEL_NAME to the Hugging Face repository and its commit: a repository may change under a name
_REPOSITORIES: Dict[str, Tuple[str, str]] = {
    "grounding-dino-tiny": ("IDEA-Research/grounding-dino-tiny", "a2bb814dd30d776dcf7e30523b00659f4f141c71"),
    "grounding-dino-base": ("IDEA-Research/grounding-dino-base", "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"),
}

HUB_CACHE_NAME = "huggingface"      # the Hugging Face cache inside the weights directory


class GroundingDinoBackend(Backend):
    """Grounding DINO of the transformers library"""
    name = "grounding_dino"
    default_model = "grounding-dino-tiny"
    models = frozenset(_REPOSITORIES)

    def __init__(self, settings: Settings, model: str) -> None:
        super().__init__(settings, model)
        self._half = self.device == "cuda"                      # FP16 on CUDA (DET-6), by autocast
        # One model object: calls from several workers go one by one
        self._lock = threading.Lock()

        repository, revision = _REPOSITORIES[model]
        cache_dir = self.weights_dir / HUB_CACHE_NAME
        self._processor = AutoProcessor.from_pretrained(repository, revision=revision, cache_dir=cache_dir)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(repository, revision=revision, cache_dir=cache_dir)
        self._model.to(self.device).eval()

        # The query format of the model: lowercase phrases, each closed with a dot
        self._prompts: List[str] = [prompt for item in self.vocabulary.classes for prompt in item.prompts]
        self._text = " ".join(f"{prompt} ." for prompt in self._prompts)
        self.imgsz = self._input_side()

        logger.info(
            "Grounding DINO model ready: model={} device={} half={} imgsz={} prompts={}",
            model, self.device, self._half, self.imgsz, len(self._prompts),
        )

    @classmethod
    def fetch(cls, settings: Settings, model: str) -> Optional[Path]:
        repository, revision = _REPOSITORIES[model]
        cache_dir = weights_dir(settings) / HUB_CACHE_NAME
        snapshot_download(repository, revision=revision, cache_dir=cache_dir)
        return cache_dir

    def class_names(self) -> Sequence[str]:
        return self._prompts

    def detect(self, snapshot: Snapshot, conf: float) -> List[RawDetection]:
        """Blocking: call it from a worker thread."""
        image = Image.fromarray(snapshot.pixels)

        with self._lock:
            inputs = self._processor(images=image, text=self._text, return_tensors="pt").to(self.device)
            with torch.inference_mode(), torch.autocast(self.device, dtype=torch.float16, enabled=self._half):
                outputs = self._model(**inputs)

            found = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=conf,
                text_threshold=CFG.TEXT_THRESHOLD,
                target_sizes=[(snapshot.height, snapshot.width)],
            )[0]

        # A box tied to the tokens of several prompts gets a merged label: the mapping drops it
        return [
            RawDetection(
                label=str(label),
                confidence=float(confidence),
                box=Box.from_pixels(*xyxy, snapshot.width, snapshot.height),
            )
            for xyxy, confidence, label in zip(found["boxes"].tolist(), found["scores"].tolist(), found["text_labels"])
        ][:CFG.MAX_DETECTIONS]

    def warmup(self) -> None:
        """Blocking: call it from a worker thread."""
        self.detect(warmup_snapshot(), CFG.CONF)

    def params(self) -> Dict[str, Any]:
        return {**super().params(), "half": self._half, "text_threshold": CFG.TEXT_THRESHOLD}

    def _input_side(self) -> int:
        """The short side the processor scales a frame to; IMGSZ when it does not say."""
        size = getattr(self._processor.image_processor, "size", None) or {}
        return int(dict(size).get("shortest_edge") or self.imgsz)
