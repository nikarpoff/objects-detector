"""
Model backends: interchangeable implementations of the service model.

A backend is chosen by MODEL_BACKEND, built once at startup with its weights loaded,
and called from the engine worker threads only.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Any, ClassVar, Dict, FrozenSet, List, Optional, Sequence

from loguru import logger

from app.core.config import Settings, get_service_settings, DETECTION_SETTINGS
from app.core.detection.image import Snapshot, decode_snapshot
from app.core.detection.mapping import ClassMapping, DetectedObject, RawDetection, to_objects
from app.core.detection.vocabulary import load_vocabulary
from app.utils.middleware import current_timings


@dataclass(frozen=True)
class DetectionResult:
    """Objects found on a snapshot of width x height pixels"""
    width: int
    height: int
    objects: List[DetectedObject]


class Backend(ABC):
    """A detection model behind the service API"""
    name: ClassVar[str]                                     # the MODEL_BACKEND value
    default_model: ClassVar[str]                            # the model when MODEL_NAME is unset
    models: ClassVar[FrozenSet[str]] = frozenset()          # known MODEL_NAME values; empty takes any

    def __init__(self, settings: Settings, model: str) -> None:
        # Models are loaded here, at startup: the first request must not pay for loading them.
        # The vocabulary comes first: open-vocabulary models take their classes from its prompts
        self.settings = settings
        self.model = model
        self.device = settings.TARGET_PLATFORM
        self.imgsz = DETECTION_SETTINGS.IMGSZ
        self.vocabulary = load_vocabulary(get_service_settings().vocabulary_path)
        logger.info("Vocabulary loaded: hash={} classes={}", self.vocabulary.hash, len(self.vocabulary.classes))

    @abstractmethod
    def class_names(self) -> Sequence[str]:
        """Class names the model can give, as it names them."""
        ...

    @abstractmethod
    def detect(self, snapshot: Snapshot, conf: float) -> List[RawDetection]:
        """Detections over `conf` in the model class names. Blocking: call it from a worker thread."""
        ...

    @abstractmethod
    def warmup(self) -> None:
        """An idle call on a typical input. Blocking: call it from a worker thread."""
        ...

    @cached_property
    def mapping(self) -> ClassMapping:
        """Model classes to the vocabulary (DET-3)."""
        mapping = ClassMapping(self.vocabulary, self.class_names())
        if mapping.unsupported:
            logger.warning("Classes the model cannot find: {}", ", ".join(mapping.unsupported))
        return mapping

    def find_objects(self, data: bytes, conf: float, classes: Optional[FrozenSet[str]], max_objects: int) -> DetectionResult:
        """Decode, detect and map to the vocabulary, each stage timed. Blocking: call it from a worker thread."""
        timings = current_timings()

        # 1. Decode in memory: no temporary files on the hot path
        with timings.stage("decode"):
            snapshot = decode_snapshot(data)

        # 2. The model itself
        with timings.stage("inference"):
            detections = self.detect(snapshot, conf)

        # 3. Vocabulary classes, direction and distance, the largest first
        with timings.stage("postprocess"):
            objects = to_objects(detections, self.mapping, self.vocabulary, conf, classes, max_objects)

        return DetectionResult(width=snapshot.width, height=snapshot.height, objects=objects)

    def params(self) -> Dict[str, Any]:
        """Backend parameters that affect the answers, for /v1/info."""
        return {
            "imgsz": self.imgsz,
            "conf": DETECTION_SETTINGS.CONF,
            "vocabulary_hash": self.vocabulary.hash,
            "unsupported_classes": list(self.mapping.unsupported),
        }

    def close(self) -> None:
        """Releases the model at shutdown."""
