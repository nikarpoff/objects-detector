"""
Model output to the objects of the response (DET-3, DET-4).

A model names its classes its own way: COCO names for a closed vocabulary, the prompts
for an open one. The mapping turns them into vocabulary ids through the prompts and
drops the rest; the vocabulary classes no model class maps to are unsupported.
"""

from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

from app.core.detection.geometry import Box, Direction, Distance, direction_of, distance_of
from app.core.detection.vocabulary import Vocabulary


@dataclass(frozen=True)
class RawDetection:
    """A detection as the model gives it: its own class name and a normalized box"""
    label: str
    confidence: float
    box: Box


@dataclass(frozen=True)
class DetectedObject:
    """An object of the response: a vocabulary class with its place in the frame"""
    label: str
    label_ru: str
    confidence: float
    box: Box
    direction: Direction
    distance: Distance
    area: float
    key: bool


def model_label(name: str) -> str:
    """A model class name as the prompts write it: «Dining_Table» -> «dining table»."""
    return " ".join(name.replace("_", " ").lower().split())


class ClassMapping:
    """Model class names to vocabulary ids through the prompts (DET-3)"""

    def __init__(self, vocabulary: Vocabulary, model_classes: Iterable[str]) -> None:
        by_prompt = {prompt: item.id for item in vocabulary.classes for prompt in item.prompts}

        self._ids: Dict[str, str] = {}
        dropped = []
        for name in model_classes:
            class_id = by_prompt.get(model_label(name))
            if class_id is None:
                dropped.append(name)
            else:
                self._ids[model_label(name)] = class_id

        covered = set(self._ids.values())
        self.unsupported: Tuple[str, ...] = tuple(class_id for class_id in vocabulary.ids if class_id not in covered)
        self.dropped: Tuple[str, ...] = tuple(dropped)

    def class_id(self, name: str) -> Optional[str]:
        """The vocabulary id of a model class; None when it is out of the vocabulary."""
        return self._ids.get(model_label(name))


def to_objects(
    detections: Iterable[RawDetection],
    mapping: ClassMapping,
    vocabulary: Vocabulary,
    conf: float,
    classes: Optional[FrozenSet[str]],
    max_objects: int,
) -> List[DetectedObject]:
    """The objects of the vocabulary over `conf`, the largest first, at most `max_objects`."""
    objects = []
    for detection in detections:
        class_id = mapping.class_id(detection.label)
        if class_id is None or detection.confidence < conf or not detection.box.is_valid:
            continue
        if classes is not None and class_id not in classes:
            continue

        item = vocabulary.get(class_id)
        objects.append(DetectedObject(
            label=class_id,
            label_ru=item.ru.one,
            confidence=round(detection.confidence, 3),
            box=detection.box,
            direction=direction_of(detection.box),
            distance=distance_of(detection.box, class_id),
            area=round(detection.box.area, 3),
            key=item.key,
        ))

    # Of equal areas the more confident goes first
    objects.sort(key=lambda found: (found.box.area, found.confidence), reverse=True)
    return objects[:max_objects]
