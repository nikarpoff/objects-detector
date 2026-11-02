"""
Direction and distance of an object by its box (DET-4).

The box is normalized to the frame: x and y in [0, 1], y grows downwards. There is no
depth model yet, so the distance is a rough near/far guess and `distance_m` stays null.
"""

from dataclasses import dataclass
from enum import StrEnum

from app.core.config import GEOMETRY_SETTINGS as CFG, NearThresholds


class Direction(StrEnum):
    LEFT = "left"
    AHEAD = "ahead"
    RIGHT = "right"


class Distance(StrEnum):
    NEAR = "near"
    FAR = "far"


@dataclass(frozen=True)
class Box:
    """A box normalized to the frame: x1 < x2, y1 < y2, all in [0, 1]"""
    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_pixels(cls, x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> "Box":
        """The pixel box of a width x height frame, clipped to the frame."""
        return cls(
            x1=_clip(x1 / width),
            y1=_clip(y1 / height),
            x2=_clip(x2 / width),
            y2=_clip(y2 / height),
        )

    @property
    def is_valid(self) -> bool:
        return 0.0 <= self.x1 < self.x2 <= 1.0 and 0.0 <= self.y1 < self.y2 <= 1.0

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Share of the frame area."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)


def direction_of(box: Box) -> Direction:
    """Third of the frame the box centre is in; the bounds belong to `ahead`."""
    if box.center_x < CFG.LEFT_BOUND:
        return Direction.LEFT
    if box.center_x > CFG.RIGHT_BOUND:
        return Direction.RIGHT
    return Direction.AHEAD


def near_thresholds(label: str) -> NearThresholds:
    """Thresholds of the class, the default ones when it has none."""
    return CFG.NEAR_BY_CLASS.get(label, CFG.NEAR)


def distance_of(box: Box, label: str) -> Distance:
    """Near when the box is tall or its bottom edge is low: the object stands close to the phone."""
    thresholds = near_thresholds(label)
    if box.height >= thresholds.height or box.y2 >= thresholds.bottom:
        return Distance.NEAR
    return Distance.FAR


def _clip(value: float) -> float:
    return min(max(value, 0.0), 1.0)
