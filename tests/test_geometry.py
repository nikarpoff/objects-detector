"""
Direction and distance by the box (DET-4) on reference cases.
"""

import pytest

from app.core.config import get_service_settings, GEOMETRY_SETTINGS
from app.core.detection.geometry import Box, Direction, Distance, direction_of, distance_of, near_thresholds
from app.core.detection.vocabulary import load_vocabulary


def box_centred_at(center_x: float, half_width: float = 0.05) -> Box:
    return Box(center_x - half_width, 0.4, center_x + half_width, 0.6)


def test_direction_bounds_are_thirds():
    assert GEOMETRY_SETTINGS.LEFT_BOUND == 1 / 3
    assert GEOMETRY_SETTINGS.RIGHT_BOUND == 2 / 3


@pytest.mark.parametrize(
    "box, expected",
    [
        (Box(0.0, 0.4, 0.2, 0.6), Direction.LEFT),
        (Box(0.0, 0.4, 0.66, 0.6), Direction.LEFT),             # centre 0.33, just left of 1/3
        (Box(0.0, 0.4, 2 / 3, 0.6), Direction.AHEAD),           # centre exactly 1/3: the bound is ahead
        (Box(0.4, 0.4, 0.6, 0.6), Direction.AHEAD),
        (Box(1 / 3, 0.4, 1.0, 0.6), Direction.AHEAD),           # centre exactly 2/3: the bound is ahead
        (Box(0.34, 0.4, 1.0, 0.6), Direction.RIGHT),            # centre 0.67, just right of 2/3
        (Box(0.8, 0.4, 1.0, 0.6), Direction.RIGHT),
        (Box(0.0, 0.0, 1.0, 1.0), Direction.AHEAD),             # the whole frame
    ],
    ids=["left", "left_near_bound", "left_bound", "centre", "right_bound", "right_near_bound", "right", "whole_frame"],
)
def test_direction_by_box_centre(box, expected):
    assert direction_of(box) == expected


def test_direction_around_bounds():
    epsilon = 1e-6
    assert direction_of(box_centred_at(1 / 3 - epsilon)) == Direction.LEFT
    assert direction_of(box_centred_at(1 / 3 + epsilon)) == Direction.AHEAD
    assert direction_of(box_centred_at(2 / 3 - epsilon)) == Direction.AHEAD
    assert direction_of(box_centred_at(2 / 3 + epsilon)) == Direction.RIGHT


@pytest.mark.parametrize(
    "box, label, expected",
    [
        (Box(0.3, 0.55, 0.72, 1.0), "stairs", Distance.NEAR),     # the DET-1 example
        (Box(0.02, 0.4, 0.2, 0.75), "chair", Distance.FAR),       # the DET-1 example
        (Box(0.4, 0.3, 0.6, 0.8), "person", Distance.NEAR),       # half the frame tall
        (Box(0.4, 0.3, 0.6, 0.79), "person", Distance.FAR),       # just under half, bottom edge high
        (Box(0.4, 0.8, 0.5, 0.9), "chair", Distance.NEAR),        # small, bottom edge at 0.9
        (Box(0.4, 0.7, 0.5, 0.89), "chair", Distance.FAR),
        (Box(0.4, 0.1, 0.6, 0.3), "chair", Distance.FAR),         # small and high
    ],
    ids=["stairs_example", "chair_example", "tall", "not_tall", "low_bottom", "high_bottom", "far"],
)
def test_distance_by_height_and_bottom(box, label, expected):
    assert distance_of(box, label) == expected


def test_distance_class_thresholds():
    # A door of 0.6 of the frame is far: doors are tall, a person of the same box is near
    door_box = Box(0.4, 0.2, 0.6, 0.8)
    assert distance_of(door_box, "person") == Distance.NEAR
    assert distance_of(door_box, "door") == Distance.FAR
    assert distance_of(Box(0.4, 0.1, 0.6, 0.85), "door") == Distance.NEAR

    # Stairs low in the frame are near before the default bottom threshold
    stairs_box = Box(0.3, 0.7, 0.7, 0.86)
    assert distance_of(stairs_box, "chair") == Distance.FAR
    assert distance_of(stairs_box, "stairs") == Distance.NEAR

    # A traffic light hangs high: its size tells
    light_box = Box(0.45, 0.05, 0.5, 0.32)
    assert distance_of(light_box, "traffic_light") == Distance.NEAR
    assert distance_of(light_box, "car") == Distance.FAR

    # A class without its own thresholds takes the default ones
    assert near_thresholds("sofa") == GEOMETRY_SETTINGS.NEAR
    assert near_thresholds("door") != GEOMETRY_SETTINGS.NEAR


def test_class_thresholds_name_vocabulary_classes():
    vocabulary = load_vocabulary(get_service_settings().vocabulary_path)
    assert set(GEOMETRY_SETTINGS.NEAR_BY_CLASS) <= set(vocabulary.ids)
    for thresholds in (GEOMETRY_SETTINGS.NEAR, *GEOMETRY_SETTINGS.NEAR_BY_CLASS.values()):
        assert 0 < thresholds.height <= 1 and 0 < thresholds.bottom <= 1


def test_box_area_and_pixels():
    box = Box(0.3, 0.55, 0.72, 1.0)
    assert round(box.area, 3) == 0.189
    assert box.is_valid

    pixels = Box.from_pixels(-10, 120, 320, 500, width=640, height=480)
    assert pixels == Box(0.0, 0.25, 0.5, 1.0)
    assert not Box(0.5, 0.2, 0.5, 0.4).is_valid
