# Model and heuristic thresholds live here, not in env: they are tuned together with the code

from dataclasses import dataclass
from typing import Dict


class ENGINE_SETTINGS:
    """Inference engine shared by the services."""
    WARMUP_CALLS = 3                    # idle calls at startup: kernels, allocators and caches are ready before the first request


class DETECTION_SETTINGS:
    """Defaults of /v1/detect and the model input (DET-1, DET-6)."""
    CONF = 0.35                         # confidence threshold when the request sets none
    MAX_OBJECTS = 20                    # objects in a response when the request sets none
    MAX_OBJECTS_LIMIT = 100             # the most a request may ask for: the gateway names a few anyway
    IMGSZ = 640                         # model input side, pixels: a fixed size compiles the graph once
    MAX_PIXELS = 25_000_000             # decoded snapshot limit, ~75 MB of RGB: a 5 MB PNG may unpack to gigabytes
    MULTIPART_OVERHEAD = 64 * 1024      # multipart headers and boundaries allowed on top of MAX_IMAGE_MB


@dataclass(frozen=True)
class NearThresholds:
    """An object is near when its box is `height` of the frame tall or more, or its bottom edge is at `bottom` or lower"""
    height: float
    bottom: float


class GEOMETRY_SETTINGS:
    """Direction and distance by the box (DET-4); coordinates are shares of the frame, y grows downwards."""
    LEFT_BOUND = 1 / 3                  # the box centre left of it: left
    RIGHT_BOUND = 2 / 3                 # the box centre right of it: right; on the bounds and between them: ahead

    # The phone is held at the chest and looks ahead: the ground ~2.3 m away is at 0.9 of
    # the frame height, and a person of 1.7 m is half the frame tall from ~2.4 m
    NEAR = NearThresholds(height=0.5, bottom=0.9)

    # Classes whose size or place in the frame mislead the default thresholds
    NEAR_BY_CLASS: Dict[str, NearThresholds] = {
        "stairs": NearThresholds(height=0.35, bottom=0.85),          # flat and wide: a near flight is not tall
        "crosswalk": NearThresholds(height=0.35, bottom=0.85),       # a marking on the ground
        "door": NearThresholds(height=0.7, bottom=0.9),              # 2 m tall: 0.7 of the frame from ~2 m
        "bus": NearThresholds(height=0.7, bottom=0.9),               # 3 m tall: 0.7 of the frame from ~3 m
        "truck": NearThresholds(height=0.7, bottom=0.9),
        "pole": NearThresholds(height=0.9, bottom=0.9),              # cut by the frame top: tall even when far
        "traffic_light": NearThresholds(height=0.25, bottom=1.0),    # hangs high: only its size tells
    }
