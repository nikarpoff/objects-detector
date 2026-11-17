"""
Letterbox: a frame fitted into the square model input (DET-6).

The frame is scaled to fit and centred on grey bars, so its proportions stay and the
input size is the same for every snapshot: the model graph is compiled once. The
boxes a model gives are in the input pixels; `to_frame` turns them back.
"""

from dataclasses import dataclass

import numpy as np

from PIL import Image

from app.core.config import DETECTION_SETTINGS as CFG
from app.core.detection.geometry import Box
from app.core.detection.image import Snapshot


@dataclass(frozen=True, eq=False)
class Letterbox:
    """A frame in the model input: scaled by (`scale_x`, `scale_y`) and shifted by (`left`, `top`) input pixels"""
    pixels: np.ndarray              # RGB, size x size x 3, uint8
    width: int                      # of the frame, pixels
    height: int
    scale_x: float                  # the axes differ by the rounding of the scaled sides only
    scale_y: float
    left: int
    top: int

    @property
    def size(self) -> int:
        return int(self.pixels.shape[0])

    def to_frame(self, x1: float, y1: float, x2: float, y2: float) -> Box:
        """The box of the input pixels normalized to the frame and clipped to it."""
        return Box.from_pixels(
            (x1 - self.left) / self.scale_x,
            (y1 - self.top) / self.scale_y,
            (x2 - self.left) / self.scale_x,
            (y2 - self.top) / self.scale_y,
            self.width,
            self.height,
        )

    def to_frame_normalized(self, x1: float, y1: float, x2: float, y2: float) -> Box:
        """The same for a box in shares of the input, as the DETR family gives it."""
        return self.to_frame(x1 * self.size, y1 * self.size, x2 * self.size, y2 * self.size)


def letterbox(snapshot: Snapshot, size: int) -> Letterbox:
    """The snapshot scaled to fit a size x size input and centred on grey bars."""
    width, height = snapshot.width, snapshot.height
    scale = min(size / width, size / height)
    new_width = min(max(round(width * scale), 1), size)
    new_height = min(max(round(height * scale), 1), size)
    left = (size - new_width) // 2
    top = (size - new_height) // 2

    resized = snapshot.pixels
    if (new_width, new_height) != (width, height):
        resized = np.asarray(Image.fromarray(snapshot.pixels).resize((new_width, new_height), Image.Resampling.BILINEAR))

    pixels = np.full((size, size, 3), CFG.LETTERBOX_FILL, dtype=np.uint8)
    pixels[top:top + new_height, left:left + new_width] = resized

    # The scale of each axis by its rounded side: an edge of the scaled frame maps to the frame edge exactly
    return Letterbox(
        pixels=pixels,
        width=width,
        height=height,
        scale_x=new_width / width,
        scale_y=new_height / height,
        left=left,
        top=top,
    )


def warmup_snapshot() -> Snapshot:
    """A grey 640x480 frame for the idle calls: the phone camera proportions."""
    return Snapshot(data=b"", pixels=np.full((480, 640, 3), CFG.LETTERBOX_FILL, dtype=np.uint8))
