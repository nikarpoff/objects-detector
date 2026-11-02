"""
Snapshot input: the request body to bytes and the bytes to pixels.

Everything stays in memory (SVC-8): the body is read chunk by chunk with the size
limit checked on the way, and a multipart body is parsed from the same stream, so
nothing is spooled to a temporary file.
"""

import io

from dataclasses import dataclass
from typing import AsyncIterator, Dict, Optional

import numpy as np

from PIL import Image, ImageOps, UnidentifiedImageError
from python_multipart.exceptions import MultipartParseError
from python_multipart.multipart import MultipartParser, parse_options_header

from app.core.config import DETECTION_SETTINGS as CFG
from app.core.exceptions import (
    InputTooLargeError,
    InvalidInputError,
    InvalidParamsError,
    UnsupportedMediaTypeError,
)

IMAGE_MEDIA_TYPES = ("image/jpeg", "image/png", "image/webp")
MULTIPART_MEDIA_TYPE = "multipart/form-data"
FILE_FIELD = "file"

# curl sends a file of an unknown extension (.webp among them) as octet-stream
_ANY_FILE_TYPE = "application/octet-stream"
_PIL_FORMATS = ("JPEG", "PNG", "WEBP")

DECODE_ERROR = "Image could not be decoded"


@dataclass(frozen=True, eq=False)
class Snapshot:
    """A decoded snapshot: the bytes as sent and its RGB pixels, height x width x 3"""
    data: bytes
    pixels: np.ndarray

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])


async def read_snapshot(content_type: Optional[str], body: AsyncIterator[bytes], limit: int) -> bytes:
    """The image of a request body: raw JPEG, PNG, WebP or the `file` field of a multipart form."""
    media_type, options = parse_options_header(content_type or "")
    media_type = media_type.decode("latin-1").lower()

    if media_type in IMAGE_MEDIA_TYPES:
        data = await _read_raw(body, limit)
    elif media_type == MULTIPART_MEDIA_TYPE:
        data = await _read_multipart(body, options.get(b"boundary"), limit)
    else:
        expected = ", ".join(IMAGE_MEDIA_TYPES)
        raise UnsupportedMediaTypeError(f"Unsupported media type '{media_type or 'none'}', expected {expected} or {MULTIPART_MEDIA_TYPE}")

    if not data:
        raise InvalidInputError("Image is empty")
    return data


def decode_snapshot(data: bytes) -> Snapshot:
    """The snapshot decoded to RGB and turned upright by its EXIF. Blocking: call it from a worker thread."""
    try:
        with Image.open(io.BytesIO(data), formats=_PIL_FORMATS) as image:
            # The header gives the size: refuse before a small file unpacks to gigabytes
            if image.width * image.height > CFG.MAX_PIXELS:
                raise InputTooLargeError(_too_many_pixels())
            upright = ImageOps.exif_transpose(image)
            pixels = np.asarray(upright.convert("RGB"))
    except Image.DecompressionBombError as e:
        raise InputTooLargeError(_too_many_pixels()) from e
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        raise InvalidInputError(DECODE_ERROR) from e

    return Snapshot(data=data, pixels=pixels)


async def _read_raw(body: AsyncIterator[bytes], limit: int) -> bytes:
    data = bytearray()
    async for chunk in body:
        data += chunk
        if len(data) > limit:
            raise InputTooLargeError(_too_large(limit))
    return bytes(data)


async def _read_multipart(body: AsyncIterator[bytes], boundary: Optional[bytes], limit: int) -> bytes:
    if not boundary:
        raise InvalidInputError("Multipart body has no boundary")

    reader = _FileFieldReader(boundary)
    received = 0
    try:
        async for chunk in body:
            received += len(chunk)
            if received > limit + CFG.MULTIPART_OVERHEAD:
                raise InputTooLargeError(_too_large(limit))
            reader.parser.write(chunk)
            if reader.file is not None and len(reader.file) > limit:
                raise InputTooLargeError(_too_large(limit))
        reader.parser.finalize()
    except MultipartParseError as e:
        raise InvalidInputError("Multipart body is malformed") from e

    if reader.file is None:
        raise InvalidParamsError(f"Multipart body has no '{FILE_FIELD}' field")
    if reader.file_type not in IMAGE_MEDIA_TYPES + (_ANY_FILE_TYPE, ""):
        expected = ", ".join(IMAGE_MEDIA_TYPES)
        raise UnsupportedMediaTypeError(f"Unsupported media type '{reader.file_type}' of the file, expected {expected}")
    return bytes(reader.file)


class _FileFieldReader:
    """Keeps the first `file` field of a multipart body in memory and skips the others"""

    def __init__(self, boundary: bytes) -> None:
        self.file: Optional[bytearray] = None
        self.file_type = ""

        self._headers: Dict[str, str] = {}
        self._field = bytearray()
        self._value = bytearray()
        self._target: Optional[bytearray] = None
        self._done = False

        self.parser = MultipartParser(boundary, callbacks={
            "on_part_begin": self._on_part_begin,
            "on_header_field": self._on_header_field,
            "on_header_value": self._on_header_value,
            "on_header_end": self._on_header_end,
            "on_headers_finished": self._on_headers_finished,
            "on_part_data": self._on_part_data,
            "on_part_end": self._on_part_end,
        })

    def _on_part_begin(self) -> None:
        self._headers = {}
        self._target = None

    def _on_header_field(self, data: bytes, start: int, end: int) -> None:
        self._field += data[start:end]

    def _on_header_value(self, data: bytes, start: int, end: int) -> None:
        self._value += data[start:end]

    def _on_header_end(self) -> None:
        self._headers[self._field.decode("latin-1").lower()] = self._value.decode("latin-1")
        self._field.clear()
        self._value.clear()

    def _on_headers_finished(self) -> None:
        _, options = parse_options_header(self._headers.get("content-disposition", ""))
        if self._done or options.get(b"name") != FILE_FIELD.encode():
            return

        media_type, _ = parse_options_header(self._headers.get("content-type", ""))
        self.file_type = media_type.decode("latin-1").lower()
        self.file = bytearray()
        self._target = self.file

    def _on_part_data(self, data: bytes, start: int, end: int) -> None:
        if self._target is not None:
            self._target += data[start:end]

    def _on_part_end(self) -> None:
        if self._target is not None:
            self._done = True
        self._target = None


def _too_large(limit: int) -> str:
    return f"Image is larger than {limit / (1024 * 1024):g} MB"


def _too_many_pixels() -> str:
    return f"Image is larger than {CFG.MAX_PIXELS / 1_000_000:g} megapixels"
