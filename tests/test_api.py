"""
Service API: health, info, vocabulary and detect; auth, request id, Server-Timing and errors.
"""

import io
import json
import os
import re
import tomllib

from contextlib import ExitStack
from pathlib import Path
from typing import AsyncIterator, Dict, Iterator

import pytest

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from loguru import logger
from PIL import Image

from app.api_v1.errors import register_error_handlers
from app.api_v1.schemas import DetectResponse, InfoParams, InfoResponse
from app.core.backends.fake import SCENES, scene_index
from app.core.config import get_service_settings, DETECTION_SETTINGS
from app.utils.middleware import current_timings, RequestIdMiddleware, ServerTimingMiddleware

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "tests" / "contract"

GENERATED_ID = re.compile(r"[0-9a-f]{8}")
SERVER_TIMING = re.compile(r"(?:[a-z_]+;dur=\d+\.\d, )*total;dur=\d+\.\d")

MEDIA_TYPES = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}


def make_image(image_format: str = "JPEG", size=(64, 48), color=(128, 128, 128)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format=image_format)
    return buffer.getvalue()


def image_of_scene(scene: int, image_format: str = "PNG") -> bytes:
    """An image the fake backend answers with the scene."""
    for shade in range(256):
        data = make_image(image_format, color=(shade, shade, shade))
        if scene_index(data) == scene:
            return data
    raise AssertionError(f"No image of scene {scene}")


def post_image(client: TestClient, data: bytes, headers: Dict[str, str], media_type: str = "image/jpeg", **params):
    return client.post("/v1/detect", content=data, params=params, headers={**headers, "Content-Type": media_type})


@pytest.fixture(scope="module")
def probe(client: TestClient) -> Iterator[TestClient]:
    """The shared middlewares, error handlers and engine behind the routes the services add later."""
    engine = client.app.state.engine
    app = FastAPI()
    register_error_handlers(app)
    app.add_middleware(ServerTimingMiddleware)
    app.add_middleware(RequestIdMiddleware)

    @app.get("/probe/log")
    async def log_lines() -> dict:
        logger.info("probe: event loop")
        await engine.run(lambda: logger.info("probe: worker thread"))
        return {}

    @app.get("/probe/stream")
    async def stream() -> StreamingResponse:
        with current_timings().stage("prepare"):
            pass

        async def chunks() -> AsyncIterator[bytes]:
            yield b"first"
            yield b"second"

        return StreamingResponse(chunks(), media_type="audio/L16; rate=24000; channels=1")

    @app.get("/probe/timings")
    async def timings() -> dict:
        with current_timings().stage("decode"):
            pass
        return current_timings().as_dict()

    @app.get("/probe/busy")
    async def busy() -> dict:
        with engine.admit():
            return {}

    @app.get("/probe/params")
    async def params(n: int = Query(..., ge=1)) -> dict:
        return {"n": n}

    @app.get("/probe/error")
    async def error() -> dict:
        raise RuntimeError("probe failure")

    with TestClient(app, raise_server_exceptions=False) as probe_client:
        yield probe_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    # Not before the warmup
    engine = client.app.state.engine
    engine.warmed_up = False
    try:
        assert client.get("/health").status_code == 503
    finally:
        engine.warmed_up = True


def test_info(client, auth):
    response = client.get("/v1/info", headers=auth)
    assert response.status_code == 200
    info = response.json()

    # The shape of the reference example: the gateway builds its config_id of these keys
    example = json.loads((CONTRACT / "info.response.json").read_text())
    InfoResponse.model_validate(example)
    assert info.keys() == example.keys()

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert info["service"] == project["name"] == example["service"]
    assert info["version"] == project["version"]
    assert info["git_sha"] == os.environ["GIT_SHA"]
    assert (info["backend"], info["model"], info["device"]) == ("fake", "fake", "cpu")
    assert info["warmed_up"] is True
    assert info["params"]["latency_ms"] == 0

    # The real hash, not the placeholder of the example: the gateway compares it with /v1/vocabulary
    params = InfoParams.model_validate(info["params"])
    vocabulary = json.loads((CONTRACT / "vocabulary.response.json").read_text())
    assert params.vocabulary_hash == vocabulary["hash"] != example["params"]["vocabulary_hash"]
    assert (params.imgsz, params.conf) == (640, DETECTION_SETTINGS.CONF)
    # The fake model knows every prompt, like an open-vocabulary one
    assert params.unsupported_classes == []


def test_vocabulary(client, auth):
    response = client.get("/v1/vocabulary", headers=auth)
    assert response.status_code == 200
    assert response.json() == json.loads((CONTRACT / "vocabulary.response.json").read_text())
    assert client.get("/v1/vocabulary").status_code == 401


def test_detect_ok(client, auth):
    # Raw body of every image type; the fake answers by the hash of the bytes
    for image_format, media_type in MEDIA_TYPES.items():
        data = make_image(image_format)
        response = post_image(client, data, auth, media_type)
        assert response.status_code == 200, image_format

        body = DetectResponse.model_validate(response.json()).model_dump(mode="json")
        assert body["image"] == {"width": 64, "height": 48}
        assert body["model"] == {"backend": "fake", "name": "fake", "imgsz": 640, "device": "cpu"}
        assert body["timings"].keys() == {"decode_ms", "inference_ms", "postprocess_ms", "total_ms"}
        assert response.headers["Server-Timing"].startswith("decode;dur=")

        objects = body["objects"]
        assert objects, image_format
        assert all(found["distance_m"] is None for found in objects)
        assert all("distance_m" in found for found in response.json()["objects"])
        assert [found["area"] for found in objects] == sorted((found["area"] for found in objects), reverse=True)
        for found in objects:
            box = found["box"]
            assert 0 <= box["x1"] < box["x2"] <= 1 and 0 <= box["y1"] < box["y2"] <= 1
            assert found["confidence"] >= DETECTION_SETTINGS.CONF

    # The stairs scene gives the DET-1 example exactly
    example = json.loads((CONTRACT / "detect.response.json").read_text())
    stairs = image_of_scene(0)
    response = post_image(client, stairs, auth, "image/png")
    assert response.json()["objects"] == example["objects"]

    # Multipart for curl; the same snapshot gives the same answer
    multipart = client.post("/v1/detect", files={"file": ("scene.png", stairs, "image/png")}, headers=auth)
    assert multipart.status_code == 200
    assert multipart.json()["objects"] == example["objects"]
    # curl sends a .webp as octet-stream
    webp = client.post("/v1/detect", files={"file": ("scene.webp", make_image("WEBP"), "application/octet-stream")}, headers=auth)
    assert webp.status_code == 200

    # classes filters the vocabulary ids, max_objects cuts the largest
    room = image_of_scene(1)
    all_objects = post_image(client, room, auth, "image/png").json()["objects"]
    assert [found["label"] for found in all_objects] == ["sofa", "table", "chair"]
    # The model label «potted plant» is out of the vocabulary and dropped
    assert len(all_objects) == len(SCENES[1]) - 1

    only = post_image(client, room, auth, "image/png", classes="chair, table").json()["objects"]
    assert [found["label"] for found in only] == ["table", "chair"]
    assert post_image(client, room, auth, "image/png", max_objects=1).json()["objects"] == all_objects[:1]

    # conf drops the less confident
    street = image_of_scene(2)
    labels = [found["label"] for found in post_image(client, street, auth, "image/png", conf=0.7).json()["objects"]]
    assert labels == ["person", "car"]
    labels = [found["label"] for found in post_image(client, street, auth, "image/png", conf=0.3).json()["objects"]]
    assert set(labels) == {"person", "car", "traffic_light", "dog"}


@pytest.mark.parametrize(
    "data, media_type",
    [
        (b"", "image/jpeg"),
        (b"not an image at all", "image/jpeg"),
        (make_image("JPEG")[:200], "image/jpeg"),
        (make_image("GIF"), "image/png"),
    ],
    ids=["empty", "not_image", "truncated", "gif"],
)
def test_detect_invalid_image_400(client, auth, data, media_type):
    response = post_image(client, data, auth, media_type)
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail in ("Image could not be decoded", "Image is empty")

    # The contract example of an error
    if data:
        assert response.json() == json.loads((CONTRACT / "error.response.json").read_text())


def test_detect_invalid_multipart_400(client, auth):
    body = b"--xyz\r\nContent-Disposition: form-data; name=\"file\"\r\n\r\nbroken"
    response = post_image(client, body, auth, "multipart/form-data")
    assert response.status_code == 400
    assert response.json() == {"detail": "Multipart body has no boundary"}


def test_detect_too_large_413(client, auth):
    limit = get_service_settings().max_image_bytes
    data = make_image("JPEG") + b"\0" * limit

    response = post_image(client, data, auth)
    assert response.status_code == 413
    assert response.json() == {"detail": "Image is larger than 5 MB"}

    multipart = client.post("/v1/detect", files={"file": ("big.jpg", data, "image/jpeg")}, headers=auth)
    assert multipart.status_code == 413

    # A small file of too many pixels is refused before it unpacks
    buffer = io.BytesIO()
    Image.new("1", (6000, 5000)).save(buffer, format="PNG")
    response = post_image(client, buffer.getvalue(), auth, "image/png")
    assert response.status_code == 413
    assert response.json() == {"detail": "Image is larger than 25 megapixels"}


@pytest.mark.parametrize(
    "media_type", [None, "application/octet-stream", "image/gif", "text/plain", "application/json"],
)
def test_detect_unsupported_media_415(client, auth, media_type):
    headers = dict(auth)
    if media_type:
        headers["Content-Type"] = media_type
    response = client.post("/v1/detect", content=make_image("JPEG"), headers=headers)
    assert response.status_code == 415
    assert response.json()["detail"].startswith("Unsupported media type")

    # The type of the multipart file is checked too
    multipart = client.post("/v1/detect", files={"file": ("scene.gif", make_image("GIF"), "image/gif")}, headers=auth)
    assert multipart.status_code == 415


@pytest.mark.parametrize(
    "params, detail",
    [
        ({"conf": 1.5}, "Invalid request: Input should be less than or equal to 1 at query.conf"),
        ({"conf": -0.1}, "Invalid request: Input should be greater than or equal to 0 at query.conf"),
        ({"conf": "high"}, "Invalid request: Input should be a valid number, unable to parse string as a number at query.conf"),
        ({"max_objects": 0}, "Invalid request: Input should be greater than or equal to 1 at query.max_objects"),
        ({"max_objects": 101}, "Invalid request: Input should be less than or equal to 100 at query.max_objects"),
        ({"classes": "door,unicorn"}, "Unknown class 'unicorn' in classes"),
    ],
    ids=["conf_high", "conf_negative", "conf_text", "max_objects_zero", "max_objects_over", "unknown_class"],
)
def test_detect_bad_params_422(client, auth, params, detail):
    response = post_image(client, make_image("JPEG"), auth, **params)
    assert response.status_code == 422
    assert response.json() == {"detail": detail}


def test_detect_multipart_without_file_422(client, auth):
    response = client.post("/v1/detect", files={"image": ("scene.jpg", make_image("JPEG"), "image/jpeg")}, headers=auth)
    assert response.status_code == 422
    assert response.json() == {"detail": "Multipart body has no 'file' field"}


@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer wrong-token"}],
    ids=["no_token", "wrong_token"],
)
def test_detect_unauthorized_401(client, headers):
    response = post_image(client, make_image("JPEG"), headers)
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid token"}
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_detect_busy_503(client, auth):
    engine = client.app.state.engine
    places = engine.settings.WORKERS + engine.settings.QUEUE_MAX

    with ExitStack() as stack:
        for _ in range(places):
            stack.enter_context(engine.admit())
        response = post_image(client, make_image("JPEG"), auth)

    assert response.status_code == 503
    assert response.json() == {"detail": "Service is busy"}
    assert response.headers["Retry-After"] == "1"

    # The places are given back
    assert post_image(client, make_image("JPEG"), auth).status_code == 200


@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer wrong-token"}, {"Authorization": "Basic dGVzdC10b2tlbg=="}],
    ids=["no_token", "wrong_token", "not_bearer"],
)
def test_info_unauthorized_401(client, headers):
    response = client.get("/v1/info", headers=headers)
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid token"}
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_request_id_echoed(client, auth, probe):
    for path, headers in (("/health", {}), ("/v1/info", auth), ("/v1/info", {})):
        response = client.get(path, headers={**headers, "X-Request-ID": "req-0001"})
        assert response.headers["X-Request-ID"] == "req-0001", path

    response = post_image(client, make_image("JPEG"), {**auth, "X-Request-ID": "req-0003"})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "req-0003"

    # Every log line of the request carries the id, the worker thread lines too
    records = []
    sink = logger.add(lambda message: records.append(message.record), level="DEBUG")
    try:
        response = probe.get("/probe/log", headers={"X-Request-ID": "req-0002"})
    finally:
        logger.remove(sink)

    assert response.headers["X-Request-ID"] == "req-0002"
    lines = [record for record in records if record["message"].startswith("probe:")]
    assert len(lines) == 2
    assert all(record["extra"]["request_id"] == "req-0002" for record in lines)


def test_request_id_generated(client, auth):
    ids = [client.get("/v1/info", headers=auth).headers["X-Request-ID"] for _ in range(2)]
    assert all(GENERATED_ID.fullmatch(request_id) for request_id in ids)
    assert ids[0] != ids[1]

    # An id that would break a log line is replaced as well
    for bad_id in ("has spaces in it", "x" * 200):
        response = client.get("/health", headers={"X-Request-ID": bad_id})
        assert GENERATED_ID.fullmatch(response.headers["X-Request-ID"])


def test_server_timing_header(client, auth, probe):
    responses = {
        "health": client.get("/health"),
        "info": client.get("/v1/info", headers=auth),
        "unauthorized": client.get("/v1/info"),
        "detect": post_image(client, make_image("JPEG"), auth),
        "detect_error": post_image(client, b"broken", auth),
        "not_found": client.get("/v1/missing", headers=auth),
        "timings": probe.get("/probe/timings"),
        "stream": probe.get("/probe/stream"),
        "internal_error": probe.get("/probe/error"),
    }
    for name, response in responses.items():
        assert SERVER_TIMING.fullmatch(response.headers.get("Server-Timing", "")), name
        assert response.headers.get("X-Request-ID"), name

    # The detect stages are in the header and in the body
    detect = responses["detect"]
    assert detect.headers["Server-Timing"].startswith("decode;dur=")
    assert "inference;dur=" in detect.headers["Server-Timing"]
    assert detect.json()["timings"].keys() == {"decode_ms", "inference_ms", "postprocess_ms", "total_ms"}

    # The same stages go to the `timings` of a body
    timings = responses["timings"]
    assert timings.headers["Server-Timing"].startswith("decode;dur=")
    assert timings.json().keys() == {"decode_ms", "total_ms"}

    # A streaming response gets the stages done before its body
    stream = responses["stream"]
    assert stream.headers["Server-Timing"].startswith("prepare;dur=")
    assert stream.content == b"firstsecond"

    # The 500 comes from outside of the middlewares and keeps the headers
    error = responses["internal_error"]
    assert error.status_code == 500
    assert error.json() == {"detail": "Internal server error"}


def test_busy_503(client, probe):
    engine = client.app.state.engine
    places = engine.settings.WORKERS + engine.settings.QUEUE_MAX

    with ExitStack() as stack:
        for _ in range(places):
            stack.enter_context(engine.admit())
        response = probe.get("/probe/busy")

    assert response.status_code == 503
    assert response.json() == {"detail": "Service is busy"}
    assert response.headers["Retry-After"] == "1"

    # The places are given back
    assert probe.get("/probe/busy").status_code == 200


def test_validation_error_422(probe):
    response = probe.get("/probe/params", params={"n": 0})
    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid request: Input should be greater than or equal to 1 at query.n"}
