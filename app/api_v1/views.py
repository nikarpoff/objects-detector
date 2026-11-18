from typing import Optional

from fastapi import (
    APIRouter,
    Request,
    Depends,
    HTTPException,
    Query,
    status,
)
from loguru import logger

from app.core.auth import verify_token
from app.core.config import get_project, get_service_settings, DETECTION_SETTINGS as CFG
from app.core.detection.image import read_snapshot, IMAGE_MEDIA_TYPES, MULTIPART_MEDIA_TYPE
from app.core.engine import InferenceEngine
from app.api_v1.schemas import (
    BoxInfo,
    DetectResponse,
    ImageInfo,
    InfoResponse,
    ModelInfo,
    ObjectInfo,
    VocabularyResponse,
)
from app.utils.middleware import current_timings

router = APIRouter(prefix="/v1", dependencies=[Depends(verify_token)])

BOX_DIGITS = 4                      # a box to 0.1 % of the frame is finer than any model gives

# The body is read by the endpoint itself: the spec shows both ways of sending the snapshot
_DETECT_BODY = {
    "requestBody": {
        "required": True,
        "content": {
            **{media_type: {"schema": {"type": "string", "format": "binary"}} for media_type in IMAGE_MEDIA_TYPES},
            MULTIPART_MEDIA_TYPE: {
                "schema": {
                    "type": "object",
                    "properties": {"file": {"type": "string", "format": "binary", "description": "Загружаемый снимок"}},
                    "required": ["file"],
                },
            },
        },
    },
}


def get_engine(request: Request) -> InferenceEngine:
    # Prefer app.state singleton created on startup
    svc = getattr(request.app.state, "engine", None)
    if isinstance(svc, InferenceEngine):
        return svc
    raise HTTPException(500, "service is not initialized")


@router.get("/info",
    responses={
        status.HTTP_401_UNAUTHORIZED: {"description": "No token or a wrong one"},
    },
)
async def info(engine: InferenceEngine = Depends(get_engine)) -> InfoResponse:
    """Service configuration for the gateway config_id"""
    project = get_project()
    backend = engine.backend
    return InfoResponse(
        service=project.name,
        version=project.version,
        git_sha=engine.settings.GIT_SHA,
        backend=backend.name,
        model=backend.model,
        device=backend.device,
        warmed_up=engine.warmed_up,
        params=backend.params(),
    )


@router.get("/vocabulary",
    responses={
        status.HTTP_401_UNAUTHORIZED: {"description": "No token or a wrong one"},
    },
)
async def vocabulary(engine: InferenceEngine = Depends(get_engine)) -> VocabularyResponse:
    """Class vocabulary of the detector and its hash"""
    vocabulary = engine.backend.vocabulary
    return VocabularyResponse.model_validate({"hash": vocabulary.hash, "classes": vocabulary.document})


@router.post("/detect", openapi_extra=_DETECT_BODY,
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "Not an image or a corrupt one"},
        status.HTTP_401_UNAUTHORIZED: {"description": "No token or a wrong one"},
        status.HTTP_413_CONTENT_TOO_LARGE: {"description": "Image is over MAX_IMAGE_MB"},
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {"description": "Content-Type is not JPEG, PNG, WebP or multipart"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Parameters failed validation"},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Queue is full; Retry-After: 1"},
    },
)
async def detect(
    request: Request,
    conf: float = Query(CFG.CONF, ge=0, le=1, description="Порог уверенности"),
    max_objects: int = Query(CFG.MAX_OBJECTS, ge=1, le=CFG.MAX_OBJECTS_LIMIT, description="Сколько объектов вернуть"),
    classes: Optional[str] = Query(None, description="Идентификаторы классов через запятую; по умолчанию все"),
    engine: InferenceEngine = Depends(get_engine),
) -> DetectResponse:
    """Objects on the snapshot: vocabulary class, box, direction and distance"""
    backend = engine.backend
    selected = backend.vocabulary.select(classes)
    data = await read_snapshot(
        request.headers.get("content-type"), request.stream(), get_service_settings().max_image_bytes,
    )

    with engine.admit():
        result = await engine.run(backend.find_objects, data, conf, selected, max_objects)

    logger.info(
        "Detect done: bytes={} size={}x{} objects={}", len(data), result.width, result.height, len(result.objects),
    )
    return DetectResponse(
        image=ImageInfo(width=result.width, height=result.height),
        objects=[
            ObjectInfo(
                label=found.label,
                label_ru=found.label_ru,
                confidence=found.confidence,
                box=BoxInfo(**{name: round(value, BOX_DIGITS) for name, value in vars(found.box).items()}),
                direction=found.direction,
                distance=found.distance,
                distance_m=None,
                area=found.area,
                key=found.key,
            )
            for found in result.objects
        ],
        model=ModelInfo(backend=backend.name, name=backend.model, imgsz=backend.imgsz, device=backend.device),
        timings=current_timings().as_dict(),
    )
