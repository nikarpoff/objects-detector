from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.core.detection.geometry import Direction, Distance


class InfoResponse(BaseModel):
    """Service configuration; the gateway puts it into its config_id"""
    model_config = ConfigDict(extra="forbid")

    service: str = Field(description="Service name")
    version: str = Field(description="Service version, the same as in pyproject.toml")
    git_sha: str = Field(description="Commit the image is built from")
    backend: str = Field(description="Model backend, MODEL_BACKEND")
    model: str = Field(description="Model of the backend, MODEL_NAME")
    device: Literal["cpu", "cuda"] = Field(description="Inference device, TARGET_PLATFORM")
    warmed_up: bool = Field(description="The model has passed its warmup calls")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters that affect the answers")


class InfoParams(BaseModel):
    """The detector part of /v1/info params; a backend adds its own keys"""
    model_config = ConfigDict(extra="allow")

    imgsz: int = Field(description="Model input side, pixels")
    conf: float = Field(ge=0, le=1, description="Default confidence threshold")
    vocabulary_hash: str = Field(pattern=r"^[0-9a-f]{12}$", description="Hash of the vocabulary, the same as in /v1/vocabulary")
    unsupported_classes: Optional[List[str]] = Field(None, description="Vocabulary classes the model cannot find (DET-3)")


class RussianFormsInfo(BaseModel):
    """Russian forms of a class name"""
    model_config = ConfigDict(extra="forbid")

    one: str = Field(description="Form for one: «стул»")
    few: str = Field(description="Form for 2-4: «стула»")
    many: str = Field(description="Form for 5 and more: «стульев»")
    gender: Literal["m", "f", "n", "pl"] = Field(description="Grammatical gender; pl - plural only")


class VocabularyClassInfo(BaseModel):
    """A class of the vocabulary"""
    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Class id, latin snake_case")
    prompts: List[str] = Field(description="English prompts and COCO names of the class")
    ru: RussianFormsInfo = Field(description="Russian forms of the name")
    key: bool = Field(description="Key class: the hypothesis metric is counted over them")
    mentions: List[str] = Field(description="Patterns of the class mentions in a normalized answer")


class VocabularyResponse(BaseModel):
    """The vocabulary and its hash"""
    model_config = ConfigDict(extra="forbid")

    hash: str = Field(description="First 12 hex of SHA-256 of the canonical JSON of classes")
    classes: List[VocabularyClassInfo] = Field(description="Classes in the vocabulary order")


class ImageInfo(BaseModel):
    """Size of the decoded snapshot"""
    model_config = ConfigDict(extra="forbid")

    width: int = Field(description="Width, pixels")
    height: int = Field(description="Height, pixels")


class BoxInfo(BaseModel):
    """Box normalized to [0, 1]: x1 < x2, y1 < y2"""
    model_config = ConfigDict(extra="forbid")

    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)
    x2: float = Field(ge=0, le=1)
    y2: float = Field(ge=0, le=1)


class ObjectInfo(BaseModel):
    """A detected object"""
    model_config = ConfigDict(extra="forbid")

    label: str = Field(description="Class id of the vocabulary")
    label_ru: str = Field(description="Russian name of the class, the form `one`")
    confidence: float = Field(ge=0, le=1, description="Model confidence")
    box: BoxInfo = Field(description="Box normalized to the frame")
    direction: Direction = Field(description="Third of the frame of the box centre")
    distance: Distance = Field(description="Rough distance by the box height and bottom edge")
    distance_m: Optional[float] = Field(description="Reserved for a depth estimate; null for now")
    area: float = Field(ge=0, le=1, description="Share of the frame area")
    key: bool = Field(description="Key class of the vocabulary")


class ModelInfo(BaseModel):
    """Model that answered"""
    model_config = ConfigDict(extra="forbid")

    backend: str = Field(description="MODEL_BACKEND")
    name: str = Field(description="Model of the backend")
    imgsz: int = Field(description="Model input side, pixels")
    device: Literal["cpu", "cuda"] = Field(description="Inference device")


class DetectResponse(BaseModel):
    """Objects on the snapshot, the largest first"""
    model_config = ConfigDict(extra="forbid")

    image: ImageInfo = Field(description="Size of the decoded snapshot")
    objects: List[ObjectInfo] = Field(description="Objects sorted by area, the largest first")
    model: ModelInfo = Field(description="Model that answered")
    timings: Dict[str, float] = Field(description="Stage -> ms, with total_ms since the request came")


class ErrorResponse(BaseModel):
    """Error of any endpoint"""
    model_config = ConfigDict(extra="forbid")

    detail: str = Field(description="Short english sentence without a period")


class HardwareInfo(BaseModel):
    """Hardware of the bench"""
    model_config = ConfigDict(extra="forbid")

    cpu: str
    gpu: Optional[str]


class BenchResult(BaseModel):
    """Result of bench/latency.py, the EXP-2 schema"""
    model_config = ConfigDict(extra="forbid")

    service: str
    backend: str
    model: str
    device: Literal["cpu", "cuda"]
    git_sha: str
    n: int = Field(ge=1)
    warmup: int = Field(ge=0)
    p50_ms: float
    p95_ms: float
    mean_ms: float
    std_ms: float
    hardware: HardwareInfo
    started_at: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
