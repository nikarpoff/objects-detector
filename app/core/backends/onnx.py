"""
ONNX backend: exported YOLO and RT-DETR on onnxruntime, without PyTorch (DET-5).

A model is a file: `<MODEL_NAME>.onnx` in the weights directory, or one of the known
exports on Hugging Face, downloaded there. The output layout is read from the outputs
themselves, so the same code serves the exports of the ultralytics library and the
DETR-style exports of transformers.
"""

import ast
import json

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort

from huggingface_hub import hf_hub_download
from loguru import logger

from app.core.backends.base import Backend, weights_dir
from app.core.config import get_service_settings, Settings, DETECTION_SETTINGS as CFG
from app.core.detection.image import Snapshot
from app.core.detection.letterbox import letterbox, warmup_snapshot, Letterbox
from app.core.detection.mapping import RawDetection
from app.core.exceptions import BackendError


@dataclass(frozen=True)
class HubModel:
    """An export on Hugging Face, pinned to a commit: a repository may change under a name"""
    repository: str
    revision: str
    file: str = "onnx/model.onnx"
    half_file: str = "onnx/model_fp16.onnx"         # FP16 weights for CUDA (DET-6)


# Known exports under Apache-2.0; MODEL_NAME is the key. Any other model is a local file
HUB_MODELS: Dict[str, HubModel] = {
    "rfdetr-nano": HubModel("onnx-community/rfdetr_nano-ONNX", "eae21cee0687a91bcf9fa071605c48d7705d2d91"),
    "rtdetr-r18": HubModel("onnx-community/rtdetr_r18vd", "f2bee7444f4c29003df7e9ef07c4c1c6632eca6f"),
    "rtdetrv2-r18": HubModel("onnx-community/rtdetr_v2_r18vd-ONNX", "936f90b6a476c6da4dfe053fc521af55285976ba"),
}

HUB_DIR_NAME = "onnx"                               # known exports inside the weights directory
TENSORRT_CACHE_NAME = "tensorrt"                    # engines TensorRT builds, kept between starts
CONFIG_FILE = "config.json"                         # class names of a transformers export
PREPROCESSOR_FILE = "preprocessor_config.json"      # its input normalization
NAMES_METADATA = "names"                            # class names in the metadata of an ultralytics export

_CPU_PROVIDER = "CPUExecutionProvider"
_CUDA_PROVIDER = "CUDAExecutionProvider"
_TENSORRT_PROVIDER = "TensorrtExecutionProvider"

_HALF_INPUT = "tensor(float16)"
_END_TO_END_FIELDS = 6                              # x1, y1, x2, y2, confidence, class

# The classes exports without names are trained on, in the model order
COCO_CLASSES: Tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)

# Old spellings of the COCO classes some exports keep, to the ones the vocabulary prompts use
_COCO_ALIASES: Dict[str, str] = {
    "motorbike": "motorcycle",
    "aeroplane": "airplane",
    "sofa": "couch",
    "pottedplant": "potted plant",
    "diningtable": "dining table",
    "tvmonitor": "tv",
}


class OnnxBackend(Backend):
    """Exported detectors on onnxruntime: CPU, CUDA or TensorRT"""
    name = "onnx"
    default_model = "rfdetr-nano"

    def __init__(self, settings: Settings, model: str) -> None:
        super().__init__(settings, model)
        self._config_dir: Optional[Path] = None             # configs of a known export; a local file has none
        self._path = self._find_model()
        self._session = ort.InferenceSession(str(self._path), providers=self._providers())

        model_input = self._session.get_inputs()[0]
        self._input_name = model_input.name
        self._half = model_input.type == _HALF_INPUT
        self.imgsz = self._input_side(model_input.shape)
        self._output_names = [output.name for output in self._session.get_outputs()]
        self._names = self._read_names()
        self._mean, self._std = self._read_normalization()

        logger.info(
            "ONNX model ready: model={} file='{}' providers={} half={} imgsz={} classes={}",
            model, self._path, self._session.get_providers(), self._half, self.imgsz, len(self._names),
        )

    def class_names(self) -> Sequence[str]:
        return list(self._names.values())

    def detect(self, snapshot: Snapshot, conf: float) -> List[RawDetection]:
        """Blocking: call it from a worker thread."""
        # 1. Letterbox to the fixed input, 0..1, channels first
        frame = letterbox(snapshot, self.imgsz)
        tensor = frame.pixels.astype(np.float32) / 255
        if self._mean is not None:
            tensor = (tensor - self._mean) / self._std
        tensor = tensor.transpose(2, 0, 1)[None].astype(np.float16 if self._half else np.float32)

        # 2. The session is thread-safe: workers need no lock
        outputs = self._session.run(self._output_names, {self._input_name: np.ascontiguousarray(tensor)})
        outputs = [output.astype(np.float32) for output in outputs]

        # 3. Boxes back to the frame
        if len(outputs) > 1:
            return self._decode_detr(dict(zip(self._output_names, outputs)), frame, conf)
        return self._decode_single(outputs[0][0], frame, conf)

    def warmup(self) -> None:
        """Blocking: call it from a worker thread."""
        self.detect(warmup_snapshot(), CFG.CONF)

    def params(self) -> Dict[str, Any]:
        return {**super().params(), "half": self._half, "providers": self._session.get_providers()}

    def _find_model(self) -> Path:
        """The model file: a local export first, then a known one, downloaded when it is not there."""
        local = self.weights_dir / f"{self.model}.onnx"
        if local.is_file():
            return local

        hub_model = HUB_MODELS.get(self.model)
        if hub_model is None:
            raise BackendError(
                f"Model '{self.model}' of backend '{self.name}' is not found: expected the file '{local}' "
                f"or one of: {', '.join(sorted(HUB_MODELS))}"
            )

        path = _download(hub_model, self.weights_dir / HUB_DIR_NAME / self.model, self.device)
        self._config_dir = path.parent.parent
        return path

    @classmethod
    def fetch(cls, settings: Settings, model: str) -> Optional[Path]:
        root = weights_dir(settings)
        local = root / f"{model}.onnx"
        if local.is_file() or model not in HUB_MODELS:
            # A local export is put there by hand; the start names a missing one
            return local
        return _download(HUB_MODELS[model], root / HUB_DIR_NAME / model, settings.TARGET_PLATFORM).parent.parent

    def _providers(self) -> List[Any]:
        if self.device != "cuda":
            return [_CPU_PROVIDER]

        available = ort.get_available_providers()
        if _CUDA_PROVIDER not in available:
            raise BackendError(f"Backend '{self.name}' on cuda requires onnxruntime-gpu of the 'gpu' profile")

        providers: List[Any] = [_CUDA_PROVIDER, _CPU_PROVIDER]
        if get_service_settings().ONNX_TENSORRT and _TENSORRT_PROVIDER in available:
            # The engine is built at the first start and cached: the next ones load it
            cache = self.weights_dir / TENSORRT_CACHE_NAME
            cache.mkdir(parents=True, exist_ok=True)
            providers.insert(0, (_TENSORRT_PROVIDER, {
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(cache),
            }))
        return providers

    def _input_side(self, shape: Sequence[Any]) -> int:
        """The side of the square input: fixed in the graph, else the one the export is made for, else IMGSZ."""
        height, width = shape[2], shape[3]
        if not isinstance(height, int) or not isinstance(width, int):
            # A dynamic graph takes any size, but the model works at the one it is trained at
            size = (self._read_json(PREPROCESSOR_FILE) or {}).get("size") or {}
            height, width = size.get("height", self.imgsz), size.get("width", self.imgsz)
        if height != width:
            raise BackendError(f"Model '{self.model}' of backend '{self.name}' has a {width}x{height} input, a square one is required")
        return int(height)

    def _read_names(self) -> Dict[int, str]:
        """Class names by the model class index: the export metadata, the config next to it, else COCO."""
        metadata = self._session.get_modelmeta().custom_metadata_map
        if NAMES_METADATA in metadata:
            names = ast.literal_eval(metadata[NAMES_METADATA])
        elif (config := self._read_json(CONFIG_FILE)) and config.get("id2label"):
            names = config["id2label"]
        else:
            names = dict(enumerate(COCO_CLASSES))

        return {int(index): _COCO_ALIASES.get(str(name), str(name)) for index, name in dict(names).items()}

    def _read_normalization(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Mean and std of a transformers export that normalizes its input; the others take 0..1."""
        config = self._read_json(PREPROCESSOR_FILE)
        if not config or not config.get("do_normalize"):
            return None, None
        return np.asarray(config["image_mean"], dtype=np.float32), np.asarray(config["image_std"], dtype=np.float32)

    def _read_json(self, name: str) -> Optional[Dict[str, Any]]:
        if self._config_dir is None or not (self._config_dir / name).is_file():
            return None
        return json.loads((self._config_dir / name).read_text(encoding="utf-8"))

    def _decode_detr(self, outputs: Dict[str, np.ndarray], frame: Letterbox, conf: float) -> List[RawDetection]:
        """A transformers export: class logits and boxes (cx, cy, w, h) in shares of the input, a row per query."""
        if "logits" not in outputs or "pred_boxes" not in outputs:
            raise BackendError(f"Model '{self.model}' of backend '{self.name}' has unknown outputs: {', '.join(outputs)}")

        scores = _sigmoid(outputs["logits"][0])
        return self._collect(_center_to_corners(outputs["pred_boxes"][0]), scores, frame, conf, normalized=True)

    def _decode_single(self, output: np.ndarray, frame: Letterbox, conf: float) -> List[RawDetection]:
        """An ultralytics export: its one output tells the layout by its shape."""
        # 1. NMS-free YOLO (YOLO26, YOLOv10): rows of corners in the input pixels, confidence and class
        if output.shape[1] == _END_TO_END_FIELDS and len(self._names) != _END_TO_END_FIELDS - 4:
            keep = output[:, 4] >= conf
            return self._build(output[keep, :4], output[keep, 4], output[keep, 5].astype(int), frame, normalized=False)

        # 2. Raw YOLO head: channels first, a column per anchor - centre boxes in the input pixels, class scores
        if output.shape[0] < output.shape[1]:
            output = output.T
            boxes, scores = _center_to_corners(output[:, :4]), output[:, 4:]
            classes = scores.argmax(axis=1)
            confidences = scores[np.arange(len(classes)), classes]
            keep = confidences >= conf
            boxes, confidences, classes = boxes[keep], confidences[keep], classes[keep]
            keep = _nms(boxes, confidences, classes, CFG.NMS_IOU)
            return self._build(boxes[keep], confidences[keep], classes[keep], frame, normalized=False)

        # 3. RT-DETR: a row per query - centre boxes in shares of the input, class scores
        return self._collect(_center_to_corners(output[:, :4]), output[:, 4:], frame, conf, normalized=True)

    def _collect(self, boxes: np.ndarray, scores: np.ndarray, frame: Letterbox, conf: float, normalized: bool) -> List[RawDetection]:
        """The best class of each query over `conf`."""
        classes = scores.argmax(axis=1)
        confidences = scores[np.arange(len(classes)), classes]
        keep = confidences >= conf
        return self._build(boxes[keep], confidences[keep], classes[keep], frame, normalized)

    def _build(self, boxes: np.ndarray, confidences: np.ndarray, classes: np.ndarray, frame: Letterbox, normalized: bool) -> List[RawDetection]:
        to_frame = frame.to_frame_normalized if normalized else frame.to_frame
        order = np.argsort(-confidences)[:CFG.MAX_DETECTIONS]
        return [
            RawDetection(label=self._names[int(classes[i])], confidence=float(confidences[i]), box=to_frame(*boxes[i].tolist()))
            for i in order
            # An index out of the names is the «no object» class of a sparse numbering
            if int(classes[i]) in self._names
        ]


def _download(hub_model: HubModel, directory: Path, device: str) -> Path:
    """The model file of a known export with its configs, downloaded to `directory` when missing."""
    file = hub_model.half_file if device == "cuda" else hub_model.file
    for name in (file, CONFIG_FILE, PREPROCESSOR_FILE):
        # A file fetched before is taken as it is: the start needs no network then
        if not (directory / name).is_file():
            logger.info("Downloading '{}' of {} to '{}'", name, hub_model.repository, directory)
            hf_hub_download(hub_model.repository, name, revision=hub_model.revision, local_dir=directory)
    return directory / file


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-values))


def _center_to_corners(boxes: np.ndarray) -> np.ndarray:
    """Boxes (cx, cy, w, h) as (x1, y1, x2, y2)."""
    centers, halves = boxes[:, :2], boxes[:, 2:4] / 2
    return np.concatenate([centers - halves, centers + halves], axis=1)


def _nms(boxes: np.ndarray, confidences: np.ndarray, classes: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Indexes of the boxes kept: of the boxes of one class overlapping over the threshold the most confident stays."""
    if len(boxes) == 0:
        return np.empty(0, dtype=int)

    # Boxes of different classes are moved apart: one pass serves all the classes
    shifted = boxes + (classes[:, None] * (boxes.max() + 1)).astype(boxes.dtype)
    x1, y1, x2, y2 = shifted.T
    areas = (x2 - x1) * (y2 - y1)

    order = np.argsort(-confidences)
    keep = []
    while len(order) > 0:
        best, rest = order[0], order[1:]
        keep.append(best)

        width = np.clip(np.minimum(x2[best], x2[rest]) - np.maximum(x1[best], x1[rest]), 0, None)
        height = np.clip(np.minimum(y2[best], y2[rest]) - np.maximum(y1[best], y1[rest]), 0, None)
        intersection = width * height
        iou = intersection / (areas[best] + areas[rest] - intersection + 1e-9)
        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=int)
