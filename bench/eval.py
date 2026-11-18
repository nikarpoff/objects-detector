"""
Оценка качества детектора по разметке COCO (DET-7).

Разметка - `annotations.json` в формате COCO с атрибутами EXP-1: у аннотации
`direction` и `distance` (в `attributes` или на верхнем уровне), имена категорий -
идентификаторы словаря. Предсказания берутся либо из файла (`--predictions`, список
`{"image_id", "label" или "category_id", "score", "bbox": [x, y, w, h]}` в пикселях),
либо от бэкенда в процессе: снимки ищутся по `file_name` рядом с разметкой.

Метрики:
  - AP50 по каждому ключевому классу (101 точка, как в COCO) и mAP50 - их среднее
    по классам, у которых есть разметка;
  - полнота по каждому ключевому классу при рабочем `--conf`;
  - точность направления и расстояния для верно найденных объектов при `--conf`.

    uv run python -m bench.eval --annotations data/annotations.json --backend onnx --model yolo11n --out eval.json
    uv run python -m bench.eval --annotations data/annotations.json --predictions predictions.json
"""

import argparse
import json
import sys

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import get_service_settings, Settings, DETECTION_SETTINGS
from app.core.detection.geometry import Box, direction_of, distance_of
from app.core.detection.vocabulary import load_vocabulary
from app.core.exceptions import BackendError
from app.utils.logging import configure_logging

IOU_THRESHOLD = 0.5
RECALL_POINTS = 101
# Для mAP нужны и неуверенные предсказания: кривая точность-полнота строится по всем
MAP_CONF = 0.001


class EvalError(Exception):
    """Разметка или предсказания не читаются - сообщение для человека."""


@dataclass(frozen=True)
class Truth:
    image_id: int
    label: str
    box: Box
    direction: str | None
    distance: str | None


@dataclass(frozen=True)
class Prediction:
    image_id: int
    label: str
    score: float
    box: Box


def load_annotations(path: Path, known: set[str]) -> tuple[dict[int, dict], list[Truth], dict[int, str]]:
    """Снимки, разметка и категории; категория вне словаря - ошибка, иначе метрика молча врёт."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise EvalError(f"Разметка '{path}' не читается: {e}") from e

    categories = {category["id"]: category["name"] for category in document.get("categories", [])}
    unknown = sorted(set(categories.values()) - known)
    if unknown:
        raise EvalError(f"Категории вне словаря: {', '.join(unknown)}")

    images = {image["id"]: image for image in document.get("images", [])}
    truths = []
    for annotation in document.get("annotations", []):
        image = images.get(annotation["image_id"])
        if image is None:
            raise EvalError(f"Аннотация {annotation.get('id')} ссылается на неизвестный снимок {annotation['image_id']}")
        attributes = {**annotation, **annotation.get("attributes", {})}
        truths.append(Truth(
            image_id=image["id"],
            label=categories[annotation["category_id"]],
            box=pixel_box(annotation["bbox"], image),
            direction=attributes.get("direction"),
            distance=attributes.get("distance"),
        ))
    return images, truths, categories


def pixel_box(bbox: list[float], image: dict) -> Box:
    """COCO [x, y, w, h] в пикселях -> рамка в долях кадра."""
    x, y, w, h = bbox
    return Box.from_pixels(x, y, x + w, y + h, image["width"], image["height"])


def load_predictions(path: Path, images: dict[int, dict], categories: dict[int, str]) -> list[Prediction]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise EvalError(f"Предсказания '{path}' не читаются: {e}") from e

    predictions = []
    for item in document:
        image = images.get(item["image_id"])
        if image is None:
            raise EvalError(f"Предсказание для неизвестного снимка {item['image_id']}")
        label = item.get("label") or categories.get(item.get("category_id"))
        predictions.append(Prediction(item["image_id"], label, float(item["score"]), pixel_box(item["bbox"], image)))
    return predictions


def predict_with_backend(args: argparse.Namespace, images: dict[int, dict], root: Path) -> list[Prediction]:
    """Предсказания бэкенда в процессе: тот же путь, что у /v1/detect, с низким порогом."""
    from app.core.backends import build_backend  # бэкенд нужен только в этом режиме

    overrides = {"MODEL_BACKEND": args.backend, "TARGET_PLATFORM": args.device}
    if args.model:
        overrides["MODEL_NAME"] = args.model
    backend = build_backend(Settings(**overrides))
    predictions = []
    try:
        for image_id, image in images.items():
            data = (root / image["file_name"]).read_bytes()
            result = backend.find_objects(data, MAP_CONF, None, DETECTION_SETTINGS.MAX_OBJECTS_LIMIT)
            predictions += [Prediction(image_id, found.label, found.confidence, found.box) for found in result.objects]
    finally:
        backend.close()
    return predictions


def iou(a: Box, b: Box) -> float:
    width = min(a.x2, b.x2) - max(a.x1, b.x1)
    height = min(a.y2, b.y2) - max(a.y1, b.y1)
    if width <= 0 or height <= 0:
        return 0.0
    intersection = width * height
    return intersection / (a.area + b.area - intersection)


def match(truths: list[Truth], predictions: list[Prediction]) -> list[tuple[Prediction, Truth | None]]:
    """Жадное сопоставление по убыванию уверенности, как в COCO: каждая разметка - не больше одного раза."""
    used: set[int] = set()
    matches = []
    for prediction in sorted(predictions, key=lambda item: item.score, reverse=True):
        best, best_iou = None, IOU_THRESHOLD
        for index, truth in enumerate(truths):
            if index in used or truth.image_id != prediction.image_id or truth.label != prediction.label:
                continue
            overlap = iou(prediction.box, truth.box)
            if overlap >= best_iou:
                best, best_iou = index, overlap
        if best is not None:
            used.add(best)
        matches.append((prediction, truths[best] if best is not None else None))
    return matches


def average_precision(matches: list[tuple[Prediction, Truth | None]], total: int) -> float:
    """AP по 101 точке полноты с огибающей точности, как в pycocotools."""
    if total == 0:
        return 0.0
    precisions, recalls, found = [], [], 0
    for rank, (_, truth) in enumerate(matches, start=1):
        found += truth is not None
        precisions.append(found / rank)
        recalls.append(found / total)

    # Огибающая: точность при полноте r - лучшая точность при полноте не меньше r
    for index in range(len(precisions) - 2, -1, -1):
        precisions[index] = max(precisions[index], precisions[index + 1])

    points, position = [], 0
    for step in range(RECALL_POINTS):
        level = step / (RECALL_POINTS - 1)
        while position < len(recalls) and recalls[position] < level:
            position += 1
        points.append(precisions[position] if position < len(recalls) else 0.0)
    return sum(points) / RECALL_POINTS


def evaluate(truths: list[Truth], predictions: list[Prediction], key_classes: list[str], conf: float) -> dict[str, Any]:
    ap50, recall = {}, {}
    direction_hits = distance_hits = direction_total = distance_total = matched = 0

    for label in key_classes:
        class_truths = [truth for truth in truths if truth.label == label]
        if not class_truths:
            continue
        class_predictions = [prediction for prediction in predictions if prediction.label == label]
        ap50[label] = round(average_precision(match(class_truths, class_predictions), len(class_truths)), 4)

        working = match(class_truths, [prediction for prediction in class_predictions if prediction.score >= conf])
        found = [(prediction, truth) for prediction, truth in working if truth is not None]
        recall[label] = round(len(found) / len(class_truths), 4)
        matched += len(found)

        # Направление и расстояние сервис считает по рамке предсказания (DET-4)
        for prediction, truth in found:
            if truth.direction is not None:
                direction_total += 1
                direction_hits += direction_of(prediction.box) == truth.direction
            if truth.distance is not None:
                distance_total += 1
                distance_hits += distance_of(prediction.box, label) == truth.distance

    return {
        "iou": IOU_THRESHOLD,
        "conf": conf,
        "map50": round(sum(ap50.values()) / len(ap50), 4) if ap50 else None,
        "ap50": ap50,
        "recall": recall,
        "direction_accuracy": round(direction_hits / direction_total, 4) if direction_total else None,
        "distance_accuracy": round(distance_hits / distance_total, 4) if distance_total else None,
        "matched": matched,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    vocabulary = load_vocabulary(get_service_settings().vocabulary_path)
    images, truths, categories = load_annotations(args.annotations, set(vocabulary.ids))
    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if args.predictions:
        predictions = load_predictions(args.predictions, images, categories)
        source = {"predictions": str(args.predictions)}
    else:
        predictions = predict_with_backend(args, images, args.annotations.parent)
        source = {"backend": args.backend, "model": args.model, "device": args.device}

    key_classes = [item.id for item in vocabulary.classes if item.key]
    return {
        "service": "objects-detector",
        **source,
        "vocabulary_hash": vocabulary.hash,
        "images": len(images),
        "annotations": len(truths),
        **evaluate(truths, predictions, key_classes, args.conf),
        "started_at": started_at,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True, help="разметка COCO с атрибутами EXP-1")
    parser.add_argument("--predictions", type=Path, default=None, help="готовые предсказания; без него снимки прогоняются через бэкенд")
    parser.add_argument("--backend", default="fake", help="бэкенд из реестра (по умолчанию %(default)s)")
    parser.add_argument("--model", default=None, help="модель бэкенда; по умолчанию - модель бэкенда по умолчанию")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"), help="устройство (по умолчанию %(default)s)")
    parser.add_argument("--conf", type=float, default=DETECTION_SETTINGS.CONF, help="рабочий порог для полноты и направления (по умолчанию %(default)s)")
    parser.add_argument("--out", type=Path, default=None, help="куда записать JSON, кроме stdout")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging("WARNING")
    try:
        result = run(args)
    except (EvalError, BackendError) as e:
        print(f"Оценка не выполнена: {e}", file=sys.stderr)
        return 1

    text = json.dumps(result, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
