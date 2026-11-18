"""
Замер задержки детектора в процессе, без HTTP (SVC-10).

Бэкенд строится так же, как в сервисе, затем `--warmup` холостых вызовов и `--n`
замеров на одном и том же снимке. Замеряется весь путь запроса внутри сервиса:
декодирование, инференс и постобработка. Результат - JSON общей схемы EXP-2
в stdout и, если задан `--out`, в файл.

    uv run python -m bench.latency --backend fake --n 5 --warmup 1
    uv run python -m bench.latency --backend onnx --model yolo11n --n 50 --warmup 5 --image photo.jpg --out result.json
"""

import argparse
import io
import json
import math
import platform
import statistics
import sys
import time

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from app.core.backends import build_backend
from app.core.config import get_project, Settings, DETECTION_SETTINGS
from app.core.exceptions import BackendError
from app.utils.logging import configure_logging


def percentile(values: list[float], q: float) -> float:
    """Перцентиль методом nearest-rank: x[ceil(q·n)-1] - то же правило, что у Шлюза и loadgen."""
    ordered = sorted(values)
    return ordered[max(math.ceil(q * len(ordered)) - 1, 0)]


def summarize(durations_ms: list[float]) -> dict[str, float]:
    """p50, p95, среднее и выборочное (n-1) стандартное отклонение, мс."""
    return {
        "p50_ms": round(percentile(durations_ms, 0.5), 1),
        "p95_ms": round(percentile(durations_ms, 0.95), 1),
        "mean_ms": round(statistics.fmean(durations_ms), 1),
        "std_ms": round(statistics.stdev(durations_ms), 1) if len(durations_ms) > 1 else 0.0,
    }


def cpu_name() -> str:
    # В /proc/cpuinfo имя модели точнее, чем platform.processor() на Linux
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def gpu_name(device: str) -> str | None:
    if device != "cuda":
        return None
    try:
        import torch  # только у настоящих бэкендов на видеокарте
        return torch.cuda.get_device_name(0)
    except Exception:
        return "unknown"


def default_image() -> bytes:
    """Типичный вход: JPEG 640x480 с градиентом, чтобы декодер работал не на пустом кадре."""
    image = Image.linear_gradient("L").resize((640, 480)).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def run(args: argparse.Namespace) -> dict[str, Any]:
    overrides = {"MODEL_BACKEND": args.backend, "TARGET_PLATFORM": args.device}
    if args.model:
        overrides["MODEL_NAME"] = args.model
    settings = Settings(**overrides)
    backend = build_backend(settings)
    data = args.image.read_bytes() if args.image else default_image()

    def call() -> float:
        started = time.perf_counter()
        backend.find_objects(data, DETECTION_SETTINGS.CONF, None, DETECTION_SETTINGS.MAX_OBJECTS)
        return (time.perf_counter() - started) * 1000

    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        for _ in range(args.warmup):
            call()
        durations = [call() for _ in range(args.n)]
    finally:
        backend.close()

    return {
        "service": get_project().name,
        "backend": backend.name,
        "model": backend.model,
        "device": backend.device,
        "git_sha": settings.GIT_SHA,
        "n": args.n,
        "warmup": args.warmup,
        **summarize(durations),
        "hardware": {"cpu": cpu_name(), "gpu": gpu_name(backend.device)},
        "started_at": started_at,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", default="fake", help="бэкенд из реестра (по умолчанию %(default)s)")
    parser.add_argument("--model", default=None, help="модель бэкенда; по умолчанию - модель бэкенда по умолчанию")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"), help="устройство (по умолчанию %(default)s)")
    parser.add_argument("--n", type=int, default=50, help="число замеров (по умолчанию %(default)s)")
    parser.add_argument("--warmup", type=int, default=5, help="холостых вызовов до замеров (по умолчанию %(default)s)")
    parser.add_argument("--image", type=Path, default=None, help="снимок для замера; по умолчанию - синтетический JPEG 640x480")
    parser.add_argument("--out", type=Path, default=None, help="куда записать JSON, кроме stdout")
    args = parser.parse_args(argv)
    if args.n < 1 or args.warmup < 0:
        parser.error("--n должно быть не меньше 1, --warmup - не меньше 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Логи сервиса - в stderr: в stdout только JSON
    configure_logging("WARNING")
    try:
        result = run(args)
    except BackendError as e:
        # Неизвестный бэкенд или нет extra - понятное сообщение вместо трейсбека
        print(f"Замер не запущен: {e}", file=sys.stderr)
        return 1

    text = json.dumps(result, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
