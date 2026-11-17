"""
Веса модели заранее, для старта без сети (DEP-6).

Скачивает веса текущих MODEL_BACKEND и MODEL_NAME в ARTIFACTS_PATH (без него - в кэш
пользователя, `~/.cache/objects-detector`) и не загружает модель в память. Исключение -
модели ultralytics с открытым словарём: их текстовый кодировщик определяет сам чекпойнт,
поэтому модель строится один раз. Уже скачанные файлы не качаются заново.
Скрипт вызывает deployments/fetch-weights.sh в образе сервиса:

    python -m scripts.fetch_weights
    MODEL_BACKEND=onnx MODEL_NAME=rfdetr-nano ARTIFACTS_PATH=./model uv run python -m scripts.fetch_weights
"""

import argparse
import sys

from app.core.backends import BACKENDS
from app.core.backends.registry import _load_class
from app.core.config import Settings
from app.core.exceptions import AppError
from app.utils.logging import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", default=None, help="MODEL_BACKEND; по умолчанию из окружения")
    parser.add_argument("--model", default=None, help="MODEL_NAME; по умолчанию из окружения или модель бэкенда")
    args = parser.parse_args(argv)

    overrides = {"MODEL_BACKEND": args.backend, "MODEL_NAME": args.model}
    settings = Settings(**{name: value for name, value in overrides.items() if value is not None})
    configure_logging(settings.LOG_LEVEL, settings.LOG_DIAGNOSE)

    name = settings.MODEL_BACKEND
    try:
        spec = BACKENDS.get(name)
        if spec is None:
            raise AppError(f"неизвестный бэкенд '{name}', ожидается один из: {', '.join(sorted(BACKENDS))}")
        # Тот же импорт, что при старте сервиса: без extra - та же понятная ошибка
        backend_class = _load_class(name, spec)
        model = settings.MODEL_NAME or backend_class.default_model
        if backend_class.models and model not in backend_class.models:
            raise AppError(f"неизвестная модель '{model}' бэкенда '{name}', ожидается одна из: {', '.join(sorted(backend_class.models))}")
        path = backend_class.fetch(settings, model)
    except AppError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1

    print(f"{name}/{model}\t{path if path is not None else 'весов нет'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
