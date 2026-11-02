# Changelog

### v0.1.0
- Добавлен `POST /v1/detect` (DET-1): снимок JPEG, PNG или WebP телом как есть или полем `file` формы multipart, чтение в память с проверкой `MAX_IMAGE_MB` по ходу чтения и лимитом 25 мегапикселей, поворот по EXIF, параметры `conf`, `max_objects`, `classes`, объекты по убыванию `area` с `key` и `distance_m: null`, `timings` и этапы `decode`, `inference`, `postprocess` в `Server-Timing`; коды 400, 413, 415, 422.
- Добавлены загрузка и проверка словаря при старте и `GET /v1/vocabulary` (DET-2): обязательные классы, формы, уникальные `id` и `prompts`, `mentions` на общем подмножестве RE2 и Python `re`, хеш `8778f20c71d0`.
- Добавлено сопоставление классов модели со словарём через `prompts` (DET-3); `/v1/info` показывает `params.imgsz`, `conf`, `vocabulary_hash` и `unsupported_classes`.
- Добавлены направление по третям кадра и расстояние `near`/`far` по высоте и нижнему краю рамки с порогами для отдельных классов (DET-4) в `app/core/config/constants.py`.
- Бэкенд `fake` отвечает одной из трёх фиксированных сцен по SHA-256 снимка: ступени впереди близко, комната без опасностей, улица.
- Добавлены настройки детектора `VOCABULARY_PATH` и `MAX_IMAGE_MB` в `app/core/config/service.py` и `.env.example`; общий `config.py` не изменился.
- Добавлены `bench/latency.py` - замер в процессе с JSON EXP-2 (SVC-10) - и `bench/eval.py` - mAP50, полнота по ключевым классам и точность направления и расстояния по разметке COCO (DET-7) - с игрушечной разметкой `tests/data/eval/`.
- Добавлены зависимости `pillow` и `pyyaml`.
- README дополнен API обнаружения и словаря, сценами `fake`, таблицей лицензий бэкендов DET-5 и разделом замеров.
- Добавлен словарь классов `configs/vocabulary.yaml`: 26 классов, из них 16 ключевых, с запросами для моделей, русскими формами и шаблонами упоминаний.
- Добавлены копии эталонных контрактов из репозитория Шлюза в `tests/contract/`.
- Добавлена выдержка из общей спецификации в `docs/spec.md`.
- Добавлен каркас сервиса по `ainergy-python-style` (`--ml`): FastAPI, uv, Python 3.12 (`requires-python = ">=3.12,<3.13"`, `.python-version`), профили `cpu` и `gpu` для torch.
- Добавлен общий слой сервисов ИИ (SVC): Bearer-аутентификация, ошибки `{"detail": ...}`, заголовки `X-Request-ID` и `Server-Timing` на чистых ASGI-middleware, `request_id` в каждой строке лога, движок с пулом `WORKERS` и очередью `QUEUE_MAX` (сверх неё - 503 с `Retry-After: 1`), реестр бэкендов с ленивым импортом и бэкендом `fake` (`FAKE_LATENCY_MS`), прогрев тремя холостыми вызовами, `/health` и `/v1/info`.
- Добавлены образы `Dockerfile` (CUDA) и `Dockerfile.cpu` с аргументами `EXTRAS` и `GIT_SHA`, `compose.yml` и `compose-cpu.yml` на `127.0.0.1:8101` в сети `alv-net`, `.env.example`.
- Добавлен CI в GitHub Actions: быстрые тесты и публикация образов в GHCR.
- README переписан на русском по разделам SVC-11.
