# Changelog

### v0.1.0
- Добавлены настоящие бэкенды DET-5 с ленивым импортом, каждый - отдельный extra: `ultralytics` (YOLO11, YOLO26, RT-DETR, YOLO-World, YOLOE; AGPL-3.0), `rfdetr` (RF-DETR Nano - Large; Apache-2.0), `onnx` (экспорты YOLO и RT-DETR на onnxruntime без PyTorch: CPU, CUDA, TensorRT по `ONNX_TENSORRT`; известные экспорты `rfdetr-nano`, `rtdetr-r18`, `rtdetrv2-r18` с Hugging Face по коммиту) и `grounding-dino` (Grounding DINO Tiny и Base; Apache-2.0). Реестр проверяет и torch у бэкендов на PyTorch.
- Добавлен letterbox до фиксированного квадратного входа модели с возвратом рамок в координаты кадра (DET-6, `app/core/detection/letterbox.py`); FP16 на CUDA у всех бэкендов; модели с открытым словарём получают классы из `prompts` при старте.
- Веса берутся из `ARTIFACTS_PATH`, без него - из `~/.cache/objects-detector`; недостающие скачиваются туда. Добавлен `scripts/fetch_weights.py` - веса текущих `MODEL_BACKEND` и `MODEL_NAME` заранее, без загрузки модели (DEP-6).
- Добавлены extras `ultralytics`, `rfdetr`, `onnx`, `grounding-dino`; профили `cpu` и `gpu` дополнены torchvision и onnxruntime (`onnxruntime-gpu` в `gpu`); onnxruntime добавлен в группу `dev` для модельного теста. Новые зависимости: `ultralytics` 8.4.215, CLIP из форка Ultralytics (архив коммита `7ffa84b`), `rfdetr` 1.12.1, `onnxruntime` и `onnxruntime-gpu` 1.32.0, `transformers` 5.23.0, `torchvision` 0.29.
- Образы по умолчанию ставят все бэкенды профиля: `Dockerfile.cpu` - `cpu ultralytics rfdetr onnx grounding-dino`, `Dockerfile` - `gpu ultralytics rfdetr grounding-dino`; в оба добавлены `libgl1` и `libglib2.0-0` для OpenCV.
- Добавлены `tests/test_model_smoke.py` (маркер `model`: nano-модели на CPU находят человека и стул или стол на снимках `tests/data/smoke/` под CC0; быстрые проверки letterbox) и проверки реестра: классы бэкендов и их extras, понятная ошибка неизвестной модели `onnx`.
- Добавлена настройка `ONNX_TENSORRT`; в `DETECTION_SETTINGS` - цвет полей letterbox, IoU для NMS, предел детекций и `text_threshold` Grounding DINO.
- README: таблица бэкендов с моделями и входом, веса и их загрузка, таблица лицензий дополнена экспортами ONNX, CLIP, Grounding DINO и PyTorch (проверено 2026-11-17), установка extras.
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
