# objects-detector

Сервис обнаружения объектов голосового ассистента для людей с нарушениями зрения. Принимает снимок с камеры телефона и возвращает найденные объекты: класс из общего словаря `configs/vocabulary.yaml`, рамку, направление (слева, впереди, справа) и расстояние (близко, далеко). Сервис не описывает сцену словами и не хранит снимки: описание строит Шлюз (`assistant-low-vision-platform`), он же - единственный потребитель API. По умолчанию сервис работает на бэкенде `fake`, настоящие модели (YOLO, RT-DETR, RF-DETR, Grounding DINO) включаются настройками. Требования - в [docs/spec.md](docs/spec.md).

## Назначение

Детектор, ASR и TTS устроены одинаково (раздел SVC спецификации), поэтому Шлюз работает с любым бэкендом любого из них одинаково:

- модель выбирается настройками `MODEL_BACKEND` и `MODEL_NAME`, загружается и прогревается при старте (3 холостых вызова), `/health` отвечает только после прогрева;
- инференс идёт в пуле из `WORKERS` потоков, ещё `QUEUE_MAX` запросов могут ждать; следующий запрос сразу получает 503: устаревшее описание сцены пользователю бесполезно;
- вход читается в память, временных файлов нет;
- у каждого запроса есть идентификатор `X-Request-ID`, он попадает в каждую строку лога о запросе;
- каждый ответ, включая потоковый, несёт заголовок `Server-Timing` с временем этапов;
- `/v1/info` описывает конфигурацию (сервис, версия, коммит, бэкенд, модель, устройство), Шлюз включает её в свой `config_id`.

## API

| Метод | Путь | |
| --- | --- | --- |
| `GET` | `/health` | `{"status": "ok"}`, когда модель загружена и прогрета |
| `GET` | `/v1/info` | конфигурация сервиса для `config_id` Шлюза |
| `GET` | `/v1/vocabulary` | словарь классов и его хеш (DET-2) |
| `POST` | `/v1/detect` | объекты на снимке (DET-1) |
| `GET` | `/docs` | Swagger UI |

Запросы к `/v1/*` требуют `Authorization: Bearer <AUTH_TOKEN>`; если `AUTH_TOKEN` не задан, аутентификация выключена.

Заголовок `X-Request-ID` запроса возвращается в ответе; без него сервис создаёт свой (8 hex-символов). Значение с пробелами, управляющими символами или длиннее 128 символов заменяется новым: оно пишется в каждую строку лога.

Заголовок `Server-Timing` перечисляет этапы, известные до начала тела ответа, и `total` - время до ответа: `decode;dur=2.1, inference;dur=11.6, postprocess;dur=0.7, total;dur=15.0`.

### POST /v1/detect

Тело - снимок как есть (`Content-Type: image/jpeg`, `image/png` или `image/webp`) или `multipart/form-data` с полем `file`, удобное для curl. Снимок читается в память с проверкой лимита `MAX_IMAGE_MB` по ходу чтения, временных файлов нет; ориентация берётся из EXIF.

| Параметр | По умолчанию | |
| --- | --- | --- |
| `conf` | `0.35` | порог уверенности, от 0 до 1 |
| `max_objects` | `20` | сколько объектов вернуть, от 1 до 100 |
| `classes` | все | идентификаторы классов словаря через запятую: `door,stairs` |

```bash
curl -H "Authorization: Bearer $AUTH_TOKEN" -H "Content-Type: image/jpeg" \
  --data-binary @photo.jpg "http://localhost:8101/v1/detect?conf=0.4&max_objects=10"

curl -H "Authorization: Bearer $AUTH_TOKEN" -F "file=@photo.jpg" "http://localhost:8101/v1/detect"
```

```json
{
  "image": {"width": 640, "height": 480},
  "objects": [
    {"label": "stairs", "label_ru": "ступени", "confidence": 0.81,
     "box": {"x1": 0.3, "y1": 0.55, "x2": 0.72, "y2": 1.0},
     "direction": "ahead", "distance": "near", "distance_m": null, "area": 0.189, "key": true},
    {"label": "chair", "label_ru": "стул", "confidence": 0.64,
     "box": {"x1": 0.02, "y1": 0.4, "x2": 0.2, "y2": 0.75},
     "direction": "left", "distance": "far", "distance_m": null, "area": 0.063, "key": true}
  ],
  "model": {"backend": "ultralytics", "name": "yoloe-11s-seg", "imgsz": 640, "device": "cuda"},
  "timings": {"decode_ms": 2.1, "inference_ms": 11.6, "postprocess_ms": 0.7, "total_ms": 15.0}
}
```

| Поле объекта | |
| --- | --- |
| `label`, `label_ru` | идентификатор класса словаря и его форма `ru.one` |
| `confidence` | уверенность модели |
| `box` | рамка в долях кадра: `x1 < x2`, `y1 < y2`, ось y вниз |
| `direction` | треть кадра центра рамки: `left`, `ahead`, `right`; границы 1/3 и 2/3 относятся к `ahead` |
| `distance` | `near` или `far`: рамка высокая или её нижний край низко в кадре (пороги - `GEOMETRY_SETTINGS` в `app/core/config/constants.py`, у части классов свои) |
| `distance_m` | всегда `null`: зарезервировано под оценку глубины |
| `area` | доля площади кадра; объекты отсортированы по ней по убыванию |
| `key` | ключевой ли класс словаря |

Классы модели сопоставляются со словарём через `prompts` (DET-3): классы модели вне словаря отбрасываются, а классы словаря, которых модель не знает, перечислены в `/v1/info` в `params.unsupported_classes`.

### GET /v1/vocabulary

```bash
curl -H "Authorization: Bearer $AUTH_TOKEN" "http://localhost:8101/v1/vocabulary"
```

```json
{
  "hash": "8778f20c71d0",
  "classes": [
    {"id": "door", "prompts": ["door"], "ru": {"one": "дверь", "few": "двери", "many": "дверей", "gender": "f"},
     "key": true, "mentions": ["двер(?:ь|и|ью|ей|ям|ями|ях)"]}
  ]
}
```

Словарь - `configs/vocabulary.yaml` (26 классов, 16 ключевых). Хеш - первые 12 hex SHA-256 от канонического JSON списка `classes` (ключи отсортированы, `ensure_ascii=False`, разделители `,` и `:`). Словарь проверяется при старте: обязательные ключевые и контекстные классы DET-2, `id` в snake_case без повторов, формы `one`, `few`, `many` и `gender`, английские `prompts` без повторов между классами, `mentions` на общем подмножестве RE2 и Python `re` (без `\`, lookaround и обратных ссылок, строчные, `е` вместо `ё`). Битый словарь останавливает старт с сообщением, где названы файл, класс и ошибка. Словарь - общий контракт со Шлюзом: эталон ответа лежит в `tests/contract/vocabulary.response.json` и меняется только синхронизацией из Шлюза.

### GET /v1/info

```bash
curl -H "Authorization: Bearer $AUTH_TOKEN" -H "X-Request-ID: 3f2c1ab0" \
  "http://localhost:8101/v1/info"
```

```json
{
  "service": "objects-detector",
  "version": "0.1.0",
  "git_sha": "3f2c1ab",
  "backend": "fake",
  "model": "fake",
  "device": "cpu",
  "warmed_up": true,
  "params": {"imgsz": 640, "conf": 0.35, "vocabulary_hash": "8778f20c71d0", "unsupported_classes": [], "latency_ms": 0}
}
```

| Поле | |
| --- | --- |
| `service`, `version` | имя и версия из `pyproject.toml` |
| `git_sha` | коммит, из которого собран образ (аргумент сборки `GIT_SHA`) |
| `backend`, `model` | `MODEL_BACKEND` и модель бэкенда (`MODEL_NAME` или модель по умолчанию) |
| `device` | `TARGET_PLATFORM`: `cpu` или `cuda` |
| `warmed_up` | модель прошла холостые вызовы |
| `params` | параметры, от которых зависит ответ: `imgsz`, порог `conf` по умолчанию, `vocabulary_hash`, `unsupported_classes`; у `fake` ещё `latency_ms` |

## Коды ответов

Ошибки приходят одним предложением на английском: `{"detail": "<причина>"}`.

| Код | Когда |
| --- | --- |
| `200` | успех |
| `400` | тело пустое, не изображение или битый файл: `{"detail": "Image could not be decoded"}`; битый multipart |
| `401` | нет токена или неверный токен: `{"detail": "Invalid token"}` |
| `413` | снимок больше `MAX_IMAGE_MB` или больше 25 мегапикселей |
| `415` | `Content-Type` не `image/jpeg`, `image/png`, `image/webp` или `multipart/form-data`; у файла в multipart - не изображение этих типов и не `application/octet-stream` |
| `422` | параметры не прошли проверку: `{"detail": "Invalid request: <что не так> at <где>"}`; неизвестный класс в `classes`; в multipart нет поля `file` |
| `500` | внутренняя ошибка: `{"detail": "Internal server error"}`, подробности только в логе |
| `503` | очередь заполнена: `{"detail": "Service is busy"}` и `Retry-After: 1`; у `/health` - модель ещё не прогрета |

Коды `400`, `413`, `415` и `422` относятся к `POST /v1/detect`.

## Конфигурация

Все переменные необязательные, значения по умолчанию - в таблице и в [.env.example](.env.example).

| Переменная | По умолчанию | |
| --- | --- | --- |
| `MODEL_BACKEND` | `fake` | бэкенд из реестра (раздел «Бэкенды и модели») |
| `MODEL_NAME` | - | модель бэкенда; не задана - модель бэкенда по умолчанию |
| `TARGET_PLATFORM` | `cpu` | `cpu` или `cuda` |
| `ARTIFACTS_PATH` | - | каталог весов; не задан - `~/.cache/objects-detector`; недостающие веса скачиваются туда |
| `FAKE_LATENCY_MS` | `0` | время инференса, которое изображает `fake`, мс на вызов |
| `AUTH_TOKEN` | - | Bearer-токен; не задан - аутентификация выключена |
| `WORKERS` | `1` | потоков инференса |
| `QUEUE_MAX` | `4` | запросов в ожидании потока; сверх этого - 503 |
| `GIT_SHA` | `unknown` | коммит для `/v1/info`; задаёт аргумент сборки образа, в `.env` не пишется |
| `LOG_LEVEL` | `INFO` | `TRACE`, `DEBUG`, `INFO`, `SUCCESS`, `WARNING`, `ERROR`, `CRITICAL` |
| `LOG_DIAGNOSE` | `False` | значения переменных в трейсбеках; только для отладки - в логи попадут данные запросов и токены |
| `VOCABULARY_PATH` | `configs/vocabulary.yaml` | словарь классов; относительный путь - от корня сервиса |
| `MAX_IMAGE_MB` | `5` | наибольший снимок `/v1/detect`, МБ (1 МБ = 1 048 576 байт) |
| `ONNX_TENSORRT` | `False` | бэкенд `onnx` на `cuda`: сначала пробовать TensorRT; первый старт строит движок несколько минут |

Пороги моделей и эвристик - не переменные окружения, а константы в `app/core/config/constants.py`: `DETECTION_SETTINGS` (порог `conf` и `max_objects` по умолчанию, `imgsz`, лимит пикселей, цвет полей letterbox, IoU для NMS, `text_threshold` Grounding DINO) и `GEOMETRY_SETTINGS` (границы направления, пороги расстояния, в том числе для отдельных классов). Общие переменные - в `app/core/config/config.py` (одинаков во всех сервисах ИИ), переменные детектора - в `app/core/config/service.py`.

## Бэкенды и модели

Бэкенд выбирается `MODEL_BACKEND`. Реестр (`app/core/backends/registry.py`) импортирует модуль только выбранного бэкенда, поэтому библиотеки настоящих моделей - необязательные extras: без них сервис устанавливается и проходит быстрые тесты. Неизвестный бэкенд, неизвестная модель или бэкенд без своего extra останавливают старт с понятным сообщением, например `Backend 'ultralytics' requires the 'ultralytics' extra`.

| `MODEL_BACKEND` | Extra | `MODEL_NAME` | По умолчанию | Вход |
| --- | --- | --- | --- | --- |
| `fake` | - | `fake`: детерминированный ответ без весов и видеокарты | `fake` | - |
| `ultralytics` | `ultralytics` | YOLO11 `yolo11{n,s,m,l,x}`, YOLO26 `yolo26{n,s,m,l,x}`, RT-DETR `rtdetr-{l,x}`; с открытым словарём YOLO-World `yolov8{s,m,l,x}-world`, `yolov8{s,m,l,x}-worldv2` и YOLOE `yoloe-v8{s,m,l}-seg`, `yoloe-11{s,m,l}-seg`, `yoloe-26{n,s,m,l,x}-seg` | `yolo11n` | letterbox до `imgsz` (640) |
| `rfdetr` | `rfdetr` | RF-DETR `rfdetr-{nano,small,medium,large}` | `rfdetr-nano` | letterbox до разрешения модели (nano - 384, small - 512, medium - 576, large - 704) |
| `onnx` | `onnx` (CPU) или профиль `gpu` | экспорты с Hugging Face `rfdetr-nano`, `rtdetr-r18`, `rtdetrv2-r18`; любой свой экспорт YOLO или RT-DETR - файл `<ARTIFACTS_PATH>/<MODEL_NAME>.onnx` | `rfdetr-nano` | letterbox до входа модели: фиксированного в графе, иначе из `preprocessor_config.json`, иначе `imgsz` |
| `grounding_dino` | `grounding-dino` | Grounding DINO `grounding-dino-tiny`, `grounding-dino-base` | `grounding-dino-tiny` | своя предобработка модели: короткая сторона 800 |

Как бэкенды выполняют DET-5 и DET-6:

- снимок декодируется в памяти и вписывается в квадратный вход модели с серыми полями (letterbox, `app/core/detection/letterbox.py`): пропорции сохраняются, размер входа один для всех снимков, поэтому граф модели компилируется один раз; рамки возвращаются в координаты исходного кадра;
- на `TARGET_PLATFORM=cuda` модели работают в FP16: `ultralytics` - `quantize=16`, `rfdetr` - снимок весов в `float16`, `grounding_dino` - `autocast`, `onnx` - файл `model_fp16.onnx` известного экспорта; `/v1/info` показывает это в `params.half`;
- модели с открытым словарём (YOLO-World, YOLOE, Grounding DINO) получают классы из `prompts` словаря один раз при старте; у них `unsupported_classes` пуст;
- модели с закрытым словарём (COCO) находят только классы COCO: `door`, `stairs`, `scooter`, `crosswalk`, `pole`, `barrier`, `wardrobe`, `window`, `trash_can` попадают в `params.unsupported_classes`;
- прогрев - 3 холостых вызова на сером кадре 640x480 до того, как `/health` ответит `ok`.

`onnx` работает на onnxruntime без PyTorch: на CPU - `CPUExecutionProvider`, на CUDA - `CUDAExecutionProvider` из `onnxruntime-gpu` профиля `gpu`, с `ONNX_TENSORRT=true` первым пробуется TensorRT (движок строится при первом старте несколько минут и кэшируется в `<ARTIFACTS_PATH>/tensorrt`). Формат выхода бэкенд определяет сам: `logits` и `pred_boxes` экспортов transformers (RF-DETR, RT-DETR), один выход экспорта ultralytics - YOLO без NMS (YOLO26, YOLOv10), сырая голова YOLO (NMS в бэкенде, IoU 0.7) или RT-DETR. Имена классов берутся из метаданных экспорта ultralytics, из `config.json` рядом с известным экспортом, иначе - классы COCO по порядку. Свой экспорт из ultralytics: `yolo export model=yolo11n.pt format=onnx imgsz=640`, файл кладётся в `ARTIFACTS_PATH` как `yolo11n.onnx`, а `MODEL_NAME=yolo11n`.

Веса не входят в образы. Бэкенд ищет их в `ARTIFACTS_PATH`, а если `ARTIFACTS_PATH` не задан - в `~/.cache/objects-detector` (`$XDG_CACHE_HOME/objects-detector`). Недостающее скачивается туда при старте; известные экспорты ONNX и Grounding DINO закреплены по коммиту репозитория на Hugging Face. Скачать веса заранее, без загрузки модели в память, для старта без сети:

```bash
MODEL_BACKEND=onnx MODEL_NAME=rfdetr-nano ARTIFACTS_PATH=./model uv run python -m scripts.fetch_weights
```

Скрипт печатает `onnx/rfdetr-nano<TAB><каталог весов>`; в развёртывании его запускает `deployments/fetch-weights.sh` в образе сервиса. Модели ultralytics с открытым словарём он строит один раз: их текстовый кодировщик (CLIP или MobileCLIP) выбирает сам чекпойнт.

Библиотека ultralytics запускается с `YOLO_AUTOINSTALL=False` и выключенной аналитикой: сервис не доустанавливает пакеты на ходу и ничего не отправляет о запросах.

На `fake` сервис пишет предупреждение при старте, а `/v1/info` показывает `backend: fake`, поэтому подмена модели в замерах сразу видна.

Бэкенд `fake` изображает модель с открытым словарём (`unsupported_classes` пуст) и отвечает одной из трёх фиксированных сцен. Сцену выбирают первые 8 байт SHA-256 тела снимка (big-endian) по модулю 3, поэтому один и тот же файл всегда даёт один и тот же ответ, а нужную сцену можно подобрать заранее: `python3 -c "import hashlib,sys;print(int.from_bytes(hashlib.sha256(open(sys.argv[1],'rb').read()).digest()[:8],'big')%3)" photo.jpg`.

| Сцена | Объекты | Пассивный режим Шлюза |
| --- | --- | --- |
| 0 | ступени впереди близко, стул слева далеко (пример DET-1) | предупреждение «Осторожно, впереди ступени» |
| 1 | стол впереди, диван слева, стул справа - всё далеко; `potted plant` вне словаря отбрасывается | `passive.clear` |
| 2 | человек слева близко, машина справа далеко, светофор впереди, собака с уверенностью 0.31 (ниже порога по умолчанию) | без предупреждения уровня `danger` |

Лицензии проверены по первоисточникам (файлы `LICENSE`, README и карточки моделей на Hugging Face): библиотеки - 2026-11-02, модели и экспорты бэкендов - 2026-11-17.

| Модель | Лицензия | Коммерческое использование |
| --- | --- | --- |
| `fake` (код сервиса, без весов) | Apache-2.0 | да |
| YOLO11, YOLO26, RT-DETR, YOLO-World, YOLOE в `ultralytics` (код и веса) | AGPL-3.0 | только с открытием кода сервиса по AGPL или с коммерческой лицензией Ultralytics Enterprise |
| CLIP в форке Ultralytics (`ultralytics/CLIP`, токенизатор YOLO-World и YOLOE, extra `ultralytics`) | AGPL-3.0 | как у `ultralytics` |
| RF-DETR Nano, Small, Medium, Large и RF-DETR-Seg всех размеров (`rfdetr`) | Apache-2.0 | да |
| RF-DETR XL, 2XL, Atto, Femto, Pico (`rfdetr[plus]`, бэкенд их не подключает) | Roboflow Platform Model License 1.0 | на платформе Roboflow; своё размещение - только с Enterprise-лицензией Roboflow |
| RF-DETR Nano в ONNX (`onnx-community/rfdetr_nano-ONNX`, `onnx`/`rfdetr-nano`) | Apache-2.0 | да |
| RT-DETR R18 и RT-DETRv2 R18 в ONNX (`onnx-community/rtdetr_r18vd`, `onnx-community/rtdetr_v2_r18vd-ONNX` из `PekingU/rtdetr_r18vd`, `PekingU/rtdetr_v2_r18vd`; `onnx`/`rtdetr-r18`, `rtdetrv2-r18`) | Apache-2.0 | да |
| RT-DETR, RT-DETRv2 из репозитория авторов (`lyuwenyu/RT-DETR`), экспорт в ONNX | Apache-2.0 | да |
| YOLO, экспортированные в ONNX из `ultralytics` | AGPL-3.0, как исходные веса | как у `ultralytics` |
| ONNX Runtime (библиотека бэкенда `onnx`) | MIT | да |
| Grounding DINO Tiny и Base (`IDEA-Research/GroundingDINO`, `IDEA-Research/grounding-dino-{tiny,base}`, `transformers`) | Apache-2.0 | да |
| PyTorch, torchvision (extras `ultralytics`, `rfdetr`, `grounding-dino`) | BSD-3-Clause | да |

Бэкенды под AGPL и GPL - необязательные extras (NFR-13): `rfdetr` и `onnx` с моделями под Apache-2.0 работают без них. В опубликованные образы входит код всех бэкендов, включая AGPL (исходники открыты); веса в образы не входят.

## Запуск

Порт на хосте - `127.0.0.1:8101`, сеть Docker `alv-net` compose создаёт сам. Каталог `./model` монтируется в `/opt/model` под веса.

На CPU:

```bash
cp .env.example .env
docker compose -f compose-cpu.yml up -d --build
curl http://localhost:8101/health
```

На GPU (нужен NVIDIA Container Toolkit; в `.env` - `TARGET_PLATFORM=cuda`):

```bash
docker compose up -d --build
```

Аргументы сборки берутся из окружения: `GIT_SHA=$(git rev-parse --short HEAD)` попадает в `/v1/info`, `EXTRAS` задаёт extras образа. По умолчанию - профиль и все бэкенды: в `Dockerfile.cpu` - `cpu ultralytics rfdetr onnx grounding-dino`, в `Dockerfile` - `gpu ultralytics rfdetr grounding-dino` (`onnx` там берёт `onnxruntime-gpu` профиля). `EXTRAS=""` - лёгкий образ только с `fake` для сквозной проверки. Образ на CPU со всеми бэкендами - около 3,4 ГБ на диске (766 МБ сжатых слоёв), сборка без кэша - около 7 минут.

Локально:

```bash
uv sync
uv run fastapi dev app/main.py
uv run pytest -q -m "not model"
```

Бэкенды - extras: на CPU `uv sync --extra cpu --extra onnx --extra rfdetr` (любой набор из `ultralytics`, `rfdetr`, `onnx`, `grounding-dino`), на GPU - `uv sync --no-dev --extra gpu --extra ultralytics`. Профили `cpu` и `gpu` взаимоисключающие, `onnx` и группа `dev` несут onnxruntime для CPU и с `gpu` не ставятся.

Тесты с маркером `model` нужны весам и в быстрый прогон не входят: `uv run pytest tests/test_model_smoke.py -q -m model` проверяет, что nano-модели на CPU находят человека, стул или стол на снимках `tests/data/smoke/` (источники - в `SOURCES.md` там же). Всегда идёт `onnx`/`rfdetr-nano` - onnxruntime есть в группе `dev`; `rfdetr`/`rfdetr-nano` и `ultralytics`/`yolo11n` - если их extra установлен. Веса скачиваются при первом прогоне в `~/.cache/objects-detector`. CI (GitHub Actions) прогоняет быстрые тесты на каждый пуш и PR, а на пуш в `dev`, `main` и тег `v*` публикует образы в `ghcr.io/nikarpoff/objects-detector` с тегами `latest` (ветка `main`), `sha-<7>` и `v<semver>` и суффиксом варианта: `latest-cpu`, `v0.1.0-cuda`. На `dev` собирается только `-cpu`.

### Устройство

```
app/
├── main.py                 # настройки -> логирование -> FastAPI(lifespan) -> /health -> middleware -> роутеры
├── api_v1/
│   ├── views.py            # /v1/info, /v1/vocabulary, /v1/detect, зависимость get_engine
│   ├── schemas.py          # контракт API (pydantic)
│   └── errors.py           # AppError, 422 и 500 -> {"detail": ...}
├── core/
│   ├── config/             # Settings (общие и детектора), версия из pyproject.toml, константы
│   ├── auth/               # Bearer-токен
│   ├── backends/           # реестр, базовый класс (decode -> detect -> postprocess), fake, ultralytics, rfdetr, onnx, grounding_dino
│   ├── detection/          # снимок (image), letterbox, словарь (vocabulary), направление и расстояние (geometry), классы модели -> словарь (mapping)
│   ├── engine.py           # бэкенд в пуле WORKERS, очередь QUEUE_MAX, прогрев
│   └── exceptions.py       # доменные ошибки со status_code
└── utils/                  # lifespan, логирование, middleware X-Request-ID и Server-Timing
bench/
├── latency.py              # замер задержки в процессе (SVC-10, EXP-2)
└── eval.py                 # оценка по разметке COCO (DET-7)
configs/vocabulary.yaml     # словарь классов (DET-2)
scripts/fetch_weights.py    # веса текущей модели заранее (DEP-6)
tests/
├── contract/               # эталонные примеры из Шлюза; правятся только синхронизацией
├── data/eval/              # игрушечная разметка для test_eval
├── data/smoke/             # снимки для test_model_smoke (CC0)
└── test_*.py
```

## Замеры

Время этапов (`decode`, `inference`, `postprocess`) - в заголовке `Server-Timing` каждого ответа и в поле `timings` ответа обнаружения.

Задержка модели в процессе, без HTTP и сети (SVC-10): прогрев, затем `--n` вызовов на одном снимке (по умолчанию синтетический JPEG 640x480, свой - `--image`). Меряется весь путь запроса внутри сервиса: декодирование, инференс и постобработка. Перцентили - nearest-rank, стандартное отклонение - выборочное, как у Шлюза.

```bash
uv run python -m bench.latency --backend fake --n 5 --warmup 1
uv run python -m bench.latency --backend onnx --model yolo11n --device cpu --n 50 --warmup 5 --out latency.json
```

```json
{"service": "objects-detector", "backend": "onnx", "model": "yolo11n", "device": "cpu", "git_sha": "3f2c1ab", "n": 50, "warmup": 5, "p50_ms": 61.2, "p95_ms": 74.9, "mean_ms": 63.0, "std_ms": 5.1, "hardware": {"cpu": "AMD EPYC 7302", "gpu": null}, "started_at": "2026-11-01T10:00:00Z"}
```

Качество (DET-7) - по разметке COCO с атрибутами EXP-1 (`direction`, `distance` у аннотации, категории - идентификаторы словаря): AP50 по каждому ключевому классу и mAP50 - их среднее по классам с разметкой, полнота по ключевым классам при рабочем `--conf`, точность направления и расстояния для верно найденных объектов (IoU не меньше 0.5). Предсказания - от бэкенда в процессе (снимки по `file_name` рядом с разметкой) или из файла `--predictions`.

```bash
uv run python -m bench.eval --annotations dataset/annotations.json --backend onnx --model yolo11n --out eval.json
uv run python -m bench.eval --annotations tests/data/eval/annotations.json --predictions tests/data/eval/predictions.json
```

```json
{"service": "objects-detector", "predictions": "tests/data/eval/predictions.json", "vocabulary_hash": "8778f20c71d0", "images": 2, "annotations": 5, "iou": 0.5, "conf": 0.35, "map50": 0.6117, "ap50": {"person": 0.835, "door": 1.0, "chair": 0.0}, "recall": {"person": 0.5, "door": 1.0, "chair": 0.0}, "direction_accuracy": 0.5, "distance_accuracy": 1.0, "matched": 2, "started_at": "2026-11-02T10:00:00Z"}
```

Сами замеры и сравнение моделей - задача `002-hypothesis-experiment`.
