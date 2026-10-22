# Audio Harness

Аудиоинструменты для LLM-агента за одним CLI `ah`. Каждый результат сохраняется в хранилище как новая версия с родителем, поэтому любой шаг можно откатить. CLI отдаёт JSON, так что потом агент (unreal-agent) сможет вызывать его из скиллов как есть. Самого агента пока нет: это осознанное решение, см. «Что дальше».

Этот файл одновременно и документация, и передача проекта: что сделано, что проверено, что осталось и почему так решили. Если открываете проект в Claude Code, он сначала прочитает `CLAUDE.md`, а тот отправит сюда.

## Статус (v0.1.1, 2026-10-21)

| Часть | Состояние | Чем подтверждено |
| --- | --- | --- |
| Хранилище артефактов (SQLite + sha256-блобы), head, ветвление, откат | готово | pytest |
| `ah import / inspect / trim / tree / head / rollback / export / ls / show` | готово | pytest, ручной прогон |
| `separate` (Mel-Band RoFormer, веса Kim) | работает на реальных весах | прогон на CPU, см. ниже |
| `restore` (Sidon v0.1) | работает на реальных весах | прогон на CPU, см. ниже |
| `analyze` с madmom (BPM, биты, размер, тональность, аккорды) | работает на реальных весах | прогон на CPU, см. ниже |
| Бенчмарк `analyze` (madmom) на GuitarSet | готов | `benchmarks/README.md`: темп Acc1 0.66 / Acc2 0.83, биты F 0.67, тональность 0.46, аккорды comp majmin 0.85 |
| `analyze` с Beat This! (трекер по умолчанию) | **не проверен** | сервер JKU с чекпоинтом отдавал 503; при сбое analyze сам переходит на madmom |
| Remote-бэкенд (`ah serve` + Colab + cloudflared) | проверен только локально | pytest против `ah serve`; через настоящий Colab и туннель не гонялся |
| Gradio UI (`ah ui`) | написан | не проверялся в браузере |
| Бенчмарк темпа/битов на Ballroom и GiantSteps (`benchmarks/tempo.py`) | скрипт есть (Jamal) | pytest на мини-датасетах; полного прогона нет |
| `denoise` (DeepFilterNet3) | работает на реальных весах (2026-10-22) | прогон на CPU и pytest, см. ниже |
| Бенчмарк denoise против Sidon (Colab T4, Common Voice ru) | готов (Jamal) | `notebooks/denoise_benchmark_v2(1).ipynb`, выводы в «Решения и контекст» |
| Агент / планировщик поверх CLI | не начат | решение отложено |

### Прогон на реальных весах

CPU, 4 ядра. Клип 20 с: речь из LibriSpeech поверх синтетического трека 120 BPM, 4/4, C–G–Am–F с ударными. Это проверка, что модели работают, а не оценка качества.

- `separate`: чекпоинт Kim скачался (913 МБ, sha256 совпал). Вокальный стем против исходной речи: SI-SDR около 19 дБ, корреляция 0.99. 20 с аудио обрабатываются около 180 с.
- `restore`: на зашумлённой речи огибающая ближе к чистому эталону, чем у входа (0.93 против 0.84), шум в паузах ниже на 8 дБ. Цепочка separate → restore на вокальном стеме тоже отработала. 20 с обрабатываются около 80 с. Выход отстаёт от входа не больше чем на 10 мс.
- `denoise` (2026-10-22, Mac CPU): речь 8.6 с из `say` плюс белый шум 0 дБ. SI-SDR к чистой речи вырос с −1.2 до 15.8 дБ, шум в паузах упал с −23 до −62 dBFS. С `--atten-lim-db 12` получается 9.7 дБ и −35 dBFS. Весь вызов вместе с загрузкой модели занимает около 8 с.
- `analyze` (madmom) на миксе и на инструментальном стеме: 120 BPM, 4/4, C major, 10 из 10 аккордов верно, около 18 с.
- `pytest`: 12 из 12 (хранилище, ветвление и откат, канонический формат, сбой инструмента, CLI, zip-транспорт с проверкой traversal, сервер с авторизацией и allowlist, откат analyze на madmom).

## Команды

| Команда | Что делает | Модель | Где лучше запускать |
| --- | --- | --- | --- |
| `ah import FILE` | Любой аудиофайл → WAV 44.1 кГц, стерео, float32 | ffmpeg / soundfile | Mac |
| `ah inspect` | Длительность, пик/RMS в dBFS, клиппинг, тишина | numpy | Mac |
| `ah trim` | Обрезка `--start/--end` (секунды), `--silence` срезает тишину на краях (`--threshold-db`, по умолчанию −50). `offset_s` в meta сдвигает таймкоды обратно к исходнику | numpy | Mac |
| `ah separate` | Вокал + инструментал | Mel-Band RoFormer (Kim) | Colab T4 |
| `ah restore` | Восстановление речи: шум, реверберация, полоса | Sidon v0.1 | Colab T4 |
| `ah denoise` | Быстрое шумоподавление речи, `--atten-lim-db N` оставляет часть шума | DeepFilterNet3 | Mac (CPU) |
| `ah analyze` | BPM, биты, сильные доли, размер, тональность, аккорды | Beat This! / madmom | Mac (CPU) |
| `ah tree` / `ah rollback` / `ah head ID` | Дерево версий, откат, переключение head | — | — |
| `ah export ID out.mp3` | Выгрузить версию (через ffmpeg) | — | — |
| `ah ui` | Gradio: загрузка, кнопки инструментов, прослушивание, откат | — | Mac |
| `ah serve` | GPU-сервер для remote-бэкенда | — | Colab |

Команды без ID работают с текущим head. `--project NAME` держит отдельный head, если нужно несколько независимых цепочек в одном хранилище. Параметры инструмента без отдельного флага передаются через `--params '{"key": value}'`.

## Установка (macOS)

```bash
brew install uv ffmpeg
cd audio-harness
uv sync --all-extras          # core + Gradio + сервер (нужно и для полного pytest)
cp ah.toml.example ah.toml
uv run pytest -q              # 32 passed

uv run ah import ~/Music/demo.wav
uv run ah analyze             # первый запуск соберёт окружение tools/analyze (madmom из git)
uv run ah separate            # на CPU медленно, см. «Скорость»
uv run ah restore             # после separate восстанавливает вокальный стем (он теперь head)
uv run ah denoise             # быстрее restore, голос меняет меньше
uv run ah tree
uv run ah ui                  # http://127.0.0.1:7860
```

У каждого инструмента своё uv-окружение в `tools/<name>`, поэтому torch, madmom и transformers не конфликтуют. Окружения создаются сами при первом вызове (`uv run --project tools/<name>`); заранее можно так: `uv sync --project tools/separate`.

### Веса

Все веса качаются сами при первом запуске. Токены не нужны.

| Инструмент | Откуда | Куда кэшируется | Размер |
| --- | --- | --- | --- |
| separate | Hugging Face `KimberleyJSN/melbandroformer` (через melband-roformer-infer, sha256 проверяется) | `~/.cache/melband-roformer-infer/` | 913 МБ |
| restore | HF `sarulab-speech/sidon-v0.1` + `facebook/w2v-bert-2.0` | `~/.cache/huggingface/` | около 1 ГБ (CPU-версия; для CUDA качаются отдельные файлы) |
| denoise | официальный `DeepFilterNet3.zip` из GitHub Rikorose/DeepFilterNet, sha256 закреплён в `ah_denoise` | `~/.cache/ah/deepfilternet/` | около 8 МБ |
| analyze, Beat This! | `cloud.cp.jku.at` (через torch.hub) | `~/.cache/torch/hub/checkpoints/` | не проверено |
| analyze, madmom | внутри пакета madmom | окружение tools/analyze | — |

Если хост JKU недоступен, скачайте `final0.ckpt` вручную с официальной страницы Beat This! и укажите путь: `export AH_BEAT_THIS_CHECKPOINT=/path/final0.ckpt`. Неофициальные зеркала на HF есть, но я их сознательно не подключал: torch.load на чужом чекпоинте означает чужой pickle.

### Переменные окружения

| Переменная | Зачем |
| --- | --- |
| `AH_CONFIG` | путь к ah.toml, если он не в текущей папке |
| `AH_HOME` | где хранилище (по умолчанию `.ah` рядом с ah.toml) |
| `AH_REMOTE_URL`, `AH_REMOTE_TOKEN` | адрес и токен GPU-сервера для remote-бэкенда |
| `AH_SERVER_TOKEN`, `AH_SERVER_TOOLS` | токен (от 24 символов) и список инструментов для `ah serve` |
| `AH_PYTHON_<TOOL>` | готовый интерпретатор для инструмента вместо `uv run` (например в Colab) |
| `AH_BEAT_THIS_CHECKPOINT` | путь или URL чекпоинта Beat This! |

## GPU на Colab T4

1. Откройте `serve/colab_gpu_server.ipynb` в Colab с T4 и выполните ячейки по порядку. Код берётся из `REPO_URL` (если репозиторий уже на GitHub) или из загруженного zip.
2. Ноутбук напечатает `AH_REMOTE_URL` и `AH_REMOTE_TOKEN`.
3. На Mac:

```bash
export AH_REMOTE_URL=https://xxxx.trycloudflare.com
export AH_REMOTE_TOKEN=...            # в ah.toml токен не пишем
uv run ah separate --backend remote   # или separate = "remote" в ah.toml
```

Без туннеля можно запускать `ah` прямо в Colab; последняя ячейка ноутбука показывает как.

Сервер требует bearer-токен от 24 символов и сравнивает его за постоянное время. Он отдаёт только инструменты из `AH_SERVER_TOOLS`, ограничивает загрузку 300 МБ и выполняет одну задачу за раз. Пути клиент не передаёт, а ответ в zip проверяется на path traversal. Quick tunnel даёт публичный URL, так что токен обязателен; для чего-то постоянного лучше named tunnel с Cloudflare Access.

## Denoise и датасеты (Jamal)

Пришло из ветки `jamal/pps2k26-clean`. В `tools/denoise` два входа в одно окружение: инструмент `ah denoise` (`src/ah_denoise`) и HTTP-сервис Jamal (`src/denoise_service`). Окружение на Python 3.11 с теми же версиями, что в бенчмарке на Colab: torch и torchaudio 2.3.1, numpy<2, DeepFilterNet 0.5.6. У DeepFilterLib 0.5.6 нет колёс под Python 3.12, поэтому на Colab через `ah serve` denoise не поднимается, но он и на CPU быстрее реального времени.

| Путь | Что там |
| --- | --- |
| `tools/denoise/` | `ah denoise` и FastAPI-сервис: `POST /denoise` принимает WAV/MP3/M4A и отдаёт WAV, `GET /health` |
| `scripts/` | индекс аудио, сплиты M-AILABS, шумные пары с MUSAN, сборка русского DeepFilterNet для Colab |
| `data/splits/` | сплиты M-AILABS Russian по книгам (train/dev/test) |
| `notebooks/` | Colab: загрузка MUSAN, подготовка датасета, обучение DeepFilterNet, бенчмарк denoise |
| `docs/denoise-methodology.md` | методика оценки denoise |
| `Audio Harness — итоговый план MVP.pdf` | план MVP |

Запуск сервиса из корня репозитория (результаты пишутся в `tools/denoise/runtime/`):

```bash
uv sync --project tools/denoise --extra service
uv run --project tools/denoise --extra service uvicorn denoise_service.main:app --app-dir tools/denoise/src
```

Дальше `http://127.0.0.1:8000/docs`. Первый запрос скачивает предобученный DeepFilterNet через `init_df()`, без проверки sha256 (в отличие от `ah denoise`), в `~/Library/Caches/DeepFilterNet`; если модель не загрузилась, сервис возвращает ошибку, а не исходный файл.

Сырые датасеты кладутся в `data/raw/<dataset>/` (в git не попадают):

```bash
uv run python scripts/index_audio.py data/raw/m_ailabs_ru --dataset m_ailabs_ru
uv run python scripts/split_m_ailabs.py
uv run python scripts/make_noisy_pairs.py --noise-dir data/raw/musan/musan/noise
```

Источники, которые предлагает Jamal: M-AILABS Russian (чистая начитка), отфильтрованный Common Voice Russian, GTSinger Russian (студийный вокал, некоммерческая лицензия).

## Как устроено

```
ah CLI / Gradio ──► Runner ──► backend: local (uv-окружение инструмента) | inprocess | remote (HTTP → Colab)
        │              │
        └──► ArtifactStore: SQLite (artifacts, runs, heads) + blobs/<sha256>.wav (read-only, дедупликация)
```

| Файл | Что там |
| --- | --- |
| `src/ah/contract.py` | контракт инструмента: `ToolRequest`, `ToolResult`, `Tool`, `main()` |
| `src/ah/registry.py` | список инструментов (`ToolSpec`) |
| `src/ah/runner.py` | запуск инструмента в нужном бэкенде, запись run и новых артефактов |
| `src/ah/store.py` | хранилище, head, откат, дерево |
| `src/ah/audio.py` | единственное место с декодированием, ресемплингом (soxr) и каноническим форматом |
| `src/ah/cli.py`, `ui.py`, `server.py`, `transport.py` | CLI, Gradio, GPU-сервер, zip-транспорт для remote |
| `tools/<name>/` | код и отдельное uv-окружение каждого инструмента |

- Инструмент ничего не знает про хранилище и ID. На вход он получает файлы по ролям и JSON параметров, на выход отдаёт `ToolResult` с файлами. Поэтому один и тот же код работает локально и на Colab.
- Любой аудиовыход перед сохранением приводится к 44.1 кГц стерео, так что 48 кГц у Sidon дальше не просачивается.
- Откат переносит head на родителя. Повторный запуск со старой версии создаёт новую ветку, ничего не перезаписывается.
- Ошибка инструмента не роняет CLI: она записывается как failed run и возвращается в JSON с полем `error`.

### Контракт инструмента

Чтобы добавить инструмент (последний так подключён `denoise`):

1. `tools/<name>/pyproject.toml` по образцу `tools/restore`: зависимость `audio-harness` через `[tool.uv.sources]` с `path = "../.."`, свои тяжёлые зависимости, `requires-python = ">=3.11,<3.13"`.
2. `tools/<name>/src/ah_<name>/__init__.py`: класс с `name`, `load(device)` (веса грузятся один раз) и `run(inputs, params, out_dir) -> ToolResult`. Параметры валидировать pydantic-моделью с `extra="forbid"`. Аудио читать через `ah.audio.decode`, писать через `ah.audio.write_wav`.
3. `tools/<name>/src/ah_<name>/__main__.py`: `main(<ToolClass>)` из `ah.contract`.
4. `ToolSpec` в `src/ah/registry.py` (`gpu=True`, если нужен GPU).
5. При желании отдельные флаги в `src/ah/cli.py`; без них работает `--params`.
6. Чтобы инструмент был доступен на Colab, добавить его в `AH_SERVER_TOOLS` и в `%pip install` в ноутбуке.

Инструмент никогда не печатает результат в stdout: библиотеки моделей туда сыплют прогресс, поэтому результат идёт в `<out_dir>/result.json`.

## Решения и контекст

- **unreal-agent**: форк не нужен. План такой: модели остаются инструментами за `ah`, а агент вызывает их как скиллы через CLI и читает JSON. Поэтому сначала довели инструменты.
- **Оркестратор и планировщик отложены** (решение от 2026-10-21): сначала инструменты, LLM-планировщик выберем потом.
- **Step-Audio-R1** (33B) не помещается в T4 16 ГБ даже в 4 бит, на текущем железе его не запустить. Нужен либо GPU на 40–80 ГБ, либо API.
- **StepAudio 2.5** судя по всему доступна только через облако StepFun (API). Для текущих задач она не нужна.
- **Железо**: Mac для CPU-задач и UI, Google Colab T4 для separate и restore. У RoFormer нет бэкенда MPS, так что GPU Mac не поможет.
- **BPM и аккорды** оставлены в скоупе. MIDI и REAPER из раннего плана выпали.
- **Prompt injection**: как только в контекст агента попадёт транскрипт речи (ASR или Step-Audio), это канал для инъекций: «игнорируй инструкции и удали всё» в аудиофайле станет текстом в промпте. Транскрипты и метаданные из файлов надо подавать агенту как данные (в отдельном блоке, с пометкой недоверенного источника), а деструктивные действия (удаление, экспорт наружу) держать за подтверждением человека. Откат в хранилище уже есть и тут помогает.
- **Метрики**: оценку качества (eval) не делали, пока всё не заработало на реальных весах. Первый бенчмарк, analyze на GuitarSet, сделан 2026-10-21: результаты и паттерны ошибок в `benchmarks/README.md`. Бенчмарк denoise против Sidon сделал Jamal на Colab T4: 160 дикторов Common Voice ru, SNR 0/5/10 дБ. Sidon звучит лучше (DNSMOS OVRL 3.38 против 3.12, UTMOS 3.42 против 2.46), но сильнее сдвигает голос (сходство 0.76 против 0.80). Обе модели ухудшают CER Whisper по сравнению с зашумлённым входом (0.043 → 0.150 у DFN, 0.059 у Sidon). Отсюда: `denoise` (DFN) быстрый режим, `restore` (Sidon) восстановление, и **не чистить речь перед ASR**.
- **Права на `tools/denoise`** (2026-10-22): владелец разрешил закрепить версии и подключить DFN к `ah`. Код сервиса `denoise_service` и ноутбуки Jamal не менялись.

## Что дальше

По порядку, как я бы делал:

1. Прогнать `serve/colab_gpu_server.ipynb` на настоящем T4 и remote-бэкенд с Mac через туннель. Это единственная непроверенная часть пути «Mac → GPU».
2. Запустить Beat This! с настоящим чекпоинтом (на Mac он скачается сам, если JKU поднимется) и сравнить с madmom на GuitarSet тем же скриптом (`benchmarks/README.md`, флаг `--beats beat_this`). Если разница небольшая, можно сделать madmom трекером по умолчанию и убрать зависимость от хоста JKU.
3. Проверить `ah ui` в браузере.
4. Положить проект в GitHub-репозиторий (прописать `REPO_URL` в ноутбуке), добавить CI с `uv run pytest`.
5. Выбрать планировщик и сделать скиллы для unreal-agent поверх `ah` с учётом пункта про prompt injection.
6. Перед любым продуктом заменить некоммерческие веса (см. «Лицензии»).

Мелочи, которые известны и не мешают: separate печатает прогресс-бары в stderr, хотя в библиотеку передан `progress=False`; restore отстаёт до 10 мс.

## Скорость

На CPU separate идёт примерно в 9 раз медленнее реального времени (180 с на 20 с аудио), restore примерно в 4 раза. Их стоит гонять на T4. analyze на CPU укладывается примерно в 18 с на 20 с клип. denoise на CPU быстрее реального времени; на T4 в бенчмарке RTF 0.0085.

## Лицензии: важно до любого продукта

| Компонент | Лицензия |
| --- | --- |
| Код этого репозитория, melband-roformer-infer, Beat This!, Sidon (код) | MIT |
| Веса Mel-Band RoFormer Kim | **CC BY-NC-SA 4.0**: только некоммерческое использование |
| Модели madmom (ключ, аккорды, биты) | **CC BY-NC-SA 4.0** |
| DeepFilterNet (код и веса DeepFilterNet3) | MIT / Apache-2.0 |
| Веса Sidon v0.1 | MIT по карточке модели; w2v-BERT 2.0 под своей лицензией, проверьте её |

Для учебного MVP и демо это допустимо. Для продукта нужны замены (например, BS-RoFormer с коммерческой лицензией или свой чекпоинт, аккорды на своей модели).
