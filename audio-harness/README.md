# Audio Harness

Инструменты для аудио за одним CLI `ah`. Каждый результат сохраняется в хранилище как новая версия с родителем, поэтому откатить можно любой шаг. Агента (unreal-agent) поверх этого пока нет: CLI отдаёт JSON, и потом его можно будет вызывать из скиллов как есть.

| Команда | Что делает | Модель | Где лучше запускать |
| --- | --- | --- | --- |
| `ah import FILE` | Любой аудиофайл → WAV 44.1 кГц, стерео, float32 | ffmpeg / soundfile | Mac |
| `ah inspect` | Длительность, пик/RMS в dBFS, клиппинг, тишина | numpy | Mac |
| `ah separate` | Вокал + инструментал | Mel-Band RoFormer (Kim) | Colab T4 |
| `ah restore` | Восстановление речи: шум, реверберация, полоса | Sidon v0.1 | Colab T4 |
| `ah analyze` | BPM, биты, сильные доли, размер, тональность, аккорды | Beat This! + madmom | Mac (CPU) |
| `ah tree` / `ah rollback` / `ah head ID` | Дерево версий, откат, переключение head | — | — |
| `ah export ID out.mp3` | Выгрузить версию (через ffmpeg) | — | — |
| `ah ui` | Gradio-страница: загрузка, кнопки инструментов, прослушивание, откат | — | Mac |

## Быстрый старт (macOS)

```bash
brew install uv ffmpeg
cd audio-harness
uv sync --extra ui            # core + Gradio
cp ah.toml.example ah.toml

uv run ah import ~/Music/demo.wav
uv run ah analyze             # первый запуск соберёт окружение tools/analyze (madmom из git)
uv run ah separate            # на CPU медленно, см. «Скорость»
uv run ah restore             # после separate восстанавливает вокальный стем (он теперь head)
uv run ah tree
uv run ah ui                  # http://127.0.0.1:7860
```

Команды без ID работают с текущим head. У каждого инструмента своё uv-окружение в `tools/<name>`, поэтому torch, madmom и transformers не конфликтуют между собой. Окружения создаются сами при первом вызове (`uv run --project tools/<name>`).

## Проверка separate на Windows

В Python 3.11 создайте окружение и установите core вместе с закреплённой
версией инструмента:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e . -e .\tools\separate
.\.venv\Scripts\ah.exe import .\demo.wav
.\.venv\Scripts\ah.exe separate --backend inprocess
.\.venv\Scripts\ah.exe tree
```

На Windows `imageio-ffmpeg` ставит FFmpeg вместе с зависимостями проекта: MP3,
M4A, AAC и другие поддерживаемые контейнеры импортируются без отдельной установки
в систему. `ah export ID out.mp3` также использует этот FFmpeg.

Первый запуск скачивает чекпоинт Kim (~913 МБ) и проверяет SHA-256. Результат
должен содержать `vocals` и `instrumental` одинаковой с входом длительности;
`vocals` становится head. `--backend inprocess` позволяет проверить инструмент
без установки `uv`. PyTorch из стандартного PyPI может оказаться CPU-сборкой:
проверьте `.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"`.

## GPU на Colab T4

1. Откройте `serve/colab_gpu_server.ipynb` в Colab с T4 и выполните ячейки по порядку.
2. Ноутбук напечатает `AH_REMOTE_URL` и `AH_REMOTE_TOKEN`.
3. На Mac:

```bash
export AH_REMOTE_URL=https://xxxx.trycloudflare.com
export AH_REMOTE_TOKEN=...            # в ah.toml токен не пишем
uv run ah separate --backend remote   # или separate = "remote" в ah.toml
```

Без туннеля можно запускать `ah` прямо в Colab; последняя ячейка ноутбука показывает как.

Сервер (`ah serve`) требует bearer-токен длиной от 24 символов и сравнивает его за постоянное время. Он отдаёт только инструменты из `AH_SERVER_TOOLS`, ограничивает размер загрузки 300 МБ и выполняет одну задачу за раз. Пути клиент не передаёт, а ответ в zip проверяется на path traversal.

## Как устроено

```
ah CLI / Gradio ──► Runner ──► backend: local (uv-окружение инструмента) | inprocess | remote (HTTP → Colab)
        │              │
        └──► ArtifactStore: SQLite (artifacts, runs, heads) + blobs/<sha256>.wav (read-only, дедупликация)
```

- Контракт инструмента описан в `src/ah/contract.py`: на вход файлы по ролям и JSON параметров, на выход `ToolResult` с файлами. Инструмент ничего не знает про хранилище и ID, поэтому одинаково работает локально и на Colab.
- Ресемплинг делается только в `src/ah/audio.py` (soxr). Любой аудиовыход перед сохранением приводится к 44.1 кГц стерео, так что 48 кГц у Sidon не просочится дальше.
- Откат переносит head на родителя. Повторный запуск с более старой версии создаёт новую ветку, и ничего не перезаписывается.

## Что проверено, а что нет

- `pytest`: 11 тестов на хранилище, ветвление и откат, канонический формат, сбой инструмента, CLI, zip-транспорт (включая traversal) и сервер (авторизация, allowlist, прогон).
- `analyze` на синтетическом треке (120 BPM, 4/4, C–G–Am–F) с битами от madmom выдал 120.0 BPM, 4/4, C major и 16 из 16 аккордов верно.
- `separate` прогнан со случайными весами: проверены I/O, частоты и манифест стемов. Настоящий чекпоинт Kim не скачивался, потому что из среды разработки huggingface.co закрыт. 32 с аудио на CPU этой машины заняли 153 с.
- `restore` прогнан с заглушками той же формы: проверены чанки, кэш кадра и длина выхода 48 кГц. Сами веса Sidon не запускались. Код повторяет официальный демо-Space `sarulab-speech/sidon_demo_beta`.
- Beat This! с реальным чекпоинтом не запускался, потому что хост JKU тоже закрыт. На Mac он скачается сам; если что-то не так, используйте `--beats madmom`.
- Remote-бэкенд проверен локально против `ah serve`. Через cloudflared и Colab не проверялся.

## Скорость

На CPU separate работает медленнее реального времени (см. выше), поэтому его стоит гонять на T4. analyze на CPU укладывается примерно в 25 с на 32 с клип с madmom. У RoFormer нет бэкенда MPS, так что GPU Mac не поможет.

## Лицензии: важно до любого продукта

| Компонент | Лицензия |
| --- | --- |
| Код этого репозитория, melband-roformer-infer, Beat This!, Sidon (код) | MIT |
| Веса Mel-Band RoFormer Kim | **CC BY-NC-SA 4.0**: только некоммерческое использование |
| Модели madmom (ключ, аккорды, биты) | **CC BY-NC-SA 4.0** |
| Веса Sidon v0.1 | MIT по карточке модели; w2v-BERT 2.0 под своей лицензией, проверьте её |

Для учебного MVP и демо это допустимо. Для продукта нужны замены (например, BS-RoFormer с коммерческой лицензией или свой чекпоинт, аккорды на своей модели).
