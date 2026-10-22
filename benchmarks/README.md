# Бенчмарки

## analyze на GuitarSet (2026-10-21)

**Что гоняли.** `ah_analyze.AnalyzeTool` как есть (тот же код, что за `ah analyze`), трекер битов madmom. Beat This! не проверен: хост чекпоинта `cloud.cp.jku.at` в день прогона отдавал 503 на стороне апстрима. Датасет GuitarSet 1.1.0, все 360 отрывков, микрофонная запись (`audio_mono-mic`), около 3 ч аудио. Метрики считает `mir_eval` 0.8.2. CPU, 4 процесса, около 35 минут wall time; один процесс работает примерно в 0.77 от реального времени.

**Лицензия датасета.** GuitarSet распространяется под CC BY 4.0 (Zenodo 3371780), можно и в коммерческих целях, нужна ссылка на Xi et al., ISMIR 2018. В репозиторий кладём только метрики (`results/`), аудио не кладём. Модели madmom по-прежнему CC BY-NC-SA 4.0. Утечки нет: модели madmom обучены до выхода GuitarSet (2018).

**Как читать.** GuitarSet это соло-гитара без барабанов и баса: comp (аккомпанемент) и solo (мелодия поверх подложки, которой в записи нет). Без ритм-секции это тяжёлый случай для любого трекера, так что это не оценка madmom «вообще», а оценка того, как наш analyze поведёт себя на гитарной записи. Темп, биты и тональность берутся из задания игроку (метроном и лид-шит). Аккорды comp сравниваются с транскрипцией сыгранного. Аккорды solo в таблице для полноты: в аудио нет аккомпанемента, так что это не оценка модели аккордов.

### Сводка

| Метрика | Все (360) | comp (180) | solo (180) |
| --- | ---: | ---: | ---: |
| Темп Acc1 (±4%) | 0.66 [0.54, 0.76] | 0.72 [0.60, 0.83] | 0.59 [0.42, 0.76] |
| Темп Acc2 (также ×2, ×½, ×3, ×⅓) | 0.83 [0.73, 0.92] | 0.87 [0.76, 0.97] | 0.78 [0.61, 0.92] |
| Биты F-measure | 0.67 [0.57, 0.76] | 0.76 [0.66, 0.85] | 0.58 [0.43, 0.71] |
| Биты AMLt | 0.76 [0.66, 0.85] | 0.84 [0.74, 0.93] | 0.67 [0.51, 0.81] |
| Сильные доли F-measure | 0.41 [0.33, 0.49] | 0.51 [0.38, 0.63] | 0.31 [0.18, 0.44] |
| Тональность, точно | 0.46 [0.35, 0.57] | 0.55 [0.41, 0.68] | 0.37 [0.24, 0.50] |
| Тональность, MIREX weighted | 0.57 [0.47, 0.66] | 0.65 [0.53, 0.76] | 0.49 [0.37, 0.59] |
| Аккорды root | | **0.81 [0.72, 0.89]** | 0.16 [0.13, 0.19] |
| Аккорды majmin | | **0.85 [0.79, 0.90]** | 0.15 [0.11, 0.20] |
| Аккорды mirex | | **0.72 [0.63, 0.80]** | 0.12 [0.09, 0.16] |

В скобках 95% доверительный интервал. Считается двусторонним бутстрэпом (2000 повторов, seed 0): исполнители и лид-шиты ресемплятся независимо, потому что каждый трек делит исполнителя с 59 другими и лид-шит с 11 другими, и бутстрэп по трекам дал бы слишком узкие интервалы. Исполнителей всего 6, так что интервалы приблизительные; они показывают, насколько цифры сдвинутся на других гитаристах и других песнях. Аккорды взвешены по длительности на всём датасете, как в MIREX. majmin не учитывает аккорды, которые не сводятся к мажору или минору (dim, hdim7, sus).

### comp по стилям

| Стиль | Acc1 | Acc2 | Биты F | Сильные доли F | Тон. точно | Аккорды majmin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bossa nova | 0.86 [0.58, 1.00] | 0.97 [0.83, 1.00] | 0.69 [0.38, 0.94] | 0.55 [0.25, 0.82] | 0.58 [0.30, 0.86] | 0.86 [0.76, 0.93] |
| funk | 0.61 [0.28, 0.92] | 0.78 [0.44, 1.00] | 0.72 [0.51, 0.91] | 0.44 [0.17, 0.74] | 0.44 [0.17, 0.72] | 0.66 [0.51, 0.82] |
| jazz | 0.72 [0.50, 0.92] | 0.94 [0.81, 1.00] | 0.82 [0.68, 0.95] | 0.38 [0.17, 0.55] | 0.42 [0.14, 0.78] | 0.82 [0.72, 0.90] |
| rock | 0.81 [0.56, 1.00] | 0.89 [0.67, 1.00] | 0.81 [0.62, 0.97] | 0.48 [0.25, 0.75] | 0.56 [0.28, 0.83] | 0.93 [0.90, 0.97] |
| singer-songwriter | 0.58 [0.33, 0.83] | 0.75 [0.47, 0.97] | 0.75 [0.58, 0.90] | 0.68 [0.51, 0.84] | 0.75 [0.50, 0.94] | 0.93 [0.88, 0.96] |

В стиле всего 6 лид-шитов, поэтому интервалы широкие и почти все различия между стилями в пределах шума. Различимы только аккорды: фанк (0.66) хуже рока и singer-songwriter (0.93), интервалы не пересекаются.

Разбивки по исполнителю, темпу и solo по стилям, все с интервалами (`ci95`), лежат в `results/guitarset-madmom/summary.json`, по каждому треку в `tracks.csv`. Пересчитать summary без датасета, из одного `tracks.csv`: `uv run python benchmarks/guitarset_analyze.py --out benchmarks/results/guitarset-madmom --from-tracks`.

### Паттерны ошибок

- **Темп: 236 из 360 точно.** Остальные ошибки почти все «музыкальные», а не мусор: 31 удвоение, 27 половин, 36 отношений 3:2 или 2:3 и только 30 совсем мимо. Удвоение чаще на медленных треках (25 из 96 в slow, меньше 100 BPM), половина на быстрых и в джазе (14 из 72; 200 BPM определяется как 100). Отношение 2:3 сидит в фанке и роке (114 → 78, 130 → 86): трекер цепляется за синкопированные группы шестнадцатых. В singer-songwriter наоборот, 3:2 (68 → 102, 17 из 72): похоже на триольное или 12/8 ощущение.
- **Тональность: главная слабость.** Мажор точно в 55% случаев, минор только в 28%. В 50 из 120 минорных треков модель называет параллельный по ключевым знакам мажор (relative), то есть ноты угаданы, а тоника нет. Confidence из ответа полезен: при confidence выше 0.7 точность 70%, ниже 0.3 только 22%. Текущий порог предупреждения 0.5 выбран разумно.
- **Биты: фаза на босса-нове.** В 26 треках трекер держит правильный период (AMLt выше 0.8), но ставит биты на слабые доли (F ниже 0.2). 13 из них босса-нова, 8 рок. Отсюда же низкие сильные доли: 0.41 в целом.
- **Аккорды на comp хорошие:** majmin 0.85, на роке и singer-songwriter 0.93. Хуже всего фанк (0.66): короткие стаккатные аккорды, много пауз. В джазе root 0.71 против majmin 0.82, потому что модель знает только мажор и минор, а там септаккорды, hdim7 и пропущенные ступени.
- **Solo:** всё хуже, а аккорды бессмысленны, потому что в одноголосой мелодии их нет. Если агенту дадут соло-партию, ответ analyze про аккорды стоит не показывать или явно помечать.
- **Исполнитель 05** заметно хуже остальных по темпу (Acc1 0.45 против 0.63–0.77) и битам (F 0.48). Это, скорее всего, манера игры (свободнее с темпом), а не баг.

### Что из этого следует

1. Темп отдавать вместе с `bpm_alternatives`, как сейчас. Стоит добавить туда и ×2/3, ×3/2: это 10% треков.
2. Тональность на одиночной гитаре ненадёжна, особенно минор. Варианты: показывать две верхние гипотезы, если между ними relative-пара; или брать тонику из аккордов (последний и самый длинный аккорд), это дешёвая эвристика.
3. Сильные доли и размер на одиночной гитаре почти случайны, агенту стоит подавать их как низкодоверенные.
4. Beat This! ещё предстоит сравнить на этом же наборе: `--beats beat_this` с локальным `AH_BEAT_THIS_CHECKPOINT`.

### Запуск

```bash
# с https://zenodo.org/records/3371780: annotation.zip и audio_mono-mic.zip (около 700 МБ)
unzip annotation.zip -d gs/ann && unzip audio_mono-mic.zip -d gs/audio
uv run --project tools/analyze --with mir_eval==0.8.2 \
    python benchmarks/guitarset_analyze.py --audio gs/audio --annotations gs/ann --out gs/out
# --beats beat_this, --workers N, --score-only (пересчитать метрики по кэшу предсказаний)
```

## Темп и биты на Ballroom и GiantSteps (Jamal)

Скрипт `benchmarks/tempo.py`, тесты в `tests/test_tempo_benchmark.py`. Результатов прогона в репозитории пока нет. Ниже оригинальная инструкция автора; команды запускать из корня репозитория.

This benchmark evaluates `ah analyze` separately on Ballroom and GiantSteps.
It prepares JSONL manifests for later GPU training, but does not train a model.
Neither dataset supplies chord reference labels, so this benchmark measures tempo
and, for Ballroom, beat/downbeat timing only.

### Data

- Ballroom: [audio archive](http://mtg.upf.edu/ismir2004/contest/tempoContest/data1.tar.gz)
  (~1.5 GB, expected MD5 `2872a3e52070bc342a4510a95e2fa0b8`) and
  [beat annotations](https://github.com/CPJKU/BallroomAnnotations). Unpack the
  audio archive to any local directory. The loader searches subdirectories by
  audio basename; duplicate basenames are reported as an error.
- GiantSteps: [tempo annotations and official folds](https://github.com/GiantSteps/giantsteps-tempo-dataset).
  Audio is available as Beatport previews through the dataset's `audio_dl.sh`
  (~1 GB). Keep the MP3 files in its `audio/` directory, or point `--audio` to
  another directory. Three `0.0` v2 labels are excluded.

Source audio may have separate terms from the annotation repositories. Keep
downloaded audio outside Git. `tmp/` is ignored by this project.

### Prepare

From the repository root (PowerShell examples):

```powershell
git clone --depth 1 https://github.com/CPJKU/BallroomAnnotations.git tmp/bench-datasets/BallroomAnnotations
git clone --depth 1 https://github.com/GiantSteps/giantsteps-tempo-dataset.git tmp/bench-datasets/giantsteps-tempo-dataset

python benchmarks/tempo.py prepare ballroom `
  --annotations tmp/bench-datasets/BallroomAnnotations `
  --audio tmp/bench-datasets/ballroom-audio `
  --output tmp/bench-ballroom.jsonl

python benchmarks/tempo.py prepare giantsteps `
  --annotations tmp/bench-datasets/giantsteps-tempo-dataset `
  --audio tmp/bench-datasets/giantsteps-tempo-dataset/audio `
  --output tmp/bench-giantsteps.jsonl
```

The command reports how many audio files it found. Manifest records with
missing audio remain in the file, so a run can report coverage honestly.
Run `prepare` again after adding audio, because manifests contain resolved
audio paths.
Ballroom records use `split: all`. GiantSteps retains the official fold ID;
fold 1 is test, fold 2 is validation and folds 3–10 are training candidates.
The benchmark's published full-dataset result uses `--split all`; a trained
model must be evaluated on the held-out `--split test` subset instead.

### Evaluate

Use the `tools/analyze` Python environment (with `ah-analyze`, Beat This! and
madmom installed). The main project environment may have different PyTorch
versions. On Windows, for example:

```powershell
tools/analyze/.venv/Scripts/python.exe benchmarks/tempo.py run `
  --manifest tmp/bench-ballroom.jsonl --output tmp/bench-ballroom-results `
  --beats beat_this --device cpu --limit 10

tools/analyze/.venv/Scripts/python.exe benchmarks/tempo.py run `
  --manifest tmp/bench-giantsteps.jsonl --output tmp/bench-giantsteps-results `
  --beats beat_this --device cuda --split test --limit 10
```

Remove `--limit` for complete runs. `--device cuda` selects an NVIDIA GPU for
Beat This! inference if PyTorch can see it. It does not enable training. The
`madmom` beat tracker runs on CPU even when `--device cuda` is given.

Each output directory contains `results.jsonl` and `summary.json`. Acc1 is
within 4% of the annotated BPM. Acc2 also accepts half or double tempo.
Ballroom beat and downbeat F1 use a 70 ms matching window within the annotated
interval. Failed/missing audio files count toward coverage, never accuracy.
Keep separate output directories for each dataset and tracker.
Long runs save each track immediately. Reuse the same command with `--resume`
after an interruption; it skips completed tracks and retries failed ones.
