# Датасеты бенчмарков

Источник, версия, sha256 архива и лицензия каждого набора. Таблица из `docs/BENCHMARKS_PLAN.md` («Сводка датасетов»); версия и sha256 заполняются, когда набор впервые скачан для задачи. Сырые данные лежат в `data/raw/<dataset>/` (в `.gitignore`), на Colab в `MyDrive/audio-harness/raw/<dataset>/`; в git не коммитятся.

**NC**: некоммерческая лицензия. Цифры на таких наборах годятся для исследования и демо, но без оговорки их нельзя выдавать за оценку коммерческого продукта.

| Датасет | Источник | Версия / ревизия | sha256 архива | Для чего | Лицензия | NC |
| --- | --- | --- | --- | --- | --- | --- |
| MUSDB18-HQ test | Zenodo 3338373 (по запросу) | | | задачи 2, 3, 4, 7 | исследовательская, треки под разными лицензиями | **да** |
| MUSDB-ALT | HF `jazasyed/musdb-alt` | | | задача 3 | CC BY-NC-SA 4.0 | **да** |
| Jam-ALT | HF `jamendolyrics/jam-alt` | | | задача 3 | **проверить** | вероятно |
| Common Voice 17 ru | HF `fixie-ai/common_voice_17_0` | `34f78a43893414e7b6e271ba94c1d5e05f18b239` (как в denoise v2) | | задача 3 | CC0 | нет |
| DNS Challenge blind set | GitHub microsoft/DNS-Challenge | | | задача 3 (по желанию) | **проверить** (CC BY 4.0?) | вероятно нет |
| GTZAN + GTZAN-Rhythm + gtzan_key | mirdata / авторские репозитории | | | задачи 2, 5, 8 | аудио не ясно, только исследование | **да** |
| Ballroom | MTG UPF + CPJKU/BallroomAnnotations | | | задача 5 (через `tempo.py`) | исследовательская | **да** |
| GiantSteps tempo / key | GitHub GiantSteps, аудио превью Beatport | | | задача 5 | аннотации CC BY-SA (**проверить**), аудио под копирайтом | **да** |
| FSL10K | Zenodo (Freesound Loops) | | | задача 5 | CC по файлам, часть NC | частично |
| GuitarSet | Zenodo 3371780 | 1.1.0, `audio_mono-mic` + `annotation` | | уже сделан | CC BY 4.0 | нет |

## Модели в метриках

Ревизии закреплены в `benchmarks/common/speech.py` (`PINS`), те же, что в бенчмарке denoise v2.

| Модель | Ревизия | Лицензия |
| --- | --- | --- |
| `openai/whisper-large-v3-turbo` (T4) | `41f01f3fe87f28c78e2fbf8b568835947dd65ed9` | MIT |
| `openai/whisper-small` (смоук на CPU) | `973afd24965f72e36ca33b3055d56a652f456b4d` | MIT |
| `prj-beatrice/utmos22-torch-native` | `eecd3e6e7ad9dd794308a4338c3ac7cb34691af9` | **проверить** |
| `speechbrain/spkrec-ecapa-voxceleb` | `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286` | Apache 2.0 |
| DNSMOS P.835 `sig_bak_ovr.onnx` | sha256 `269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd` | MIT (DNS-Challenge, **проверить**) |

Веса инструментов: Kim RoFormer (separate) и модели madmom (analyze) под **CC BY-NC-SA 4.0**.
