# Tempo benchmark

This benchmark evaluates `ah analyze` separately on Ballroom and GiantSteps.
It prepares JSONL manifests for later GPU training, but does not train a model.
Neither dataset supplies chord reference labels, so this benchmark measures tempo
and, for Ballroom, beat/downbeat timing only.

## Data

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

## Prepare

From the `audio-harness` directory (PowerShell examples):

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

## Evaluate

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
