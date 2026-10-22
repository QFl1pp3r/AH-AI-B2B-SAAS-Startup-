"""Build the shareable Colab fine-tuning notebook from the benchmark's runners.

The benchmark notebook is read only for its metric and inference scripts. Its
configuration cell and saved outputs are never copied into the new notebook.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "denoise_benchmark_v2(1).ipynb"
TARGET = ROOT / "notebooks" / "train_russian_deepfilternet_colab.ipynb"


def cell(kind: str, source: str) -> dict:
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def code(source: str) -> dict:
    return cell("code", source)


def md(source: str) -> dict:
    return cell("markdown", source)


def benchmark_script(index: int, target: str) -> str:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][index]["source"])
    assert source.startswith("%%writefile {BENCH}/"), index
    return "%%writefile " + target + "\n" + source.split("\n", 1)[1]


cells = [
    md("""# Russian DeepFilterNet3 fine-tune on Colab

Run on the colleague's Google account with **T4 GPU**. This notebook downloads M-AILABS Russian and the shared MUSAN archive to Colab's **temporary `/content` disk**. Only a small run directory (checkpoints, manifests, metrics) goes to Google Drive. No Hugging Face upload or large Drive storage is required.

It starts from the exact DeepFilterNet3 weights used in `denoise_benchmark_v2(1).ipynb`, trains on held-out-book M-AILABS and disjoint MUSAN noise files, then scores the adapted model on that notebook's **frozen 480-pair Common Voice benchmark** using the same metric code and model revisions. The benchmark's noise files are excluded from training by filename. Existing benchmark data and baseline results are left untouched.

**Before Run all:** share the MUSAN Drive file with the colleague (Viewer access suffices) and confirm the frozen benchmark folder is in `MyDrive/audio-harness/derived/denoise_eval_v2_full`. If the colleague already has `MyDrive/audio-harness/raw/ru_RU.tgz`, this notebook copies it locally; otherwise it downloads the public source archive. The default pilot uses 4,000 train clips, 400 validation clips, and 3 additional epochs. Increase `TRAIN_CLIPS` only after this first run succeeds. Reruns resume checkpoints but must redownload raw data to a new Colab runtime.

Sources: [DeepFilterNet training guide](https://github.com/Rikorose/DeepFilterNet#training), [official M-AILABS dataset](https://www.caito.de/2019/01/03/the-m-ailabs-speech-dataset/). MUSAN link: the file shared for this project.
"""),
    code("""from pathlib import Path
import csv, hashlib, json, os, random, shutil, subprocess, sys, tarfile, zipfile
from urllib.request import urlopen

from google.colab import drive
drive.mount('/content/drive')

assert subprocess.run(['nvidia-smi', '-L'], capture_output=True).returncode == 0, 'Choose a T4 GPU runtime first.'
WORK = Path('/content/dfn_ru_work')
DRIVE = Path('/content/drive/MyDrive/audio-harness')
BENCHMARK = DRIVE / 'derived/denoise_eval_v2_full'
RUN = DRIVE / 'training/dfn3_ru_pilot_v1'
MUSAN_ID = '1kngNXIxSjXTV2JtS9PAbE03yQcnx7iu_'
MAILABS_URL = 'https://www.caito.de/data/Training/stt_tts/ru_RU.tgz'
TRAIN_CLIPS, VALID_CLIPS, TEST_CLIPS = 4000, 400, 400
MAX_EPOCHS, BATCH_SIZE, SEED = 3, 8, 42

for p in (WORK, RUN): p.mkdir(parents=True, exist_ok=True)
assert (BENCHMARK / 'manifest.csv').is_file() and (BENCHMARK / 'testset.json').is_file(), f'Frozen benchmark not found: {BENCHMARK}'
assert shutil.disk_usage('/content').free > 30 * 1024**3, 'Need at least 30 GiB free on Colab local disk.'
print('GPU ready; local free GiB:', round(shutil.disk_usage('/content').free / 1024**3, 1))
print('Persistent run:', RUN)
"""),
    md("""## 1. Install isolated training environment

The baseline notebook used Python 3.11, torch 2.3.1, and DeepFilterNet 0.5.6. Training also needs `deepfilterdataloader`, which has a Linux cp311 wheel. The separate metric environment below matches the benchmark's Python 3.12 stack.
"""),
    code("""%pip -q install uv gdown
import shutil
UV = shutil.which('uv')
PY = WORK / 'venv311/bin/python'
if not PY.exists():
    subprocess.run([UV, 'venv', '--python', '3.11', str(PY.parent.parent)], check=True)
    subprocess.run([UV, 'pip', 'install', '--python', str(PY), 'torch==2.3.1', 'torchaudio==2.3.1', '--index-url', 'https://download.pytorch.org/whl/cu121'], check=True)
    subprocess.run([UV, 'pip', 'install', '--python', str(PY), 'DeepFilterNet==0.5.6', 'DeepFilterDataLoader==0.5.6', 'numpy<2', 'soundfile==0.13.1', 'h5py', 'icecream', 'librosa<0.11'], check=True)
subprocess.run([str(PY), '-c', 'import torch, df, libdfdata, h5py; print("torch", torch.__version__, "CUDA", torch.cuda.is_available())'], check=True)
"""),
    md("""## 2. Download and unpack sources on Colab local disk

This cell never writes source audio to Drive. Google Drive downloads may fail if the shared file is not accessible to the colleague; the error tells them to change sharing or place the archive in their Drive. Archive paths are checked before extraction.
"""),
    code("""import gdown

RAW = WORK / 'raw'
RAW.mkdir(exist_ok=True)
MAILABS_ARCHIVE = RAW / 'ru_RU.tgz'
MUSAN_ARCHIVE = RAW / 'musan_shared_archive'
SPEECH_ROOT = RAW / 'mailabs'
NOISE_ROOT = RAW / 'musan'

if not MAILABS_ARCHIVE.exists():
    saved = DRIVE / 'raw/ru_RU.tgz'
    if saved.is_file():
        shutil.copyfile(saved, MAILABS_ARCHIVE)
    else:
        with urlopen(MAILABS_URL, timeout=60) as response, MAILABS_ARCHIVE.open('wb') as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
assert MAILABS_ARCHIVE.stat().st_size > 1_000_000_000, 'M-AILABS archive looks incomplete.'

if not MUSAN_ARCHIVE.exists():
    downloaded = gdown.download(id=MUSAN_ID, output=str(MUSAN_ARCHIVE), quiet=False)
    assert downloaded and MUSAN_ARCHIVE.is_file(), 'MUSAN download failed. Share the linked file with the colleague (Viewer), then rerun.'
assert MUSAN_ARCHIVE.stat().st_size > 100_000_000, 'MUSAN archive looks incomplete or is an HTML error page.'

def safe_path(root, name):
    target = (root / name).resolve()
    if not target.is_relative_to(root.resolve()):
        raise RuntimeError(f'Unsafe archive member: {name}')
    return target

def extract_archive(archive, target, want):
    marker = target / '.complete'
    if marker.exists(): return
    target.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            members = [m for m in z.infolist() if want(m.filename) and not m.is_dir()]
            for m in members:
                safe_path(target, m.filename)
                if (m.external_attr >> 16) & 0o170000 == 0o120000: raise RuntimeError('Symlink in ZIP')
            for m in members:
                out = safe_path(target, m.filename); out.parent.mkdir(parents=True, exist_ok=True)
                with z.open(m) as src, out.open('wb') as dst: shutil.copyfileobj(src, dst)
    else:
        with tarfile.open(archive, 'r:*') as t:
            members = [m for m in t if m.isfile() and want(m.name)]
            for m in members: safe_path(target, m.name)
            for m in members:
                out = safe_path(target, m.name); out.parent.mkdir(parents=True, exist_ok=True)
                with t.extractfile(m) as src, out.open('wb') as dst: shutil.copyfileobj(src, dst)
    marker.write_text(str(len(members)))
    print(archive.name, 'extracted', len(members), 'files')

extract_archive(MAILABS_ARCHIVE, SPEECH_ROOT, lambda n: n.lower().endswith('.wav') and '/by_book/' in '/' + n)
extract_archive(MUSAN_ARCHIVE, NOISE_ROOT, lambda n: n.lower().endswith('.wav') and '/noise/' in '/' + n)
speech_files = sorted(SPEECH_ROOT.rglob('*.wav'))
noise_files = sorted(NOISE_ROOT.rglob('*.wav'))
assert len(speech_files) > 15_000 and len(noise_files) > 500, (len(speech_files), len(noise_files))
print('M-AILABS clips:', len(speech_files), '| MUSAN noise files:', len(noise_files))
"""),
    md("""## 3. Freeze disjoint speech and noise manifests

M-AILABS is split by book. MUSAN files already used in the Common Voice benchmark are excluded entirely; the rest are split by source file. Every split and source file is saved as a small manifest on Drive. The final comparison uses the existing Common Voice benchmark, which is never used for training or model selection.
"""),
    code("""from collections import defaultdict

benchmark_rows = list(csv.DictReader((BENCHMARK / 'manifest.csv').open(encoding='utf-8')))
used_noise_names = {Path(r['source_noise']).name.casefold() for r in benchmark_rows}
assert len(benchmark_rows) == 480 and len(used_noise_names) > 0

books = defaultdict(list)
for p in speech_files:
    parts = p.parts
    i = parts.index('by_book')
    key = tuple(parts[i+1:i+4])  # gender, speaker, book
    books[key].append(p)
by_speaker = defaultdict(list)
for key in books: by_speaker[key[:2]].append(key)
candidates = [key for keys in by_speaker.values() if len(keys) > 1 for key in keys]
random.Random(SEED).shuffle(candidates)
assert len(candidates) >= 2
speech_split = {'train': [], 'valid': [], 'test': []}
for book, paths in books.items():
    part = 'valid' if book == candidates[0] else 'test' if book == candidates[1] else 'train'
    speech_split[part].extend(paths)

def take(paths, n, salt):
    return sorted(paths, key=lambda p: hashlib.sha256(f'{SEED}:{salt}:{p.name}'.encode()).hexdigest())[:n]

speech_split = {s: take(paths, n, s) for s, paths, n in [('train', speech_split['train'], TRAIN_CLIPS), ('valid', speech_split['valid'], VALID_CLIPS), ('test', speech_split['test'], TEST_CLIPS)]}
available_noise = [p for p in noise_files if p.name.casefold() not in used_noise_names]
assert len(available_noise) >= 100, 'Too few noises remain after benchmark exclusion.'
available_noise.sort(key=lambda p: hashlib.sha256(f'{SEED}:noise:{p.name}'.encode()).hexdigest())
n = len(available_noise)
noise_split = {'train': available_noise[:int(.8*n)], 'valid': available_noise[int(.8*n):int(.9*n)], 'test': available_noise[int(.9*n):]}
assert all(speech_split.values()) and all(noise_split.values())
assert len({p.name.casefold() for p in noise_split['train']} & used_noise_names) == 0

MAN = WORK / 'manifests'; MAN.mkdir(exist_ok=True)
for part in ('train', 'valid', 'test'):
    for kind, collection in [('speech', speech_split), ('noise', noise_split)]:
        paths = collection[part]
        (MAN / f'{kind}_{part}.txt').write_text(''.join(f'{p.resolve()}\\n' for p in paths))
        print(part, kind, len(paths))
snapshot = {'seed': SEED, 'benchmark_manifest_sha256': hashlib.sha256((BENCHMARK / 'manifest.csv').read_bytes()).hexdigest(),
            'speech': {s: [str(p.relative_to(SPEECH_ROOT)) for p in v] for s,v in speech_split.items()},
            'noise': {s: [str(p.relative_to(NOISE_ROOT)) for p in v] for s,v in noise_split.items()}}
manifest_json = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, indent=2)
saved_manifest = RUN / 'source_manifest.json'
if saved_manifest.exists():
    assert json.loads(saved_manifest.read_text()) == snapshot, 'Source split changed since the run began; use a new RUN name.'
else:
    saved_manifest.write_text(manifest_json, encoding='utf-8')
print('Benchmark noises excluded:', len(used_noise_names))
"""),
    md("""## 4. Convert source audio to DeepFilterNet HDF5

This is local temporary storage. The DeepFilterNet loader creates new noisy mixtures during training. HDF5 files are reused within the current Colab session and safely completed with an atomic rename.
"""),
    code("""PREP = WORK / 'prepare_data.py'
if not PREP.exists():
    url = 'https://raw.githubusercontent.com/Rikorose/DeepFilterNet/v0.5.6/DeepFilterNet/df/scripts/prepare_data.py'
    source = urlopen(url, timeout=60).read().decode('utf-8')
    # Upstream nests _check_file inside __main__; Pool.imap cannot pickle it.
    old = 'with Pool(max(args.num_workers, 1)) as p:\\n            res = p.imap(_check_file, f, 100)\\n            data[args.type]["files"] = list(res)'
    new = 'data[args.type]["files"] = [_check_file(line) for line in f]'
    assert old in source, 'Upstream prepare_data.py changed; review before preparing data.'
    PREP.write_text(source.replace(old, new))
H5 = WORK / 'hdf5'; H5.mkdir(exist_ok=True)
for part in ('train', 'valid', 'test'):
    for kind in ('speech', 'noise'):
        target = H5 / f'{kind}_{part}.hdf5'
        if target.exists():
            print('reuse', target.name, round(target.stat().st_size / 1024**2), 'MiB')
            continue
        partial = H5 / f'{kind}_{part}.partial.hdf5'
        if partial.exists(): partial.unlink()  # incomplete output from this notebook only
        log = WORK / f'prepare_{kind}_{part}.log'
        cmd = [str(PY), str(PREP), kind, str(MAN / f'{kind}_{part}.txt'), str(partial), '--sr', '48000', '--dtype', 'int16', '--codec', 'pcm', '--mono', '--num_workers', '2']
        if kind == 'speech': cmd += ['--max_freq', '8000']
        print('preparing', target.name, flush=True)
        with log.open('w') as stream: result = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            print(''.join(log.read_text(errors='replace').splitlines(True)[-40:]))
            raise RuntimeError(f'Preparation failed: {target.name}')
        partial.replace(target)
        print('done', target.name, round(target.stat().st_size / 1024**2), 'MiB')

dataset = {part: [[f'speech_{part}.hdf5', 1.0], [f'noise_{part}.hdf5', 1.0]] for part in ('train', 'valid', 'test')}
DATASET_CFG = WORK / 'dataset.cfg'
DATASET_CFG.write_text(json.dumps(dataset, indent=2))
print('HDF5 total GiB:', round(sum(p.stat().st_size for p in H5.glob('*.hdf5')) / 1024**3, 2))
"""),
    md("""## 5. Seed and train from stock DeepFilterNet3

The downloaded model ZIP is SHA-256 checked against the original benchmark. Its best weights are copied to epoch 0, so this is a genuine fine-tune with a fresh optimizer. Each epoch writes a checkpoint to Drive; running this cell again resumes. Only one recent and one best checkpoint are retained.
"""),
    code("""MODEL_ZIP = WORK / 'DeepFilterNet3.zip'
MODEL_URL = 'https://raw.githubusercontent.com/Rikorose/DeepFilterNet/main/models/DeepFilterNet3.zip'
MODEL_SHA256 = '49c52edc8947ae1f9bf50d81530beaf3a2c3245aeaf34b6f31ff535cd22284d2'
if not MODEL_ZIP.exists():
    with urlopen(MODEL_URL, timeout=60) as response, MODEL_ZIP.open('wb') as out: shutil.copyfileobj(response, out)
assert hashlib.sha256(MODEL_ZIP.read_bytes()).hexdigest() == MODEL_SHA256, 'Stock model SHA-256 mismatch.'
extract_archive(MODEL_ZIP, WORK / 'stock', lambda n: True)
STOCK = WORK / 'stock/DeepFilterNet3'
assert (STOCK / 'config.ini').exists() and list((STOCK / 'checkpoints').glob('model_*.ckpt.best'))

import configparser
settings = {'train_clips': TRAIN_CLIPS, 'valid_clips': VALID_CLIPS, 'test_clips': TEST_CLIPS,
            'epochs': MAX_EPOCHS, 'batch_size': BATCH_SIZE, 'seed': SEED,
            'benchmark_manifest_sha256': snapshot['benchmark_manifest_sha256']}
settings_path = RUN / 'run_settings.json'
if settings_path.exists():
    assert json.loads(settings_path.read_text()) == settings, 'Training settings changed; use a new RUN name.'
else:
    cpdir = RUN / 'checkpoints'; cpdir.mkdir(exist_ok=True)
    config = configparser.ConfigParser()
    config.read(STOCK / 'config.ini')
    config['train'].update({'max_epochs': str(MAX_EPOCHS), 'batch_size': str(BATCH_SIZE),
                            'batch_size_eval': str(BATCH_SIZE), 'num_workers': '2',
                            'num_prefetch_batches': '4', 'max_sample_len_s': '3.0',
                            'early_stopping_patience': '4', 'n_checkpoint_history': '1',
                            'n_best_checkpoint_history': '1', 'seed': str(SEED),
                            'dataloader_snrs': '-5,0,5,10,20'})
    config['optim'].update({'lr': '0.0001', 'lr_min': '0.00001', 'lr_warmup': '0.000001', 'warmup_epochs': '0'})
    if 'distortion' in config: config['distortion']['p_reverb'] = '0.0'  # no RIR corpus in this pilot
    with (RUN / 'config.ini').open('w') as out: config.write(out)
    source_cp = max((STOCK / 'checkpoints').glob('model_*.ckpt.best'))
    shutil.copyfile(source_cp, cpdir / 'model_0.ckpt')
    settings_path.write_text(json.dumps(settings, indent=2))
    print('seeded from', source_cp.name)

cmd = [str(PY), '-m', 'df.train', str(DATASET_CFG), str(H5), str(RUN)]
print('Training; checkpoints are in', RUN / 'checkpoints', flush=True)
subprocess.run(cmd, check=True)
print('Best checkpoints:', [p.name for p in (RUN / 'checkpoints').glob('model_*.ckpt.best')])
"""),
    md("""## 6. Run the adapted model on the frozen benchmark

The runner below comes from the prior benchmark notebook, so inference and output format are identical. Outputs are kept on Colab local disk. Only the metrics CSV is copied to Drive.
"""),
    code("""BENCH = WORK / 'bench'; BENCH.mkdir(exist_ok=True)
EVAL = WORK / 'evaluation'; EVAL.mkdir(exist_ok=True)
for name in ('clean', 'noisy'):
    link = EVAL / name
    if not link.exists(): link.symlink_to(BENCHMARK / name, target_is_directory=True)
shutil.copyfile(BENCHMARK / 'manifest.csv', EVAL / 'manifest.csv')
assert hashlib.sha256((EVAL / 'manifest.csv').read_bytes()).hexdigest() == snapshot['benchmark_manifest_sha256']
assert list((RUN / 'checkpoints').glob('model_*.ckpt.best')), 'No trained best checkpoint; finish training first.'
"""),
    code(benchmark_script(9, "{BENCH}/run_deepfilter.py")),
    code("""subprocess.run([str(PY), str(BENCH / 'run_deepfilter.py'), '--root', str(EVAL),
                '--model-dir', str(RUN), '--output-dir', str(EVAL / 'ru_finetuned')], check=True)
"""),
    md("""## 7. Score with the same metric script as benchmark v2

The metric script is copied directly from `denoise_benchmark_v2(1).ipynb` while building this shareable notebook. The comparison uses the same Whisper large-v3-turbo, DNSMOS, UTMOS and ECAPA revisions. The `metrics/ru_finetuned.csv` file is the only evaluation artifact saved to Drive.
"""),
    code("""PY_EV = WORK / 'venv312/bin/python'
if not PY_EV.exists():
    subprocess.run([UV, 'venv', '--python', '3.12', str(PY_EV.parent.parent)], check=True)
    subprocess.run([UV, 'pip', 'install', '--python', str(PY_EV), 'torch==2.8.0', 'torchaudio==2.8.0', '--index-url', 'https://download.pytorch.org/whl/cu126'], check=True)
    subprocess.run([UV, 'pip', 'install', '--python', str(PY_EV), 'transformers==4.56.2', 'huggingface_hub>=0.34,<1.0', 'speechbrain==1.0.3', 'onnxruntime==1.23.2', 'pesq==0.0.4', 'pystoi==0.4.1', 'jiwer==4.0.0', 'soundfile==0.13.1', 'soxr==1.0.0', 'scipy', 'numpy<2.3'], check=True)
DNSMOS = WORK / 'dnsmos_sig_bak_ovr.onnx'
if not DNSMOS.exists():
    url = 'https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx'
    DNSMOS.write_bytes(urlopen(url, timeout=60).read())
assert hashlib.sha256(DNSMOS.read_bytes()).hexdigest() == '269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd'
"""),
    code(benchmark_script(8, "{BENCH}/metrics.py")),
    code("""subprocess.run([str(PY_EV), str(BENCH / 'metrics.py'), 'evaluate', '--root', str(EVAL),
    '--systems', 'ru_finetuned', '--dnsmos-model', str(DNSMOS),
    '--whisper', 'openai/whisper-large-v3-turbo', '--whisper-revision', '41f01f3fe87f28c78e2fbf8b568835947dd65ed9',
    '--utmos-revision', 'eecd3e6e7ad9dd794308a4338c3ac7cb34691af9',
    '--ecapa-revision', '0f99f2d0ebe89ac095bcc5903c4dd8f72b367286',
    '--cache', str(WORK / 'models'), '--batch-size', '16'], check=True)
dest = RUN / 'ru_finetuned_metrics.csv'
shutil.copyfile(EVAL / 'metrics/ru_finetuned.csv', dest)
print('Saved:', dest)
"""),
    md("""## 8. Paired comparison on the same 480 audio pairs

For CER lower is better; for the other displayed metrics higher is better. A 95% bootstrap interval crossing zero is inconclusive. Treat this pilot as a candidate: listen to failures before choosing it as a product default.
"""),
    code("""import numpy as np
import pandas as pd
from IPython.display import display

baseline = pd.read_csv(BENCHMARK / 'metrics/deepfilter.csv').set_index('id')
noisy = pd.read_csv(BENCHMARK / 'metrics/noisy.csv').set_index('id')
adapted = pd.read_csv(RUN / 'ru_finetuned_metrics.csv').set_index('id')
assert len(baseline) == len(noisy) == len(adapted) == 480
assert baseline.index.equals(adapted.index) and noisy.index.equals(adapted.index)
pair_ids = pd.read_csv(BENCHMARK / 'manifest.csv').set_index('id')['clean_id']
cols = ['dnsmos_ovrl', 'cer', 'spk_sim', 'si_sdr', 'estoi']
print('Mean scores:')
display(pd.DataFrame({'noisy': noisy[cols].mean(), 'stock DFN3': baseline[cols].mean(), 'Russian fine-tune': adapted[cols].mean()}))

rng = np.random.default_rng(SEED)
diff = adapted[cols].subtract(baseline[cols]).join(pair_ids).groupby('clean_id')[cols].mean()
rows = []
for col in cols:
    x = diff[col].dropna().to_numpy()
    means = x[rng.integers(0, len(x), size=(2000, len(x)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    rows.append({'metric': col, 'fine-tune minus stock': round(x.mean(), 4), '95% CI low': round(lo, 4), '95% CI high': round(hi, 4)})
comparison = pd.DataFrame(rows)
display(comparison)
comparison.to_csv(RUN / 'paired_comparison.csv', index=False)
print('Saved:', RUN / 'paired_comparison.csv')
"""),
]

notebook = {
    "cells": cells,
    "metadata": {"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "nbformat": 4,
    "nbformat_minor": 4,
}
for index, item in enumerate(cells):
    source = "".join(item["source"])
    if item["cell_type"] == "code":
        if source.startswith("%%writefile"):
            source = source.split("\n", 1)[1]
        elif source.startswith("%"):
            source = source.split("\n", 1)[1]
        ast.parse(source, filename=f"cell-{index}")
assert all(not item.get("outputs") for item in cells)
assert not any(marker in json.dumps(notebook).lower() for marker in ("hf_gfsl", "hf_token ="))
TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(TARGET, f"({len(cells)} cells)")
