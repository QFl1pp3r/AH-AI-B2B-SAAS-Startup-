# Audio Harness

First working component: a small HTTP wrapper around DeepFilterNet for speech denoising.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .\tools\denoise
uvicorn denoise_service.main:app --app-dir .\tools\denoise\src --reload
```

Open `http://127.0.0.1:8000/docs`, upload a WAV/MP3/M4A file to `POST /denoise`, and download the returned WAV.

The first request may download the pretrained DeepFilterNet model. The service intentionally returns an error if the model cannot load; it never pretends a file was denoised.

## Dataset work, for now

Put downloaded source audio under `data/raw/<dataset-name>/`. The starter inventory script only lists files and makes a small CSV; it does not impose a complex data contract yet.

```powershell
python scripts/index_audio.py data/raw/m_ailabs_ru --dataset m_ailabs_ru
```

Recommended initial sources: M-AILABS Russian (clean narrated speech), filtered Common Voice Russian (diverse speech), and GTSinger Russian (studio singing, non-commercial licence).

### Offline dataset preparation

After indexing M-AILABS, create reproducible book-level splits:

```powershell
python scripts/split_m_ailabs.py
```

After MUSAN is unpacked, generate a small 100-item test set with 0, 5 and 10 dB SNR examples:

```powershell
.\.venv\Scripts\python.exe -m pip install soundfile
python scripts/make_noisy_pairs.py --noise-dir data/raw/musan/musan/noise
```
