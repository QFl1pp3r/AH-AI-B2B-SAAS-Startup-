"""Speech denoising with DeepFilterNet3 (48 kHz, faster than real time on CPU).

Weights are the official DeepFilterNet3.zip, sha256-pinned as in the Colab benchmark
(notebooks/denoise_benchmark_v2), unpacked into ~/.cache/ah/deepfilternet. On Russian
speech that benchmark found DFN cleaner than the noisy input but with worse Whisper CER
and speaker similarity, so do not denoise before ASR. Sidon (`restore`) sounds better
and shifts the voice more; DFN is the fast option.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
from pydantic import BaseModel, Field

from ah import audio
from ah.contract import ToolOutput, ToolResult

MODEL = "DeepFilterNet3"
MODEL_URL = "https://raw.githubusercontent.com/Rikorose/DeepFilterNet/main/models/DeepFilterNet3.zip"
MODEL_SHA256 = "49c52edc8947ae1f9bf50d81530beaf3a2c3245aeaf34b6f31ff535cd22284d2"
MODEL_SR = 48_000
CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "ah" / "deepfilternet"


class Params(BaseModel, extra="forbid"):
    # Suppress at most this many dB and keep the rest of the noise; None = full suppression.
    atten_lim_db: float | None = Field(default=None, gt=0, le=100)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_unzip(archive: Path, dest: Path) -> None:
    root = dest.resolve()
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            if not (root / name).resolve().is_relative_to(root):
                raise ValueError(f"zip entry escapes target dir: {name}")
        zf.extractall(root)


def fetch_model(cache: Path = CACHE, url: str = MODEL_URL, sha256: str = MODEL_SHA256) -> Path:
    """Download and unpack the pinned checkpoint once; the checkpoint is a pickle, so the hash must match."""
    model_dir = cache / MODEL
    if (model_dir / "config.ini").is_file():
        return model_dir
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / f"{MODEL}.zip"
    if not archive.is_file() or sha256_file(archive) != sha256:
        partial = archive.with_suffix(".zip.part")
        with urllib.request.urlopen(url) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        actual = sha256_file(partial)
        if actual != sha256:
            partial.unlink()
            raise ValueError(f"{MODEL}.zip sha256 mismatch: {actual} != {sha256}")
        partial.replace(archive)
    safe_unzip(archive, cache)
    return model_dir


class DenoiseTool:
    name = "denoise"

    def __init__(self) -> None:
        self.device = "cpu"
        self._enhance: Callable[[np.ndarray, float | None], np.ndarray] | None = None

    def load(self, device: str) -> None:
        if self._enhance is not None:
            return
        import torch
        from df.enhance import enhance, init_df
        from df.utils import get_device

        model, state, _ = init_df(model_base_dir=str(fetch_model()), log_file=None, log_level="WARNING")
        if state.sr() != MODEL_SR:
            raise ValueError(f"{MODEL} runs at {state.sr()} Hz, expected {MODEL_SR}")
        self.device = str(get_device())  # DFN picks cuda or cpu by itself

        def run(channels: np.ndarray, atten_lim_db: float | None) -> np.ndarray:
            with torch.inference_mode():
                out = enhance(model, state, torch.from_numpy(channels), pad=True, atten_lim_db=atten_lim_db)
            return out.numpy()

        self._enhance = run

    def run(self, inputs: dict[str, Path], params: dict[str, Any], out_dir: Path) -> ToolResult:
        options = Params.model_validate(params)
        data, sr = audio.decode(inputs["audio"])
        channels = np.ascontiguousarray(audio.resample(data, sr, MODEL_SR).T)  # DFN takes [C, T], channels as batch
        result = self._enhance(channels, options.atten_lim_db).T
        target = out_dir / "denoised.wav"
        audio.write_wav(target, np.clip(result, -1, 1).astype(np.float32), MODEL_SR)
        meta = {"model": MODEL, "device": self.device, "model_sample_rate": MODEL_SR,
                "atten_lim_db": options.atten_lim_db}
        return ToolResult(
            tool=self.name,
            outputs=[ToolOutput(role="denoised", path=str(target), kind="audio", meta={"model": MODEL})],
            meta=meta,
            warnings=["DeepFilterNet is trained on speech; music and singing get suppressed as noise"],
        )
