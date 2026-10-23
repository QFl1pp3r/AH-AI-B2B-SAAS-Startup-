"""Speech metrics for the chain benchmarks: Whisper WER/CER, DNSMOS P.835, UTMOS22, ECAPA similarity, SI-SDR.

Ported from Jamal's denoise benchmark v2 (notebooks/denoise_benchmark_v2(1).ipynb, the
metrics.py cell), with the same pins, so numbers stay comparable with that benchmark.
Runs in the benchmarks/envs/speech env; the model libraries are imported lazily, so the
pure functions here also work (and are tested) in the core env.

Everything is compared at 16 kHz: DNSMOS, Whisper and ECAPA all work there.
Resampling is soxr VHQ.
"""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from common.manifest import sha256

SR = 16_000
TARGET_DBFS = -26.0
MAX_LAG_MS = 40

# HF commit for models, sha256 for plain downloads; same values as the denoise v2 notebook.
PINS = {
    "whisper": ("openai/whisper-large-v3-turbo", "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"),
    "whisper_small": ("openai/whisper-small", "973afd24965f72e36ca33b3055d56a652f456b4d"),  # CPU smoke runs
    "utmos": ("prj-beatrice/utmos22-torch-native", "eecd3e6e7ad9dd794308a4338c3ac7cb34691af9"),
    "ecapa": ("speechbrain/spkrec-ecapa-voxceleb", "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"),
    "dnsmos_onnx": ("https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx",
                    "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd"),
    "cv17": ("fixie-ai/common_voice_17_0", "34f78a43893414e7b6e271ba94c1d5e05f18b239"),
}


def load16(path: Path) -> np.ndarray:
    import soundfile as sf
    import soxr

    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != SR:
        audio = soxr.resample(audio, sr, SR, quality="VHQ").astype(np.float32)
    return audio


def active_rms(audio: np.ndarray, frame: int = 320, floor_db: float = 40.0) -> float:
    """RMS over frames within floor_db of the loudest frame: pauses do not dilute the level."""
    n = len(audio) // frame
    if n == 0:
        return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)) + 1e-12)
    energy = np.mean(audio[:n * frame].astype(np.float64).reshape(n, frame) ** 2, axis=1) + 1e-20
    db = 10 * np.log10(energy)
    return float(np.sqrt(np.mean(energy[db > db.max() - floor_db])))


def normalize_level(audio: np.ndarray) -> np.ndarray:
    return (audio * (10 ** (TARGET_DBFS / 20) / active_rms(audio))).astype(np.float32)


def align(estimate: np.ndarray, reference: np.ndarray, max_lag: int = int(SR * MAX_LAG_MS / 1000)) -> tuple[np.ndarray, int]:
    """Shift estimate by the lag that maximises cross-correlation within +-max_lag samples."""
    from scipy.signal import correlate

    n = min(len(estimate), len(reference))
    corr = correlate(estimate[:n], reference[:n], mode="full", method="fft")
    mid = n - 1
    window = corr[mid - max_lag: mid + max_lag + 1]
    lag = int(np.argmax(window)) - max_lag  # > 0: estimate lags behind the reference
    if lag > 0:
        shifted = np.concatenate([estimate[lag:], np.zeros(lag, np.float32)])
    elif lag < 0:
        shifted = np.concatenate([np.zeros(-lag, np.float32), estimate[:lag]])
    else:
        shifted = estimate
    shifted = shifted[:len(reference)]
    if len(shifted) < len(reference):
        shifted = np.pad(shifted, (0, len(reference) - len(shifted)))
    return shifted, lag


def si_sdr(estimate: np.ndarray, reference: np.ndarray) -> float:
    estimate = estimate.astype(np.float64) - estimate.mean()
    reference = reference.astype(np.float64) - reference.mean()
    scale = np.dot(estimate, reference) / max(np.dot(reference, reference), 1e-12)
    target = scale * reference
    residual = estimate - target
    return float(10 * np.log10(max(np.dot(target, target), 1e-12) / max(np.dot(residual, residual), 1e-12)))


def normalize_text(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def error_rates(reference: str, hypothesis: str) -> dict[str, float]:
    """WER and CER on normalized text; an empty hypothesis counts as everything missed."""
    import jiwer

    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
    if not hyp:
        return {"wer": 1.0, "cer": 1.0}
    return {"wer": float(jiwer.wer(ref, hyp)), "cer": float(jiwer.cer(ref, hyp))}


def fetch(url: str, digest: str, dest: Path) -> Path:
    """Download once and refuse a file whose sha256 differs from the pin."""
    if dest.is_file() and sha256(dest) == digest:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    if (actual := sha256(tmp)) != digest:
        tmp.unlink()
        raise RuntimeError(f"sha256 mismatch for {url}: {actual} != {digest}")
    tmp.rename(dest)
    return dest


class DNSMOS:
    """Official DNSMOS P.835 (sig_bak_ovr.onnx), same windowing and polynomial mapping as dnsmos_local.py."""

    INPUT_LENGTH = 9.01

    def __init__(self, model_path: Path):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
        self.p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
        self.p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])

    def __call__(self, audio: np.ndarray) -> dict[str, float]:
        audio = audio.astype(np.float32)
        length = int(self.INPUT_LENGTH * SR)
        while len(audio) < length:
            audio = np.append(audio, audio)
        hops = int(np.floor(len(audio) / SR) - self.INPUT_LENGTH) + 1
        scores = []
        for idx in range(hops):
            seg = audio[idx * SR: idx * SR + length]
            if len(seg) < length:
                continue
            sig, bak, ovr = self.session.run(None, {"input_1": seg[np.newaxis, :]})[0][0]
            scores.append((self.p_sig(sig), self.p_bak(bak), self.p_ovr(ovr)))
        sig, bak, ovr = np.mean(scores, axis=0)
        return {"dnsmos_sig": float(sig), "dnsmos_bak": float(bak), "dnsmos_ovrl": float(ovr)}


def load_models(cache: Path, device: str = "cpu", whisper: str = "whisper") -> dict[str, Any]:
    """Whisper pipeline, UTMOS22, ECAPA and DNSMOS at the pinned revisions; whisper="whisper_small" for CPU smoke runs."""
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    from transformers import AutoModel, pipeline

    asr_repo, asr_rev = PINS[whisper]
    return {
        "asr": pipeline("automatic-speech-recognition", model=asr_repo, revision=asr_rev, device=device,
                        dtype=torch.float16 if device.startswith("cuda") else torch.float32),
        "utmos": AutoModel.from_pretrained(PINS["utmos"][0], revision=PINS["utmos"][1], trust_remote_code=True).eval().to(device),
        "ecapa": EncoderClassifier.from_hparams(source=PINS["ecapa"][0], revision=PINS["ecapa"][1],
                                                savedir=str(Path(cache) / "ecapa"), run_opts={"device": device}),
        "dnsmos": DNSMOS(fetch(*PINS["dnsmos_onnx"], Path(cache) / "dnsmos_sig_bak_ovr.onnx")),
        "revisions": {name: f"{PINS[name][0]}@{PINS[name][1]}" for name in (whisper, "utmos", "ecapa")}
                     | {"dnsmos": f"sig_bak_ovr.onnx sha256:{PINS['dnsmos_onnx'][1]}"},
    }
