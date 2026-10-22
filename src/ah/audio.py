"""Audio I/O and the only place in the project that resamples.

Canonical artifact format: WAV, 44.1 kHz, stereo, float32. The music models
(Mel-Band RoFormer, madmom) are native 44.1 kHz; Sidon outputs 48 kHz and is
converted back here. Tools that need another rate call `resample` from this
module instead of bringing their own resampler.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import imageio_ffmpeg

CANONICAL_SR = 44_100
CANONICAL_CHANNELS = 2

# Formats libsndfile reads directly; anything else goes through ffmpeg.
_NATIVE_SUFFIXES = {".wav", ".flac", ".ogg", ".aiff", ".aif"}
# Demuxers we let ffmpeg use for untrusted uploads.
_FFMPEG_FORMATS = "mp3,mov,mp4,m4a,3gp,3g2,mj2,aac,wav,flac,ogg,matroska,webm"

MAX_SECONDS = 15 * 60
MAX_BYTES = 500 * 1024 * 1024


class AudioError(ValueError):
    pass


def ffmpeg_exe() -> str:
    """Use a system FFmpeg when present, otherwise the binary in the venv."""
    system_exe = shutil.which("ffmpeg")
    if system_exe:
        return system_exe
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except RuntimeError as err:
        raise AudioError("ffmpeg is required to read or export this format") from err


@dataclass(frozen=True)
class AudioInfo:
    sample_rate: int
    channels: int
    frames: int

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate


def decode(path: str | Path) -> tuple[np.ndarray, int]:
    """Read any supported file as float32 [frames, channels]."""
    path = Path(path)
    if not path.is_file():
        raise AudioError(f"no such file: {path}")
    if path.stat().st_size > MAX_BYTES:
        raise AudioError(f"file is larger than {MAX_BYTES // (1024 * 1024)} MB: {path}")
    if path.suffix.lower() in _NATIVE_SUFFIXES:
        try:
            data, sr = sf.read(path, dtype="float32", always_2d=True)
        except RuntimeError as err:
            raise AudioError(f"cannot read {path}: {err}") from err
    else:
        data, sr = _decode_ffmpeg(path)
    if data.shape[0] == 0:
        raise AudioError(f"empty audio: {path}")
    if data.shape[0] / sr > MAX_SECONDS:
        raise AudioError(f"audio is longer than {MAX_SECONDS} s: {path}")
    return data, sr


def _decode_ffmpeg(path: Path) -> tuple[np.ndarray, int]:
    ffmpeg = ffmpeg_exe()
    command = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-protocol_whitelist", "file,pipe",
        "-format_whitelist", _FFMPEG_FORMATS,
        "-t", str(MAX_SECONDS + 1),
        "-i", str(path),
        "-vn", "-ac", str(CANONICAL_CHANNELS), "-ar", str(CANONICAL_SR),
        "-f", "f32le", "pipe:1",
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=True, timeout=600)
    except subprocess.CalledProcessError as err:
        raise AudioError(f"ffmpeg failed on {path}: {err.stderr.decode(errors='replace').strip()}") from err
    data = np.frombuffer(result.stdout, dtype="<f4").reshape(-1, CANONICAL_CHANNELS)
    return data.copy(), CANONICAL_SR


def resample(data: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return data
    return soxr.resample(data, sr_in, sr_out, quality="VHQ").astype(np.float32, copy=False)


def to_mono(data: np.ndarray) -> np.ndarray:
    """[frames, channels] -> [frames]."""
    return data.mean(axis=1) if data.ndim == 2 else data


def to_channels(data: np.ndarray, channels: int) -> np.ndarray:
    if data.ndim == 1:
        data = data[:, None]
    if data.shape[1] == channels:
        return data
    if channels == 1:
        return to_mono(data)[:, None]
    if data.shape[1] == 1:
        return np.repeat(data, channels, axis=1)
    raise AudioError(f"cannot map {data.shape[1]} channels to {channels}")


def to_canonical(data: np.ndarray, sr: int) -> np.ndarray:
    data = to_channels(data, CANONICAL_CHANNELS)
    return np.ascontiguousarray(resample(data, sr, CANONICAL_SR), dtype=np.float32)


def load_canonical(path: str | Path) -> np.ndarray:
    data, sr = decode(path)
    return to_canonical(data, sr)


def write_wav(path: str | Path, data: np.ndarray, sr: int = CANONICAL_SR) -> None:
    sf.write(Path(path), data, sr, subtype="FLOAT", format="WAV")


def info(path: str | Path) -> AudioInfo:
    meta = sf.info(Path(path))
    return AudioInfo(sample_rate=meta.samplerate, channels=meta.channels, frames=meta.frames)


def stats(data: np.ndarray, sr: int) -> dict:
    """Cheap level statistics, used by `ah inspect` and as metadata on every artifact."""
    mono = to_mono(data)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64)))) if mono.size else 0.0
    frame = max(1, sr // 20)
    usable = (mono.size // frame) * frame
    frame_rms = np.sqrt(np.mean(np.square(mono[:usable].reshape(-1, frame), dtype=np.float64), axis=1)) if usable else np.zeros(0)
    return {
        "duration_s": round(data.shape[0] / sr, 3),
        "sample_rate": sr,
        "channels": 1 if data.ndim == 1 else int(data.shape[1]),
        "peak_dbfs": _db(peak),
        "rms_dbfs": _db(rms),
        "clipped_ratio": round(float(np.mean(np.abs(data) >= 0.999)), 6) if data.size else 0.0,
        "silence_ratio": round(float(np.mean(frame_rms < 10 ** (-60 / 20))), 4) if frame_rms.size else 0.0,
    }


def _db(value: float) -> float | None:
    return round(20 * np.log10(value), 2) if value > 0 else None
