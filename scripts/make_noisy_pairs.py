"""Mix clean WAV clips with real noise at chosen SNR values.

This deliberately produces transparent evaluation artifacts: each output has a
matching clean WAV, noise WAV, noisy WAV and CSV row. It is for baseline and
evaluation data, not a replacement for DeepFilterNet's full training pipeline.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np

try:
    import soundfile as sf
except ImportError as exc:  # pragma: no cover - gives the user a useful message
    raise SystemExit("Install soundfile first: .\\.venv\\Scripts\\python -m pip install soundfile") from exc


def mono(audio: np.ndarray) -> np.ndarray:
    return audio.mean(axis=1) if audio.ndim == 2 else audio


def resample_linear(audio: np.ndarray, old_rate: int, new_rate: int) -> np.ndarray:
    if old_rate == new_rate:
        return audio
    target_size = round(len(audio) * new_rate / old_rate)
    old_points = np.linspace(0, 1, num=len(audio), endpoint=False)
    new_points = np.linspace(0, 1, num=target_size, endpoint=False)
    return np.interp(new_points, old_points, audio).astype(np.float32)


def fit_noise(noise: np.ndarray, target_length: int, rng: random.Random) -> np.ndarray:
    if len(noise) >= target_length:
        start = rng.randrange(0, len(noise) - target_length + 1)
        return noise[start : start + target_length]
    repeats = (target_length + len(noise) - 1) // len(noise)
    return np.tile(noise, repeats)[:target_length]


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-manifest", type=Path, default=Path("data/splits/m_ailabs_ru_test.csv"))
    parser.add_argument("--clean-root", type=Path, default=Path("data/raw/m_ailabs_ru"))
    parser.add_argument("--noise-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/derived/denoise_eval"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--snr", type=float, nargs="+", default=[0, 5, 10])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    noise_files = sorted(path for path in args.noise_dir.rglob("*") if path.suffix.lower() == ".wav")
    if not noise_files:
        raise SystemExit(f"No WAV files found in {args.noise_dir}")
    with args.clean_manifest.open(encoding="utf-8", newline="") as handle:
        clean_rows = list(csv.DictReader(handle))
    if not clean_rows:
        raise SystemExit(f"No rows found in {args.clean_manifest}")

    rng = random.Random(args.seed)
    selected = [clean_rows[index % len(clean_rows)] for index in range(args.count)]
    rng.shuffle(selected)
    for directory in ("clean", "noise", "noisy"):
        (args.output_dir / directory).mkdir(parents=True, exist_ok=True)

    manifest_path = args.output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "clean", "noise", "noisy", "snr_db", "source_clean", "source_noise"])
        writer.writeheader()
        for index, row in enumerate(selected):
            clean_path = args.clean_root / row["path"]
            clean, sample_rate = sf.read(clean_path, always_2d=False, dtype="float32")
            clean = mono(clean)
            noise_path = rng.choice(noise_files)
            noise, noise_rate = sf.read(noise_path, always_2d=False, dtype="float32")
            noise = fit_noise(resample_linear(mono(noise), noise_rate, sample_rate), len(clean), rng)
            snr_db = rng.choice(args.snr)
            noise *= rms(clean) / max(rms(noise), 1e-8) / (10 ** (snr_db / 20))
            noisy = clean + noise
            peak = max(float(np.max(np.abs(noisy))), 1.0)
            clean, noise, noisy = clean / peak, noise / peak, noisy / peak
            item_id = f"{index:04d}_snr{snr_db:g}"
            clean_out = f"clean/{item_id}.wav"
            noise_out = f"noise/{item_id}.wav"
            noisy_out = f"noisy/{item_id}.wav"
            sf.write(args.output_dir / clean_out, clean, sample_rate, subtype="PCM_16")
            sf.write(args.output_dir / noise_out, noise, sample_rate, subtype="PCM_16")
            sf.write(args.output_dir / noisy_out, noisy, sample_rate, subtype="PCM_16")
            writer.writerow({"id": item_id, "clean": clean_out, "noise": noise_out, "noisy": noisy_out, "snr_db": snr_db, "source_clean": row["path"], "source_noise": noise_path})
    print(f"Created {len(selected)} noisy-clean pairs in {args.output_dir}")


if __name__ == "__main__":
    main()
