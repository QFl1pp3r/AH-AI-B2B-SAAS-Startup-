"""Benchmark ``ah separate`` on the MUSDB18 or MUSDB18-HQ test split.

Supports DATASET/test/*.stem.mp4 or test/TRACK/*.wav.
Scores are whole-track, stereo, sample-aligned SDR and zero-mean SI-SDR.
They are *not* museval's windowed SDR, so do not compare the numbers directly
with MUSDB/SiSEC leaderboards. See benchmarks/README.md for usage.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import soundfile as sf

# Audio stream order in official MUSDB18 .stem.mp4 containers.
STEMS = ("mixture", "drums", "bass", "other", "vocals")
SCORE_STEMS = ("vocals", "instrumental")
SCORER_VERSION = 2  # v1 mapped MUSDB18 compressed streams to the wrong stem names.
BLOCK_FRAMES = 441_000
MAX_FRAME_DIFFERENCE = 4_410  # 100 ms at 44.1 kHz
FIELDS = ("track", "status", "duration_s", "seconds", "inference_s", "rtf", "run_id",
          "vocals_sdr_db", "vocals_si_sdr_db", "instrumental_sdr_db",
          "instrumental_si_sdr_db", "error")


def _db_ratio(numerator: float, denominator: float) -> float | None:
    if numerator <= 1e-12:
        return None  # silent reference: a ratio is undefined
    if denominator <= 1e-12:
        return 100.0  # perfect separation; finite ceiling keeps JSON portable
    return round(10 * math.log10(numerator / denominator), 4)


class ScoreTotals:
    def __init__(self) -> None:
        self.count = 0
        self.ref_sum = self.est_sum = 0.0
        self.ref_sq = self.est_sq = self.cross = self.error_sq = 0.0

    def add(self, reference: np.ndarray, estimate: np.ndarray) -> None:
        reference = reference.astype(np.float64, copy=False)
        estimate = estimate.astype(np.float64, copy=False)
        self.count += reference.size
        self.ref_sum += float(reference.sum())
        self.est_sum += float(estimate.sum())
        self.ref_sq += float(np.sum(reference * reference))
        self.est_sq += float(np.sum(estimate * estimate))
        self.cross += float(np.sum(reference * estimate))
        self.error_sq += float(np.sum((reference - estimate) ** 2))

    def result(self) -> tuple[float | None, float | None]:
        if not self.count:
            raise ValueError("empty audio")
        sdr = _db_ratio(self.ref_sq, self.error_sq)
        ref_energy = self.ref_sq - self.ref_sum**2 / self.count
        est_energy = self.est_sq - self.est_sum**2 / self.count
        dot = self.cross - self.ref_sum * self.est_sum / self.count
        if ref_energy <= 1e-12:
            return sdr, None
        target_energy = dot**2 / ref_energy
        noise_energy = max(0.0, est_energy - target_energy)
        return sdr, _db_ratio(target_energy, noise_energy)


def score_track(reference_dir: Path, estimates: dict[str, Path]) -> dict[str, float | None]:
    """Stream the four ground-truth stems and both estimates in lockstep."""
    paths = {stem: reference_dir / f"{stem}.wav" for stem in STEMS[1:]}
    paths.update({f"estimated_{stem}": path for stem, path in estimates.items()})
    with ExitStack() as stack:
        files = {key: stack.enter_context(sf.SoundFile(path)) for key, path in paths.items()}
        rates = {(audio.samplerate, audio.channels) for audio in files.values()}
        if rates != {(44_100, 2)}:
            raise ValueError(f"all references and estimates must be stereo 44.1 kHz; got {rates}")
        lengths = [audio.frames for audio in files.values()]
        if max(lengths) - min(lengths) > MAX_FRAME_DIFFERENCE:
            raise ValueError(f"reference/estimate duration differs by over 100 ms: {lengths}")
        # Minor padding at a separator boundary is ignored for all stems equally.
        frames = min(lengths)
        totals = {stem: ScoreTotals() for stem in SCORE_STEMS}
        for offset in range(0, frames, BLOCK_FRAMES):
            count = min(BLOCK_FRAMES, frames - offset)
            chunks = {name: audio.read(count, dtype="float64", always_2d=True)
                      for name, audio in files.items()}
            instrumental = chunks["drums"] + chunks["bass"] + chunks["other"]
            totals["vocals"].add(chunks["vocals"], chunks["estimated_vocals"])
            totals["instrumental"].add(instrumental, chunks["estimated_instrumental"])
    scores: dict[str, float | None] = {}
    for stem, total in totals.items():
        scores[f"{stem}_sdr_db"], scores[f"{stem}_si_sdr_db"] = total.result()
    return scores


def tracks_in(dataset: Path) -> list[Path]:
    if dataset.name != "test" and not (dataset / "test").is_dir():
        raise ValueError(f"expected the MUSDB test directory or a root containing test/: {dataset}")
    test = dataset / "test" if (dataset / "test").is_dir() else dataset
    tracks = sorted((path for path in test.iterdir() if path.is_dir() or path.name.endswith(".stem.mp4")),
                    key=lambda path: path.name) if test.is_dir() else []
    if not tracks:
        raise ValueError(f"no test tracks found under {test}")
    return tracks


def track_name(path: Path) -> str:
    return path.name.removesuffix(".stem.mp4")


def decode_stems(source: Path, destination: Path) -> None:
    """Decode a standard MUSDB18 container: mix, drums, bass, other, vocals."""
    from ah.audio import ffmpeg_exe

    for stream, stem in enumerate(STEMS):
        command = [ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error",
                   "-i", str(source), "-map", f"0:a:{stream}", "-vn",
                   "-ar", "44100", "-ac", "2", "-c:a", "pcm_f32le",
                   str(destination / f"{stem}.wav")]
        try:
            subprocess.run(command, check=True, capture_output=True, timeout=600)
        except subprocess.CalledProcessError as exc:
            raise ValueError(f"could not decode {stem} from {source.name}: "
                             f"{exc.stderr.decode(errors='replace').strip()}") from exc


def read_results(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["track"]] = row
    return rows


def summarize(rows: list[dict], *, dataset: Path, backend: str, model: str) -> dict:
    done = [row for row in rows if row["status"] == "ok"]
    summary = {"dataset": str(dataset), "split": "test", "backend": backend,
               "model": model, "metric": "whole-track stereo SDR and zero-mean SI-SDR (not museval)",
               "tracks": len(rows), "succeeded": len(done), "failed": len(rows) - len(done),
               "success_rate": round(len(done) / len(rows), 4) if rows else None}
    for field in ("vocals_sdr_db", "vocals_si_sdr_db", "instrumental_sdr_db",
                  "instrumental_si_sdr_db", "rtf"):
        values = [row[field] for row in done if row.get(field) is not None]
        summary[field] = {"mean": round(statistics.mean(values), 4),
                          "median": round(statistics.median(values), 4)} if values else None
    return summary


def write_reports(out: Path, rows: list[dict], summary: dict) -> None:
    with (out / "tracks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def run(dataset: Path, out: Path, *, backend: str, model: str, limit: int | None,
        resume: bool) -> dict:
    from ah.config import load_settings
    from ah.runner import Runner
    from ah.store import ArtifactStore

    dataset = dataset.resolve()
    out = out.resolve()
    tracks = tracks_in(dataset)
    if limit is not None:
        tracks = tracks[:limit]
    out.mkdir(parents=True, exist_ok=True)
    config = {"dataset": str(dataset), "backend": backend, "model": model,
              "scorer_version": SCORER_VERSION}
    config_path = out / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("output directory has results for another dataset/backend/model")
    else:
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    results_path = out / "results.jsonl"
    if results_path.exists() and not resume:
        raise ValueError("results already exist; use --resume or another --out directory")
    previous = read_results(results_path) if resume else {}
    settings = load_settings()
    runner = Runner(ArtifactStore(out / "artifacts"), settings)
    for index, track in enumerate(tracks, 1):
        name = track_name(track)
        if previous.get(name, {}).get("status") == "ok":
            continue
        row = {field: None for field in FIELDS}
        row.update(track=name, status="failed")
        started = time.perf_counter()
        try:
            with tempfile.TemporaryDirectory(prefix="musdb-", dir=out) as scratch:
                reference = track
                if track.is_file():
                    reference = Path(scratch)
                    decode_stems(track, reference)
                missing = [stem for stem in STEMS if not (reference / f"{stem}.wav").is_file()]
                if missing:
                    raise ValueError(f"missing stems: {', '.join(missing)}")
                info = sf.info(reference / "mixture.wav")
                if (info.samplerate, info.channels) != (44_100, 2) or info.frames == 0:
                    raise ValueError("mixture must be nonempty stereo 44.1 kHz")
                row["duration_s"] = round(info.duration, 3)
                source = runner.ingest(reference / "mixture.wav", project="benchmark-separate")
                inference_started = time.perf_counter()
                outcome = runner.run("separate", {"audio": source.id}, {"model": model},
                                     backend=backend, project=None)
                row["inference_s"] = round(time.perf_counter() - inference_started, 3)
                row["run_id"] = outcome.run_id
                outputs = {artifact.role: runner.store.path(artifact.id) for artifact in outcome.artifacts}
                if set(outputs) != set(SCORE_STEMS):
                    raise ValueError(f"expected vocals and instrumental; got {sorted(outputs)}")
                row.update(score_track(reference, outputs))
                row["status"] = "ok"
        except Exception as exc:
            if hasattr(exc, "run_id"):
                row["run_id"] = exc.run_id
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["seconds"] = round(time.perf_counter() - started, 3)
        if row["duration_s"] and row["inference_s"] is not None:
            row["rtf"] = round(row["inference_s"] / row["duration_s"], 4)
        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        previous[name] = row
        print(f"[{index}/{len(tracks)}] {name}: {row['status']} "
              f"({row['seconds']:.1f}s){' - ' + row['error'] if row['error'] else ''}", file=sys.stderr)
    rows = [previous[track_name(track)] for track in tracks if track_name(track) in previous]
    summary = summarize(rows, dataset=dataset, backend=backend, model=model)
    write_reports(out, rows, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="MUSDB18 or MUSDB18-HQ root or test directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", choices=("local", "remote"), default="local")
    parser.add_argument("--model", default="melband-roformer-kim-vocals")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    try:
        summary = run(args.dataset, args.out, backend=args.backend, model=args.model,
                      limit=args.limit, resume=args.resume)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
