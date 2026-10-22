"""Prepare and evaluate the two official tempo benchmarks.

No training is performed here. Manifests retain source paths and GiantSteps folds
so a later GPU trainer can use the same data without contaminating evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path


def audio_index(root: Path) -> dict[str, Path]:
    """Index files by basename; fail on ambiguous duplicate names."""
    result: dict[str, Path] = {}
    if not root.exists():
        return result
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
            continue
        key = path.stem.casefold()
        if key in result:
            raise ValueError(f"duplicate audio basename {path.stem!r}: {result[key]} and {path}")
        result[key] = path.resolve()
    return result


def ballroom_rows(annotations: Path, audio: Path) -> list[dict]:
    files = audio_index(audio)
    rows = []
    for path in sorted(annotations.glob("*.beats")):
        beats, positions = [], []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            t, position = line.split()[:2]
            beats.append(float(t))
            positions.append(int(position))
        if len(beats) < 4 or any(b <= a for a, b in zip(beats, beats[1:])):
            raise ValueError(f"invalid beat annotation: {path}")
        bpm = 60 / statistics.median(b - a for a, b in zip(beats, beats[1:]))
        rows.append({"dataset": "ballroom", "id": path.stem,
                     "audio": str(files[path.stem.casefold()]) if path.stem.casefold() in files else None,
                     "reference_bpm": bpm, "reference_beats": beats,
                     "reference_downbeats": [t for t, p in zip(beats, positions) if p == 1],
                     "split": "all", "annotation": str(path.resolve())})
    return rows


def giantsteps_rows(annotations: Path, audio: Path) -> list[dict]:
    files = audio_index(audio)
    tempo_dir = annotations / "annotations_v2" / "tempo"
    if not tempo_dir.is_dir():
        raise ValueError(f"GiantSteps v2 tempo annotations missing: {tempo_dir}")
    folds: dict[str, int] = {}
    for number in range(1, 11):
        path = annotations / "splits" / f"fold{number:02}.txt"
        for name in path.read_text(encoding="utf-8").splitlines():
            if name.strip():
                key = Path(name.strip()).stem.casefold()
                if key in folds:
                    raise ValueError(f"duplicate GiantSteps fold assignment: {key}")
                folds[key] = number
    rows = []
    for path in sorted(tempo_dir.glob("*.bpm")):
        key = path.stem.casefold()
        bpm = float(path.read_text(encoding="utf-8").strip())
        if bpm <= 0:  # Three missing v2 labels are recorded as zero.
            continue
        fold = folds.get(key)
        if fold is None:
            raise ValueError(f"no official fold for {path.stem}")
        rows.append({"dataset": "giantsteps", "id": path.stem,
                     "audio": str(files[key]) if key in files else None,
                     "reference_bpm": bpm, "fold": fold,
                     "split": "test" if fold == 1 else "validation" if fold == 2 else "train",
                     "annotation": str(path.resolve())})
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tempo_metrics(reference: float, estimate: float | None) -> dict:
    if estimate is None or not math.isfinite(estimate) or estimate <= 0:
        return {"acc1": False, "acc2": False, "absolute_error": None, "relative_error": None}
    ratios = [estimate / (reference * factor) for factor in (0.5, 1, 2)]
    return {"acc1": abs(estimate / reference - 1) <= 0.04,
            "acc2": any(abs(ratio - 1) <= 0.04 for ratio in ratios),
            "absolute_error": abs(estimate - reference),
            "relative_error": abs(estimate / reference - 1)}


def event_f1(reference: list[float], estimate: list[float], tolerance: float = 0.07) -> float | None:
    if not reference:
        return None
    used = set()
    matches = 0
    for event in sorted(estimate):
        candidates = [(abs(event - ref), index) for index, ref in enumerate(reference)
                      if index not in used and abs(event - ref) <= tolerance]
        if candidates:
            _, index = min(candidates)
            used.add(index)
            matches += 1
    return 2 * matches / (len(reference) + len(estimate))


def summarize(rows: list[dict]) -> dict:
    scored = [row for row in rows if row.get("status") == "ok"]
    def mean(field: str) -> float | None:
        values = [row[field] for row in scored if row.get(field) is not None]
        return statistics.mean(values) if values else None
    return {"total": len(rows), "scored": len(scored),
            "missing_audio": sum(row.get("status") == "missing_audio" for row in rows),
            "failed": sum(row.get("status") == "failed" for row in rows),
            "acc1": mean("acc1"), "acc2": mean("acc2"),
            "mean_absolute_error_bpm": mean("absolute_error"),
            "mean_relative_error": mean("relative_error"),
            "beat_f1_70ms": mean("beat_f1_70ms"),
            "downbeat_f1_70ms": mean("downbeat_f1_70ms")}


def evaluate(manifest: Path, output: Path, tracker: str, device: str,
             split: str, limit: int | None, resume: bool = False) -> dict:
    from ah_analyze import AnalyzeTool

    if device.startswith("cuda") and tracker == "beat_this":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this PyTorch environment")
    rows = read_jsonl(manifest)
    if split != "all":
        rows = [row for row in rows if row["split"] == split]
    if limit is not None:
        rows = rows[:limit]
    tool = AnalyzeTool()
    tool.load(device)
    output.mkdir(parents=True, exist_ok=True)
    config = {"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
              "tracker": tracker, "device": device, "split": split, "limit": limit}
    config_path = output / "run_config.json"
    results_path = output / "results.jsonl"
    if resume:
        if not config_path.exists() or json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("cannot resume: run configuration differs or is missing")
        previous = {row["id"]: row for row in read_jsonl(results_path)} if results_path.exists() else {}
    else:
        previous = {}
        write_jsonl(results_path, [])
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    results = []
    for index, row in enumerate(rows, 1):
        if row["id"] in previous and previous[row["id"]]["status"] == "ok":
            results.append(previous[row["id"]])
            continue
        result = {"id": row["id"], "dataset": row["dataset"], "split": row["split"],
                  "reference_bpm": row["reference_bpm"]}
        path = Path(row["audio"]) if row.get("audio") else None
        if not path or not path.is_file():
            result["status"] = "missing_audio"
        else:
            try:
                start = time.perf_counter()
                # Key/chords have no reference labels in these tempo datasets.
                outcome = tool.run({"audio": path}, {"beats": tracker, "key": False, "chords": False}, output)
                analysis = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
                result.update(tempo_metrics(row["reference_bpm"], analysis["bpm"]))
                result.update({"status": "ok", "estimated_bpm": analysis["bpm"],
                               "seconds": time.perf_counter() - start,
                               "beat_count": len(analysis["beats"]),
                               "warnings": outcome.warnings})
                if "reference_beats" in row:
                    # Ballroom leaves some silent intros unannotated. Score only
                    # the reference interval, including a small matching margin.
                    first, last = row["reference_beats"][0], row["reference_beats"][-1]
                    beat_estimates = [t for t in analysis["beats"] if first - 0.07 <= t <= last + 0.07]
                    downbeat_estimates = [t for t in analysis["downbeats"] if first - 0.07 <= t <= last + 0.07]
                    result["beat_f1_70ms"] = event_f1(row["reference_beats"], beat_estimates)
                    result["downbeat_f1_70ms"] = event_f1(row["reference_downbeats"], downbeat_estimates)
            except Exception as exc:
                result.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        results.append(result)
        with results_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"[{index}/{len(rows)}] {row['id']}: {result['status']}", file=sys.stderr, flush=True)
    # Compact repeated attempts into one current result per input.
    write_jsonl(results_path, results)
    summary = {"dataset": rows[0]["dataset"] if rows else None, "tracker": tracker,
               "device": device, "split": split, "metrics": summarize(results),
               "manifest_sha256": config["manifest_sha256"]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "analysis.json").unlink(missing_ok=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="create a manifest; report missing audio")
    prep.add_argument("dataset", choices=["ballroom", "giantsteps"])
    prep.add_argument("--annotations", type=Path, required=True)
    prep.add_argument("--audio", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run", help="evaluate the existing ah analyze tool")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--beats", choices=["beat_this", "madmom"], default="beat_this")
    run.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or cuda")
    run.add_argument("--split", choices=["all", "train", "validation", "test"], default="all")
    run.add_argument("--limit", type=int, help="first N examples for a smoke test")
    run.add_argument("--resume", action="store_true", help="resume an interrupted run with the same configuration")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        rows = (ballroom_rows if args.dataset == "ballroom" else giantsteps_rows)(args.annotations, args.audio)
        if not rows:
            parser.error("no annotations found")
        write_jsonl(args.output, rows)
        print(json.dumps({"dataset": args.dataset, "total": len(rows),
                          "audio_found": sum(bool(row["audio"]) for row in rows),
                          "manifest": str(args.output)}, indent=2))
    else:
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        summary = evaluate(args.manifest, args.output, args.beats,
                           args.device, args.split, args.limit, args.resume)
        print(json.dumps(summary, indent=2))
        if summary["metrics"]["scored"] == 0:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
