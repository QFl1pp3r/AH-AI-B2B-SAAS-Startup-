"""Prepare and evaluate tempo/rhythm benchmarks.

No training is performed here. Manifests retain source paths and GiantSteps folds
so a later GPU trainer can use the same data without contaminating evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
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


def gtzan_rows(annotations: Path, audio: Path) -> list[dict]:
    """GTZAN-Rhythm v2: use its stated mean tempo and manually corrected events."""
    jams_dir = annotations / "jams"
    stats_path = annotations / "stats.csv"
    if not jams_dir.is_dir() or not stats_path.is_file():
        raise ValueError("GTZAN-Rhythm needs a directory containing jams/ and stats.csv")
    with stats_path.open(newline="", encoding="utf-8-sig") as stream:
        stats = list(csv.DictReader(stream))
    by_name = {row["filename"]: row for row in stats}
    if len(by_name) != len(stats):
        raise ValueError("duplicate filenames in GTZAN-Rhythm stats.csv")
    files = audio_index(audio)
    rows = []
    for path in sorted(jams_dir.glob("*.jams")):
        filename = path.name.removesuffix(".jams")
        if filename not in by_name:
            raise ValueError(f"GTZAN-Rhythm stats.csv lacks {filename}")
        doc = json.loads(path.read_text(encoding="utf-8"))
        typed = {}
        for ann in doc["annotations"]:
            if ann["namespace"] == "beat":
                kind = ann.get("sandbox", {}).get("annotation_type")
                if kind in typed:
                    raise ValueError(f"duplicate {kind} annotation in {path}")
                typed[kind] = ann["data"]
        if "beat" not in typed or "downbeat" not in typed:
            raise ValueError(f"missing beat/downbeat annotation in {path}")
        beats = [float(event["time"]) for event in typed["beat"]]
        downbeats = [float(event["time"]) for event in typed["downbeat"]]
        # JAMS duration is nominally 30 s; some GTZAN WAVs and valid beat
        # annotations extend about 0.65 s beyond it.
        duration = float(doc["file_metadata"]["duration"])
        for kind, events in (("beat", beats), ("downbeat", downbeats)):
            if any(not math.isfinite(t) or t < 0 or t > duration + 1 for t in events):
                raise ValueError(f"invalid {kind} time in {path}")
            if any(b <= a for a, b in zip(events, events[1:])):
                raise ValueError(f"non-increasing {kind} times in {path}")
        if len(beats) < 4:
            raise ValueError(f"too few beats in {path}")
        bpm = float(by_name[filename]["tempo mean"])
        if not math.isfinite(bpm) or bpm <= 0:
            raise ValueError(f"invalid tempo mean in {stats_path}: {filename}")
        track_id = Path(filename).stem
        key = track_id.casefold()
        rows.append({"dataset": "gtzan", "id": track_id,
                     "audio": str(files[key]) if key in files else None,
                     "reference_bpm": bpm, "reference_bpm_source": "stats.csv:tempo mean",
                     "reference_beats": beats, "reference_downbeats": downbeats,
                     "genre": track_id.split(".", 1)[0],
                     "artist": str(doc["file_metadata"].get("artist") or "").strip(),
                     "split": "all", "annotation": str(path.resolve())})
    if not rows or len(rows) != len(stats):
        raise ValueError(f"GTZAN-Rhythm JAMS/stats mismatch: {len(rows)} JAMS, {len(stats)} stats rows")
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
    # Sorted 1-D threshold matching: matching the earliest feasible pair is
    # maximum-cardinality, unlike assigning each estimate its nearest reference.
    refs, ests = sorted(reference), sorted(estimate)
    i = j = 0
    matches = 0
    while i < len(refs) and j < len(ests):
        if abs(refs[i] - ests[j]) <= tolerance:
            matches += 1
            i += 1
            j += 1
        elif refs[i] < ests[j]:
            i += 1
        else:
            j += 1
    return 2 * matches / (len(refs) + len(ests))


def summarize(rows: list[dict]) -> dict:
    scored = [row for row in rows if row.get("status") == "ok"]
    fields = ("acc1", "acc2", "absolute_error", "relative_error",
              "beat_f1_70ms", "downbeat_f1_70ms")
    def mean(field: str) -> float | None:
        values = [row[field] for row in scored if row.get(field) is not None]
        return statistics.mean(values) if values else None
    return {"total": len(rows), "scored": len(scored),
            "missing_audio": sum(row.get("status") == "missing_audio" for row in rows),
            "failed": sum(row.get("status") == "failed" for row in rows),
            "metric_n": {field: sum(row.get(field) is not None for row in scored) for field in fields},
            "acc1": mean("acc1"), "acc2": mean("acc2"),
            "mean_absolute_error_bpm": mean("absolute_error"),
            "mean_relative_error": mean("relative_error"),
            "beat_f1_70ms": mean("beat_f1_70ms"),
            "downbeat_f1_70ms": mean("downbeat_f1_70ms")}


def genre_macro_ci(rows: list[dict], draws: int = 2000, seed: int = 20261023) -> dict:
    """Equal-genre macro means with artist-cluster bootstrap intervals."""
    rng = random.Random(seed)
    fields = ("acc1", "acc2", "beat_f1_70ms", "downbeat_f1_70ms")
    result = {}
    for field in fields:
        by_genre: dict[str, dict[str, list[float]]] = {}
        for row in rows:
            value = row.get(field)
            if row.get("status") != "ok" or value is None:
                continue
            genre = row["genre"]
            # Unknown artists are separate clusters, not one giant cluster.
            artist = row.get("artist") or row["id"]
            by_genre.setdefault(genre, {}).setdefault(artist, []).append(float(value))
        if not by_genre:
            continue
        groups = {genre: [(sum(values), len(values)) for values in artists.values()]
                  for genre, artists in sorted(by_genre.items())}
        point = statistics.mean(sum(s for s, _ in clusters) / sum(n for _, n in clusters)
                                for clusters in groups.values())
        samples = []
        for _ in range(draws):
            genre_means = []
            for clusters in groups.values():
                selected = rng.choices(clusters, k=len(clusters))
                genre_means.append(sum(s for s, _ in selected) / sum(n for _, n in selected))
            samples.append(statistics.mean(genre_means))
        samples.sort()
        result[field] = {"mean": point, "ci95": [samples[int(0.025 * draws)],
                                                   samples[min(draws - 1, int(0.975 * draws))]],
                         "genres": len(groups), "tracks": sum(sum(n for _, n in c) for c in groups.values()),
                         "bootstrap_unit": "artist within genre; unknown artist as individual track"}
    return result


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
        for field in ("genre", "artist", "reference_bpm_source"):
            if field in row:
                result[field] = row[field]
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
                    beat_estimates = analysis["beats"]
                    downbeat_estimates = analysis["downbeats"]
                    if row["dataset"] == "ballroom":
                        # Ballroom leaves some intros/outros unannotated.
                        first, last = row["reference_beats"][0], row["reference_beats"][-1]
                        beat_estimates = [t for t in beat_estimates if first - 0.07 <= t <= last + 0.07]
                        downbeat_estimates = [t for t in downbeat_estimates if first - 0.07 <= t <= last + 0.07]
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
    if rows and rows[0]["dataset"] == "gtzan":
        summary["genre_macro_ci95"] = genre_macro_ci(results)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "analysis.json").unlink(missing_ok=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="create a manifest; report missing audio")
    prep.add_argument("dataset", choices=["ballroom", "giantsteps", "gtzan"])
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
    run.add_argument("--expect-scored", type=int,
                     help="exit nonzero unless exactly this many tracks were scored")
    run.add_argument("--expect-total", type=int,
                     help="exit nonzero unless the manifest selection contains this many tracks")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        adapters = {"ballroom": ballroom_rows, "giantsteps": giantsteps_rows, "gtzan": gtzan_rows}
        rows = adapters[args.dataset](args.annotations, args.audio)
        if not rows:
            parser.error("no annotations found")
        write_jsonl(args.output, rows)
        print(json.dumps({"dataset": args.dataset, "total": len(rows),
                          "audio_found": sum(bool(row["audio"]) for row in rows),
                          "manifest": str(args.output)}, indent=2))
    else:
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        if args.expect_scored is not None and args.expect_scored < 1:
            parser.error("--expect-scored must be positive")
        if args.expect_total is not None and args.expect_total < 1:
            parser.error("--expect-total must be positive")
        summary = evaluate(args.manifest, args.output, args.beats,
                           args.device, args.split, args.limit, args.resume)
        print(json.dumps(summary, indent=2))
        if (summary["metrics"]["scored"] == 0 or
                (args.expect_scored is not None and summary["metrics"]["scored"] != args.expect_scored) or
                (args.expect_total is not None and summary["metrics"]["total"] != args.expect_total)):
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
