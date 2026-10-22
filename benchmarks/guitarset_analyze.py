"""Benchmark the analyze tool on GuitarSet (tempo, beats, downbeats, key, chords).

GuitarSet (Xi et al., ISMIR 2018, CC BY 4.0): 360 excerpts, 6 players x 5 styles
x 3 progressions x 2 tempi x {comp, solo}, hand-verified annotations. Get
annotation.zip and audio_mono-mic.zip from https://zenodo.org/records/3371780
and unzip them into two folders.

    uv run --project tools/analyze --with mir_eval==0.8.2 \
        python benchmarks/guitarset_analyze.py --audio DIR --annotations DIR --out DIR

Predictions are cached per track in --out/pred, so a rerun only scores.
Writes --out/tracks.csv (one row per excerpt) and --out/summary.json with 95%
bootstrap CIs. --from-tracks rebuilds summary.json from a saved tracks.csv alone,
without the dataset.

Notes on the references:
- tempo, beats and key are what the player was told to play (metronome and
  lead sheet), so they hold for both comp and solo excerpts;
- comp chords are scored against the performed chord transcription; solo
  excerpts have only a melody line in the audio (the backing track is not in
  the recording), so their chord scores say what happens when someone feeds a
  solo to analyze, not how good the chord model is.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# One BLAS thread per worker; the pool provides the parallelism.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

NAME = re.compile(r"^(?P<player>\d\d)_(?P<style>[A-Za-z]+)(?P<prog>\d)-(?P<bpm>\d+)-(?P<key>[A-G][b#]?)_(?P<kind>comp|solo)$")
STYLES = {"BN": "bossa nova", "Funk": "funk", "Jazz": "jazz", "Rock": "rock", "SS": "singer-songwriter"}
CHORD_METRICS = ("root", "majmin", "mirex")
BOOTSTRAP = 2000


# --- prediction ------------------------------------------------------------

def _predict_one(wav: str, out_dir: str, beats: str) -> tuple[str, float, list[str]]:
    from ah_analyze import AnalyzeTool

    tool = AnalyzeTool()
    tool.load("cpu")
    start = time.perf_counter()
    result = tool.run({"audio": Path(wav)}, {"beats": beats}, Path(out_dir))
    return out_dir, time.perf_counter() - start, result.warnings


def predict(audio_dir: Path, pred_dir: Path, beats: str, workers: int) -> None:
    jobs = []
    for wav in sorted(audio_dir.glob("*_mic.wav")):
        track = wav.name.removesuffix("_mic.wav")
        out = pred_dir / track
        if (out / "analysis.json").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        jobs.append((str(wav), str(out)))
    print(f"{len(jobs)} excerpts to analyze", file=sys.stderr)
    with ProcessPoolExecutor(workers) as pool:
        futures = [pool.submit(_predict_one, wav, out, beats) for wav, out in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            out, seconds, warnings = future.result()
            (Path(out) / "runtime.json").write_text(json.dumps({"seconds": seconds, "warnings": warnings}))
            print(f"[{done}/{len(jobs)}] {Path(out).name} {seconds:.1f}s", file=sys.stderr)


# --- references --------------------------------------------------------------

def load_reference(jams_path: Path) -> dict[str, Any]:
    """Read the GuitarSet JAMS as plain JSON (the jams package pins old deps)."""
    doc = json.loads(jams_path.read_text())
    ref: dict[str, Any] = {"duration": doc["file_metadata"]["duration"], "chords": []}
    for ann in doc["annotations"]:
        ns, data = ann["namespace"], ann["data"]
        if ns == "tempo":
            ref["bpm"] = float(data[0]["value"])
        elif ns == "key_mode":
            ref["key"] = data[0]["value"].replace(":", " ")
        elif ns == "beat_position":
            ref["beats"] = np.array([d["time"] for d in data])
            ref["downbeats"] = np.array([d["time"] for d in data if d["value"]["position"] == 1])
        elif ns == "chord":
            # Two chord annotations: the lead sheet ("instructed") and the
            # transcription of what was actually played ("performed").
            which = "performed" if ann["annotation_metadata"].get("data_source") else "instructed"
            ref[which] = (np.array([[d["time"], d["time"] + d["duration"]] for d in data]),
                          [d["value"] for d in data])
    return ref


# --- metrics -----------------------------------------------------------------

def tempo_scores(estimate: float | None, reference: float, tolerance: float = 0.04) -> dict[str, float]:
    """MIREX Acc1 (within 4%) and Acc2 (also at 2x, 1/2, 3x, 1/3)."""
    if estimate is None:
        return {"tempo_acc1": 0.0, "tempo_acc2": 0.0}
    ok = lambda factor: abs(estimate - reference * factor) <= tolerance * reference * factor
    return {"tempo_acc1": float(ok(1.0)), "tempo_acc2": float(any(ok(f) for f in (1.0, 2.0, 0.5, 3.0, 1 / 3)))}


def chord_scores(pred: list[dict[str, Any]], ref: tuple[np.ndarray, list[str]], prefix: str) -> dict[str, float]:
    import mir_eval

    ref_intervals, ref_labels = ref
    est_intervals = np.array([[c["start"], c["end"]] for c in pred]) if pred else np.array([[0.0, ref_intervals[-1, 1]]])
    est_labels = [c["label"] for c in pred] if pred else ["N"]
    est_intervals, est_labels = mir_eval.util.adjust_intervals(
        est_intervals, est_labels, ref_intervals.min(), ref_intervals.max(), mir_eval.chord.NO_CHORD, mir_eval.chord.NO_CHORD)
    merged, ref_l, est_l = mir_eval.util.merge_labeled_intervals(ref_intervals, ref_labels, est_intervals, est_labels)
    durations = mir_eval.util.intervals_to_durations(merged)
    out = {}
    for metric in CHORD_METRICS:
        comparisons = getattr(mir_eval.chord, metric)(ref_l, est_l)
        # Durations of excluded chords (e.g. dim/hdim7 under majmin) are not counted.
        weight = durations[comparisons >= 0].sum()
        out[f"{prefix}_{metric}"] = float(mir_eval.chord.weighted_accuracy(comparisons, durations)) if weight > 0 else float("nan")
        out[f"{prefix}_{metric}_weight"] = float(weight)
    return out


def score_track(pred: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    import mir_eval

    row: dict[str, Any] = {"bpm_est": pred.get("bpm"), "beat_tracker": pred.get("beat_tracker")}
    row.update(tempo_scores(pred.get("bpm"), ref["bpm"]))
    est_beats = mir_eval.beat.trim_beats(np.array(pred["beats"]))
    ref_beats = mir_eval.beat.trim_beats(ref["beats"])
    row["beat_f"] = mir_eval.beat.f_measure(ref_beats, est_beats)
    cmlc, cmlt, amlc, amlt = mir_eval.beat.continuity(ref_beats, est_beats)
    row["beat_cmlt"], row["beat_amlt"] = cmlt, amlt
    row["downbeat_f"] = mir_eval.beat.f_measure(
        mir_eval.beat.trim_beats(ref["downbeats"]), mir_eval.beat.trim_beats(np.array(pred["downbeats"])))
    if "key" in pred:
        row["key_est"] = pred["key"]["key"]
        row["key_confidence"] = pred["key"]["confidence"]
        row["key_weighted"] = mir_eval.key.weighted_score(ref["key"], pred["key"]["key"])
        row["key_exact"] = float(row["key_weighted"] == 1.0)
        row["key_relation"] = key_relation(row["key_weighted"])
    if "chords" in pred:
        row.update(chord_scores(pred["chords"], ref["performed"], "chord_perf"))
        row.update(chord_scores(pred["chords"], ref["instructed"], "chord_instr"))
    return row


def key_relation(weighted: float) -> str:
    return {1.0: "correct", 0.5: "fifth", 0.3: "relative", 0.2: "parallel"}.get(weighted, "other")


# --- aggregation -----------------------------------------------------------

def mean(rows: list[dict[str, Any]], key: str, weight: str | None = None) -> float | None:
    pairs = [(r[key], r[weight] if weight else 1.0) for r in rows if key in r and r[key] is not None and not np.isnan(r[key])]
    if not pairs:
        return None
    values, weights = np.array(pairs).T
    return round(float(np.average(values, weights=weights)), 4) if weights.sum() > 0 else None


def bootstrap_ci(rows: list[dict[str, Any]], key: str, weight: str | None = None,
                 n_boot: int = BOOTSTRAP, seed: int = 0) -> list[float] | None:
    """95% percentile CI of the (weighted) mean under a two-way bootstrap over players and lead sheets.

    Every excerpt shares a player with 59 others and a lead sheet with 11 others, so
    resampling excerpts would treat correlated rows as independent and give too narrow
    intervals. Players and sheets are resampled independently (pigeonhole bootstrap);
    a row's multiplicity is the product of its player's and its sheet's draw counts.
    With only 6 players the intervals are wide and approximate; they bound how far the
    numbers would move with other guitarists and other songs.
    """
    rows = [r for r in rows if r.get(key) is not None and not np.isnan(r[key])]
    if not rows:
        return None
    values = np.array([r[key] for r in rows], dtype=float)
    base = np.array([r[weight] if weight else 1.0 for r in rows], dtype=float)
    players, p_idx = np.unique([r["player"] for r in rows], return_inverse=True)
    sheets, s_idx = np.unique([sheet(r) for r in rows], return_inverse=True)
    rng = np.random.default_rng(seed)
    p_count = rng.multinomial(len(players), np.full(len(players), 1 / len(players)), size=n_boot)
    s_count = rng.multinomial(len(sheets), np.full(len(sheets), 1 / len(sheets)), size=n_boot)
    w = p_count[:, p_idx] * s_count[:, s_idx] * base
    total = w.sum(axis=1)
    means = (w @ values)[total > 0] / total[total > 0]
    return [round(float(q), 4) for q in np.percentile(means, [2.5, 97.5])]


def sheet(row: dict[str, Any]) -> str:
    """Lead sheet = style, progression and tempo; all 6 players recorded each one in comp and solo."""
    return f"{row['style']}-{row['prog']}-{row['bpm']}"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(rows)}
    ci: dict[str, Any] = {}
    for key in ("tempo_acc1", "tempo_acc2", "beat_f", "beat_cmlt", "beat_amlt", "downbeat_f", "key_exact", "key_weighted"):
        out[key] = mean(rows, key)
        ci[key] = bootstrap_ci(rows, key)
    for prefix in ("chord_perf", "chord_instr"):
        for metric in CHORD_METRICS:
            # Duration-weighted over the dataset, as in MIREX.
            key, weight = f"{prefix}_{metric}", f"{prefix}_{metric}_weight"
            out[key] = mean(rows, key, weight)
            ci[key] = bootstrap_ci(rows, key, weight)
    out["ci95"] = ci
    return out


def load_tracks(path: Path) -> list[dict[str, Any]]:
    """Read tracks.csv back; numeric columns become floats, empty cells None."""
    def cast(value: str) -> Any:
        if value == "":
            return None
        try:
            return float(value)
        except ValueError:
            return value
    keep_text = {"track", "player", "style", "prog", "bpm", "key", "kind", "key_ref", "key_est",
                 "key_relation", "tempo_band", "beat_tracker"}
    with path.open(newline="") as f:
        return [{k: (v if k in keep_text else cast(v)) for k, v in r.items()} for r in csv.DictReader(f)]


def score(audio_dir: Path, ann_dir: Path, pred_dir: Path, out_dir: Path) -> dict[str, Any]:
    rows = []
    for pred_path in sorted(pred_dir.glob("*/analysis.json")):
        track = pred_path.parent.name
        meta = NAME.match(track)
        if meta is None:
            raise ValueError(f"unexpected GuitarSet name: {track}")
        ref = load_reference(ann_dir / f"{track}.jams")
        row: dict[str, Any] = {"track": track, **meta.groupdict(), "style": STYLES[meta["style"]],
                               "bpm_ref": ref["bpm"], "key_ref": ref["key"], "duration": ref["duration"]}
        row["tempo_band"] = "slow" if ref["bpm"] < 100 else "fast" if ref["bpm"] >= 160 else "mid"
        runtime = pred_path.parent / "runtime.json"
        if runtime.exists():
            row["seconds"] = json.loads(runtime.read_text())["seconds"]
        row.update(score_track(json.loads(pred_path.read_text()), ref))
        rows.append(row)

    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in rows[0], list(rows[0]).index(k) if k in rows[0] else 0, k))
    with (out_dir / "tracks.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return aggregate(rows, out_dir)


def aggregate(rows: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {"all": summarize(rows)}
    for field in ("kind", "style", "tempo_band", "player"):
        buckets = defaultdict(list)
        for r in rows:
            buckets[r[field]].append(r)
        for value, bucket in sorted(buckets.items()):
            groups[f"{field}={value}"] = summarize(bucket)
    for kind in ("comp", "solo"):
        for style in STYLES.values():
            bucket = [r for r in rows if r["kind"] == kind and r["style"] == style]
            groups[f"kind={kind},style={style}"] = summarize(bucket)

    errors = {
        "tempo": {label: sum(1 for r in rows if tempo_error(r) == label)
                  for label in ("ok", "half", "double", "ternary", "other")},
        "key": {label: sum(1 for r in rows if r.get("key_relation") == label)
                for label in ("correct", "fifth", "relative", "parallel", "other")},
    }
    seconds = [r["seconds"] for r in rows if "seconds" in r]
    summary = {
        "dataset": "GuitarSet 1.1.0, audio_mono-mic, 360 excerpts (CC BY 4.0)",
        "beat_trackers": sorted({r["beat_tracker"] for r in rows}),
        "groups": groups,
        "errors": errors,
        "runtime": {"audio_seconds": round(sum(r["duration"] for r in rows), 1),
                    "analyze_seconds": round(sum(seconds), 1)} if seconds else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def tempo_error(row: dict[str, Any]) -> str:
    est, ref = row["bpm_est"], row["bpm_ref"]
    if est is None:
        return "other"
    for label, factors in (("ok", (1.0,)), ("half", (0.5,)), ("double", (2.0,)), ("ternary", (1 / 3, 3.0, 2 / 3, 1.5))):
        if any(abs(est - ref * f) <= 0.04 * ref * f for f in factors):
            return label
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--audio", type=Path, help="folder with *_mic.wav")
    parser.add_argument("--annotations", type=Path, help="folder with *.jams")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--beats", choices=["beat_this", "madmom"], default="madmom")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--from-tracks", action="store_true", help="re-aggregate --out/tracks.csv only")
    args = parser.parse_args()

    if args.from_tracks:
        summary = aggregate(load_tracks(args.out / "tracks.csv"), args.out)
        print(json.dumps(summary["groups"]["all"], indent=2))
        return
    if args.audio is None or args.annotations is None:
        parser.error("--audio and --annotations are required unless --from-tracks")

    pred_dir = args.out / "pred"
    pred_dir.mkdir(parents=True, exist_ok=True)
    if not args.score_only:
        predict(args.audio, pred_dir, args.beats, args.workers)
    summary = score(args.audio, args.annotations, pred_dir, args.out)
    print(json.dumps(summary["groups"]["all"], indent=2))


if __name__ == "__main__":
    main()
