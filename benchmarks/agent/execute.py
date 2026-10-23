"""Execute the plans an agent run recorded: does each plan reach the end and leave the artifacts it promised?

Plan mode only checks the plan against the request. Here every recorded plan runs step by step
through the real Runner (tool envs, canonical audio, outputs wired from step to step) on short
clips standing in for the fixtures, and each step is checked for what its tool must leave:
all output roles from the registry, audio that is non-empty and finite, a BPM in the analyze
report, levels in the inspect report, a trimmed length that matches the requested range.

Clips are short so separate and restore stay affordable on CPU: a 20 s synthetic backing track
(drums at 120 BPM, bass, chords) with a macOS text-to-speech voice on top as the vocal, and the
same text-to-speech with hum and hiss for the spoken fixtures. A synthetic sung tone did not
work as a vocal: separate left it in the instrumental and the vocals stem came out at -46 dBFS. Trim times are scaled by clip length / fixture length, which
keeps "inside the file" and "past its end" as they were in the plan. Content quality is not
judged here; that needs annotated datasets (task 8 of the plan).

    uv run python benchmarks/agent/execute.py benchmarks/results/agent-heldout-or-v4.1-flash

Identical plans on the same fixture run once; tool calls are cached by input sha256, tool and
params under data/derived/agent-exec, shared by all runs, so a rerun only rescores. Writes <runs>/exec/runs.csv (one row
per attempt), plans.jsonl (one per distinct plan) and summary.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ah import audio
from ah.agent import JSON_ROLES
from ah.config import load_settings
from ah.registry import TOOLS
from ah.runner import RunFailed

from agent.score import TEXT
from common import results, stats
from common.manifest import read
from common.runner import CachedRunner

HERE = Path(__file__).parent
# Shared by every run folder, so two planners' identical tool calls run once; data/derived is gitignored.
WORK = HERE.parents[1] / "data" / "derived" / "agent-exec"
SR = audio.CANONICAL_SR
TRIM_TOLERANCE_S = 0.05
SILENT_DBFS = -60.0  # an audio output quieter than this at its peak is no result (a stem with nothing in it)
SONG_FIXTURES = {"song_fresh", "song_analyzed", "song_separated", "song_named_memo"}
CLIP_SECONDS = {"song": 20.0, "memo": 15.0, "talk": 30.0}


# --- clips -----------------------------------------------------------------------

def synth_song(seconds: float, bpm: float = 120.0, seed: int = 0, lead: bool = True) -> np.ndarray:
    """Drums, bass, triads of Am-F-C-G and optionally a vibrato lead: enough for analyze to beat-track."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    beat = 60.0 / bpm
    mix = np.zeros_like(t)
    for n, start in enumerate(np.arange(0, seconds, beat / 2)):
        i = int(start * SR)
        seg = t[: min(int(0.25 * SR), len(t) - i)]
        if n % 2 == 0:  # kick on every beat
            mix[i:i + len(seg)] += 0.8 * np.sin(2 * np.pi * (50 + 80 * np.exp(-seg * 30)) * seg) * np.exp(-seg * 12)
        if n % 4 == 2:  # snare on 2 and 4
            mix[i:i + len(seg)] += 0.35 * rng.normal(0, 1, len(seg)) * np.exp(-seg * 25)
        mix[i:i + len(seg)] += 0.08 * rng.normal(0, 1, len(seg)) * np.exp(-seg * 80)  # hi-hat on eighths
    roots = [220.0, 174.61, 130.81, 196.0]  # A F C G
    thirds = [1.189, 1.26, 1.26, 1.26]  # minor third on A, major elsewhere
    bar = 4 * beat
    melody = [440.0, 523.25, 493.88, 392.0, 349.23, 440.0, 392.0, 329.63]
    for b, start in enumerate(np.arange(0, seconds, bar)):
        i, j = int(start * SR), min(int((start + bar) * SR), len(t))
        seg = t[i:j] - start
        root = roots[b % 4]
        mix[i:j] += 0.25 * np.sin(2 * np.pi * root / 2 * seg)
        for ratio in (1.0, thirds[b % 4], 1.498):
            mix[i:j] += 0.06 * np.sin(2 * np.pi * root * ratio * seg)
        for k in range(2 if lead else 0):  # two sung notes per bar, harmonics with 5 Hz vibrato
            a, z = i + int(k * bar / 2 * SR), min(i + int((k + 1) * bar / 2 * SR), j)
            s = t[a:z] - t[a]
            f = melody[(2 * b + k) % len(melody)] * (1 + 0.01 * np.sin(2 * np.pi * 5 * s))
            phase = 2 * np.pi * np.cumsum(f) / SR
            env = np.minimum(1, s * 20) * np.minimum(1, (s[-1] - s) * 20 + 1e-3)
            mix[a:z] += 0.2 * env * sum(np.sin(h * phase) / h for h in range(1, 7))
    mix /= np.abs(mix).max() / 0.8
    return np.stack([mix, mix], axis=1).astype(np.float32)


def speech(text: str, seconds: float, voice: str, tmp: Path, seed: int = 1, noise: bool = True) -> np.ndarray:
    """macOS say, looped to length, with mains hum and hiss so the cleaners have something to remove."""
    aiff = tmp / f"{voice}.aiff"
    subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
    voice_audio = audio.load_canonical(aiff)[:, 0]
    n = int(seconds * SR)
    voice_audio = np.tile(voice_audio, n // len(voice_audio) + 1)[:n]
    t = np.arange(n) / SR
    hum = 0.02 * np.sin(2 * np.pi * 50 * t) + 0.01 * np.sin(2 * np.pi * 150 * t)
    hum += 0.015 * np.random.default_rng(seed).normal(0, 1, n)
    mono = 0.6 * voice_audio / (np.abs(voice_audio).max() + 1e-9) + (hum if noise else 0)
    return np.stack([mono, mono], axis=1).astype(np.float32)


SUNG = "Lights out on the highway, I keep driving through the night, every mile a little closer, every song a little light."
MEMO = ("Hi, it's me. Quick note before the rehearsal: we start at seven, bring the new set list, "
        "and please check the second verse, the timing is still off.")
TALK = ("Welcome back to the show. Today we talk about home recording, how to treat a small room, "
        "where to put the microphone, and why the first take is often the best one.")


def build_clips(clip_dir: Path, runner: CachedRunner) -> dict[str, Path]:
    clip_dir.mkdir(parents=True, exist_ok=True)
    song = clip_dir / "song.wav"
    with tempfile.TemporaryDirectory() as tmp:
        if not song.exists():
            voice = speech(SUNG, CLIP_SECONDS["song"], "Samantha", Path(tmp), noise=False)
            audio.write_wav(song, 0.6 * synth_song(CLIP_SECONDS["song"], lead=False) + 0.5 * voice)
        for name, text, voice, seconds in (("memo", MEMO, "Samantha", CLIP_SECONDS["memo"]),
                                           ("talk", TALK, "Daniel", CLIP_SECONDS["talk"])):
            if not (clip_dir / f"{name}.wav").exists():
                audio.write_wav(clip_dir / f"{name}.wav", speech(text, seconds, voice, Path(tmp)))
    clips = {name: song for name in SONG_FIXTURES}
    clips |= {"voice_memo": clip_dir / "memo.wav", "podcast_long": clip_dir / "talk.wav", "speech_named_song": clip_dir / "talk.wav"}
    # The selected version of this fixture is a vocals stem, so the clip is one, made by separate itself.
    clips["song_vocals_selected"] = runner.run("separate", song).files["vocals"]
    return clips


# --- running a plan --------------------------------------------------------------

def scaled(step: dict[str, Any], scale: float) -> dict[str, Any]:
    params = dict(step["params"])
    for key in ("start_s", "end_s"):
        if isinstance(params.get(key), (int, float)) and not isinstance(params.get(key), bool):
            params[key] = round(params[key] * scale, 3)
    return params


def duration(path: Path) -> float:
    info = sf.info(path)
    return info.frames / info.samplerate


def check_outputs(tool: str, params: dict[str, Any], source: Path, files: dict[str, Path]) -> str | None:
    """None when the step left what its tool promises, else the first thing missing."""
    for role in TOOLS[tool].outputs:
        if role not in files or not files[role].is_file():
            return f"no {role} output"
        if role in JSON_ROLES:
            report = json.loads(files[role].read_text())
            if role == "analysis" and not (isinstance(report.get("bpm"), (int, float)) and report["bpm"] > 0):
                return "analysis has no BPM"
            if role == "report" and "peak_dbfs" not in json.dumps(report):
                return "report has no levels"
            continue
        data, _ = sf.read(files[role], dtype="float32")
        if len(data) == 0 or not np.isfinite(data).all():
            return f"{role} is empty or not finite"
        peak = 20 * np.log10(np.abs(data).max() + 1e-12)
        if peak < SILENT_DBFS:
            return f"{role} is silent (peak {round(float(peak), 1)} dBFS)"
    if tool == "trim" and not params.get("silence"):
        total = duration(source)
        want = min(params.get("end_s") or total, total) - params.get("start_s", 0)
        got = duration(files["trimmed"])
        if abs(got - want) > TRIM_TOLERANCE_S:
            return f"trimmed to {round(got, 2)} s, asked {round(want, 2)} s"
    return None


def run_plan(runner: CachedRunner, steps: list[dict[str, Any]], clip: Path, scale: float) -> dict[str, Any]:
    produced: dict[int, dict[str, Path]] = {}
    log = []
    for n, step in enumerate(steps, 1):
        if step["input"] == "track":
            source = clip
        else:
            k, _, role = step["input"].partition(".")
            source = produced[int(k)][role]
        params = scaled(step, scale)
        entry: dict[str, Any] = {"step": n, "tool": step["tool"], "params": params}
        try:
            out = runner.run(step["tool"], source, params)
        except (RunFailed, ValueError) as err:
            log.append(entry | {"ok": False, "error": str(err)[:300]})
            break
        problem = check_outputs(step["tool"], params, source, out.files)
        log.append(entry | {"ok": problem is None, "error": problem, "seconds": out.seconds, "cached": out.cached,
                            "warnings": out.warnings[:3]})
        if problem:
            break
        produced[n] = out.files
    return {"ok": len(log) == len(steps) and all(s["ok"] for s in log), "steps": log}


# --- scoring ---------------------------------------------------------------------

def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = [r["task"] for r in rows]
    acting = [r for r in rows if r["has_steps"]]
    out = {"attempts": len(rows), "attempts_with_steps": len(acting),
           "e2e_success": stats.by_unit_ci([r["e2e_ok"] for r in rows], tasks)}
    if acting:
        out["exec_success"] = stats.by_unit_ci([r["exec_ok"] for r in acting], [r["task"] for r in acting])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", type=Path, help="an agent run folder with raw/runs.jsonl and runs.csv from score.py")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out_dir = args.runs / "exec"
    config = json.loads((args.runs / "raw" / "config.json").read_text())
    tasks = {t["id"]: t for t in read(Path(config["tasks"]), required=("id", "gold"))}
    scored = {(r["task"], int(r["attempt"])): r for r in results.read_tracks(args.runs / "runs.csv", text=TEXT)}
    latest = {(r["task"], r["attempt"]): r for r in map(json.loads, (args.runs / "raw" / "runs.jsonl").read_text().splitlines())}

    with CachedRunner(WORK / "cache", load_settings(), device=args.device) as runner:
        clips = build_clips(WORK / "clips", runner)
        plans: dict[str, dict[str, Any]] = {}
        rows = []
        for (task_id, attempt), run in sorted(latest.items()):
            task = tasks[task_id]
            fixture_name = Path(task["track_data"]).stem
            steps = run["steps"] or []
            key = json.dumps([fixture_name, steps], sort_keys=True)
            if steps and key not in plans:
                fixture = json.loads((HERE / task["track_data"]).read_text())
                clip = clips[fixture_name]
                scale = duration(clip) / fixture["selected_version"]["duration_s"]
                plans[key] = {"fixture": fixture_name, "plan": steps, "scale": round(scale, 5)} | run_plan(runner, steps, clip, scale)
                done = plans[key]
                print(f"[{len(plans)}] {fixture_name} {'>'.join(s['tool'] for s in steps)}: "
                      f"{'ok' if done['ok'] else 'FAIL ' + str(done['steps'][-1]['error'])}", file=sys.stderr)
            executed = plans.get(key)
            failed = next((s for s in executed["steps"] if not s["ok"]), None) if executed else None
            correct = scored[(task_id, attempt)]["correct"] == 1.0
            rows.append({"task": task_id, "category": task["category"], "lang": task["lang"], "attempt": attempt,
                         "has_steps": bool(steps), "plan_correct": float(correct),
                         "exec_ok": float(executed["ok"]) if executed else None,
                         # A plan with no steps (an answer, a refusal) has nothing to execute: it succeeds if it was right.
                         "e2e_ok": float(correct and (executed is None or executed["ok"])),
                         "failed_tool": failed["tool"] if failed else "", "error": failed["error"] if failed else ""})

    results.write_tracks(out_dir / "runs.csv", rows)
    (out_dir / "plans.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False, default=str) + "\n" for p in plans.values()))
    by_tool: dict[str, dict[str, Any]] = defaultdict(lambda: {"steps": 0, "ok": 0, "seconds": [], "errors": defaultdict(int)})
    for plan in plans.values():
        for s in plan["steps"]:
            t = by_tool[s["tool"]]
            t["steps"] += 1
            t["ok"] += s["ok"]
            if s.get("seconds") is not None and not s.get("cached"):
                t["seconds"].append(s["seconds"])
            if not s["ok"]:
                t["errors"][s["error"]] += 1
    tools = {name: {"steps": t["steps"], "ok": t["ok"], "rate": round(t["ok"] / t["steps"], 4),
                    "ci95": stats.wilson_ci(t["ok"], t["steps"]),
                    "seconds_p50": round(float(np.median(t["seconds"])), 2) if t["seconds"] else None,
                    "errors": dict(t["errors"])} for name, t in sorted(by_tool.items())}
    groups = {"all": summarize(rows)}
    for field in ("category", "lang"):
        buckets = defaultdict(list)
        for r in rows:
            buckets[r[field]].append(r)
        groups |= {f"{field}={k}": summarize(v) for k, v in sorted(buckets.items())}
    meta = results.run_meta(dataset=f"agent tasks {Path(config['tasks']).name}, recorded plans of {config['llm']}",
                            device=args.device, coverage={"attempts": len(rows), "distinct_plans": len(plans)},
                            clips={k: str(v.name) for k, v in clips.items()}, clip_seconds=CLIP_SECONDS, planner=config)
    results.write_summary(out_dir, {"groups": groups, "tools": tools}, meta)
    print(json.dumps({"all": groups["all"], "tools": tools}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
