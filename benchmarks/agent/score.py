"""Score the agent's plans against the gold plans of the tasks. Deterministic, no LLM judge.

Per attempt: did the plan pass validate (first try or after the retry), is the tool sequence
one of the accepted ones, are the inputs wired right (restore on 1.vocals, not on track),
are the params right (trim times within 0.1 s), any forbidden tool, tool-set precision and
recall, an honest "not possible yet" where nothing can do the job, the answer in the reply
when the data already has it. Per task: pass^k, all k attempts correct.

    uv run python benchmarks/agent/score.py --out benchmarks/results/agent-plan-deepseek-flash-thinking

Reads <out>/raw/runs.jsonl, writes <out>/runs.csv (one row per attempt), tracks.csv (one per
task) and summary.json. CIs resample tasks, since the k attempts of one task are correlated.
--from-runs rebuilds summary.json from runs.csv alone.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import results, stats
from common.manifest import read

HERE = Path(__file__).parent
PARAM_TOLERANCE = 0.1
# "Not possible yet" in ru or en; replies in other languages fail here and in reply_lang_ok alike.
UNSUPPORTED = re.compile("|".join([
    r"нельзя", r"не умею", r"не могу", r"пока не", r"не поддерж", r"недоступ", r"не получится", r"невозможно", r"нечем",
    r"нет такого", r"у меня нет", r"не входит", r"(инструмент|такого)[^.!]{0,40}нет", r"нет[^.!]{0,40}(инструмент|такого)",
    r"can't", r"cannot", r"can not", r"not possible", r"not supported", r"isn't supported", r"isn't possible",
    r"isn't something", r"not something", r"not yet", r"unable", r"don't support", r"doesn't support", r"not available",
    r"isn't available", r"there's no", r"there is no", r"don't have", r"doesn't exist", r"\bno\b[^.!]{0,40}\btool",
    r"not part of", r"nothing here can", r"вне инструмент",
    r"aren't something", r"aren't possible", r"aren't supported", r"none of (the|our|these) tools", r"не умеет",
    r"ни один из [^.!]{0,40}инструмент",
]))
EN_WORDS = {"the", "and", "to", "of", "is", "it", "you", "i", "i'll", "this", "that", "with", "for", "can", "will", "your",
            "on", "as", "from", "so", "but", "not", "there's", "here", "then", "be", "get", "it's", "we", "no", "need", "run", "already"}
# USD per 1M tokens at peak, DeepSeek pricing page as of 2026-10-22. Off-peak is half price.
# deepseek-v4-pro has no published cache-hit price in our notes, so hits are priced as misses there.
PRICES = {
    "deepseek-flash": {"hit": 0.006, "miss": 0.30, "out": 1.20},
    "deepseek-v4-flash": {"hit": 0.006, "miss": 0.30, "out": 1.20},
    "deepseek-v4-pro": {"hit": 1.32, "miss": 1.32, "out": 3.96},
    # OpenRouter list prices as of 2026-10-23: one tariff around the clock.
    "deepseek/deepseek-v4.1-flash-1017": {"hit": 0.003, "miss": 0.15, "out": 0.60, "flat": True},
    "deepseek/deepseek-v4.1-flash": {"hit": 0.006, "miss": 0.30, "out": 1.20, "flat": True},
    "deepseek/deepseek-v4-pro": {"hit": 0.0796, "miss": 0.955, "out": 1.911, "flat": True},
}
PEAK_UTC = ((1, 4), (6, 10))  # weekdays only
METRICS = ("correct", "plan_ok", "valid_first_try", "valid", "seq_ok", "inputs_ok", "params_ok", "no_forbidden",
           "precision", "recall", "unsupported_ok", "reply_ok", "reply_lang_ok")
TEXT = {"task", "category", "lang", "tools", "forbidden", "error", "reply", "failed"}


def peak(at: float) -> bool:
    t = datetime.fromtimestamp(at, timezone.utc)
    return t.weekday() < 5 and any(lo <= t.hour < hi for lo, hi in PEAK_UTC)


def call_cost(call: dict[str, Any], model: str, at_peak: bool = False) -> float | None:
    """Price of one call at the tariff of its hour; at_peak prices every call at peak, to compare runs made at different times."""
    price = PRICES.get(model)
    usage = call.get("usage")
    if price is None or not usage:
        return None
    hit = usage.get("prompt_cache_hit_tokens", usage.get("cached_tokens", 0))
    miss = usage.get("prompt_cache_miss_tokens", usage.get("prompt_tokens", 0) - hit)
    cost = (hit * price["hit"] + miss * price["miss"] + usage.get("completion_tokens", 0) * price["out"]) / 1e6
    return cost if at_peak or price.get("flat") or peak(call["at"]) else cost / 2


# Function words of the languages the planner drifted into (it, es, pt, de, fr, nl, pl), none of them English.
FOREIGN_WORDS = {"il", "la", "di", "che", "per", "non", "della", "sono", "una", "con", "è", "el", "los", "las", "del", "que",
                 "y", "para", "por", "es", "lo", "os", "da", "não", "com", "uma", "em", "você", "der", "die", "das", "und",
                 "ist", "nicht", "ich", "mit", "ein", "eine", "den", "zu", "le", "les", "des", "et", "est", "une", "pour",
                 "pas", "je", "het", "een", "ik", "niet", "van", "jest", "nie", "się", "na"}


# Requests in a third language: the harness does not name the reply language for them, the model decides.
OTHER_WORDS = {
    "de": {"der", "die", "das", "und", "ist", "nicht", "ich", "mit", "ein", "eine", "den", "zu", "dem", "du", "es", "auf", "für",
           "vom", "von", "dann", "aus", "wird", "bekommst"},
    "es": {"el", "los", "las", "del", "que", "y", "para", "por", "es", "lo", "una", "un", "con", "te", "tu", "pista", "la", "de",
           "voy", "a", "mezcla", "canción", "voz", "en", "al", "se"},
    "it": {"il", "la", "di", "che", "per", "non", "della", "sono", "una", "un", "con", "è", "ti", "lo", "traccia", "i", "e",
           "del", "brano", "primi", "secondi", "resto", "voce", "taglio", "dalla", "al"},
}
# English words that are also function words of the languages above; they say nothing about which one it is.
SHARED = {"i", "a", "in", "e", "no"}


def reply_lang_ok(reply: str, lang: str) -> bool:
    """ru: mostly Cyrillic letters. en: no Cyrillic or CJK, and English function words outnumber those of it/es/de/fr/nl/pl/pt."""
    letters = re.findall(r"[^\W\d_]", reply)
    cyrillic = sum("а" <= c.lower() <= "я" or c.lower() == "ё" for c in letters)
    if lang == "ru":
        return bool(letters) and cyrillic / len(letters) > 0.5
    if cyrillic or re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", reply):
        return False
    words = re.findall(r"[^\W\d_']+(?:'[a-z]+)?", reply.lower())
    if lang in OTHER_WORDS:
        return sum(w in OTHER_WORDS[lang] for w in words) > sum(w in EN_WORDS and w not in SHARED for w in words)
    english, foreign = sum(w in EN_WORDS for w in words), sum(w in FOREIGN_WORDS for w in words)
    # A short plain-ASCII reply may have no function words at all ("One trim, cutting everything before 0:12.").
    return bool(words) and (english > foreign or (english == foreign == 0 and all(c.isascii() for c in letters)))


# Params that decide what the user gets. When the gold gives params for such a tool, the plan's effective
# values (defaults filled in) must match the gold's, so a stray start_s cannot slip past a gold that only names end_s.
SEMANTIC = {"trim": {"start_s": 0.0, "end_s": None, "silence": False}}


def effective(tool: str, params: dict[str, Any], duration: float | None) -> dict[str, Any]:
    out = {}
    for key, default in SEMANTIC[tool].items():
        value = params.get(key, default)
        if key == "start_s" and value is None:
            value = 0.0
        if key == "end_s" and value is not None and duration and value >= duration - PARAM_TOLERANCE:
            value = None  # the end of the file, however it was written
        out[key] = value
    return out


def step_params_ok(step: dict[str, Any], want: dict[str, Any], duration: float | None) -> bool:
    if step["tool"] in SEMANTIC:
        got, exp = effective(step["tool"], step["params"], duration), effective(step["tool"], want, duration)
        return all(got[k] is None if exp[k] is None else param_ok(got[k], exp[k]) for k in exp)
    return all(param_ok(step["params"].get(k), v) for k, v in want.items())


def param_ok(got: Any, want: Any) -> bool:
    if isinstance(want, bool) or not isinstance(want, (int, float)):
        return got == want
    return isinstance(got, (int, float)) and not isinstance(got, bool) and abs(got - want) <= PARAM_TOLERANCE


def f1_against(tools: list[str], accept: list[list[str]]) -> tuple[float, float]:
    """Tool-set precision and recall against the accepted plan they fit best."""
    best = (-1.0, 0.0, 0.0)
    for gold in accept:
        got, want = set(tools), set(gold)
        p = len(got & want) / len(got) if got else float(not want)
        r = len(got & want) / len(want) if want else float(not got)
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        best = max(best, (f1, p, r))
    return best[1], best[2]


def check(run: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    gold = task["gold"]
    calls = [c for c in run["llm_calls"] if "usage" in c or "error" in c]
    row: dict[str, Any] = {"task": task["id"], "category": task["category"], "lang": task["lang"], "attempt": run["attempt"],
                           "valid": float(run["ok"]), "valid_first_try": float(run["ok"] and len(calls) == 1),
                           "retried": float(len(calls) > 1), "error": run["error"] or ""}
    failed = []
    if run["ok"]:
        steps = run["steps"]
        tools = [s["tool"] for s in steps]
        reply = (run["reply"] or "").lower()
        accept = gold.get("accept")
        row["tools"] = ">".join(tools) or "-"
        row["seq_ok"] = float(len(tools) <= gold.get("max_steps", 8) if accept is None else tools in accept)
        # inputs: by step number; tool_inputs: by tool, for gold plans whose steps may come in more than one order.
        by_tool = {tool: [want] if isinstance(want, str) else want for tool, want in gold.get("tool_inputs", {}).items()}
        row["inputs_ok"] = float(all(steps[int(n) - 1]["input"] in ([want] if isinstance(want, str) else want)
                                     for n, want in gold.get("inputs", {}).items() if int(n) <= len(steps))
                                 and all(s["input"] in by_tool[s["tool"]] for s in steps if s["tool"] in by_tool))
        duration = task.get("_duration")
        tool_params = gold.get("tool_params", {})
        # needs: what the request asked analyze for (key, chords); switching it off is a wrong plan.
        row["params_ok"] = float(all(int(n) <= len(steps) and step_params_ok(steps[int(n) - 1], want, duration)
                                     for n, want in gold.get("params", {}).items())
                                 and all(step_params_ok(st, tool_params[st["tool"]], duration) for st in steps if st["tool"] in tool_params)
                                 and all(st["params"].get(need) is not False for st in steps if st["tool"] == "analyze"
                                         for need in gold.get("needs", [])))
        forbidden = sorted(set(tools) & set(gold.get("forbid", [])))
        row["forbidden"] = ",".join(forbidden)
        row["no_forbidden"] = float(not forbidden)
        row["precision"], row["recall"] = f1_against(tools, accept) if accept is not None else (None, None)
        row["unsupported_ok"] = float(bool(UNSUPPORTED.search(reply))) if gold.get("must_say_unsupported") else None
        row["reply_ok"] = float(any(s.lower() in reply for s in gold["reply_any"])) if gold.get("reply_any") else None
        # Scored only where the scorer can tell the language: translit, mixed ru/en, uk and the rest are not.
        row["reply_lang_ok"] = (float(reply_lang_ok(run["reply"] or "", task["lang"]))
                                if run["reply"] and task["lang"] in ("ru", "en", *OTHER_WORDS) else None)
        row["reply"] = run["reply"][:300]
        failed = [k for k in ("seq_ok", "inputs_ok", "params_ok", "no_forbidden", "unsupported_ok", "reply_ok") if row[k] == 0.0]
    else:
        failed = ["valid"]
    # plan_ok: the plan alone. correct also needs the reply in the user's language: a right plan explained in Italian
    # to someone who wrote in English is not a success for them.
    row["plan_ok"] = float(not failed)
    if row.get("reply_lang_ok") == 0.0:
        failed.append("reply_lang_ok")
    row["correct"] = float(not failed)
    row["failed"] = ",".join(failed)
    usage = [c["usage"] for c in calls if "usage" in c]
    row["latency_s"] = round(sum(c.get("latency_s") or 0 for c in calls if "usage" in c), 3)
    for key, name in (("prompt_tokens", "prompt_tokens"), ("completion_tokens", "completion_tokens"),
                      ("prompt_cache_hit_tokens", "cache_hit_tokens"), ("reasoning_tokens", "reasoning_tokens")):
        row[name] = sum(u.get(key, 0) for u in usage)
    costs = [call_cost(c, task["_model"]) for c in calls if "usage" in c]
    row["cost_usd"] = round(sum(costs), 7) if costs and None not in costs else None
    peak_costs = [call_cost(c, task["_model"], at_peak=True) for c in calls if "usage" in c]
    row["cost_usd_at_peak"] = round(sum(peak_costs), 7) if peak_costs and None not in peak_costs else None
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = [r["task"] for r in rows]
    out: dict[str, Any] = {"tasks": len(set(tasks)), "attempts": len(rows)}
    for key in METRICS:
        values = [np.nan if r.get(key) is None else r[key] for r in rows]
        out[key] = stats.by_unit_ci(values, tasks)
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r["correct"])
    out["pass_k"] = stats.by_unit_ci([float(all(v)) for v in by_task.values()], list(by_task))
    out["invalid_after_retry"] = round(1 - out["valid"]["mean"], 4) if out["valid"]["mean"] is not None else None
    costs = [r["cost_usd"] for r in rows if r.get("cost_usd") is not None]
    out["cost_usd_per_attempt"] = round(float(np.mean(costs)), 6) if costs else None
    out["cost_usd_total"] = round(float(np.sum(costs)), 4) if costs else None
    at_peak = [r["cost_usd_at_peak"] for r in rows if r.get("cost_usd_at_peak") is not None]
    out["cost_usd_per_attempt_at_peak"] = round(float(np.mean(at_peak)), 6) if at_peak else None
    for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
        out[f"{key}_per_attempt"] = round(float(np.mean([r[key] for r in rows])), 1)
    return out


def aggregate(rows: list[dict[str, Any]], meta: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    groups = {"all": summarize(rows)}
    for field in ("category", "lang"):
        buckets = defaultdict(list)
        for r in rows:
            buckets[r[field]].append(r)
        for value, bucket in sorted(buckets.items()):
            groups[f"{field}={value}"] = summarize(bucket)
    wrong = defaultdict(list)
    for r in rows:
        if not r["correct"] and len(wrong[r["category"]]) < 3:
            wrong[r["category"]].append({k: r.get(k) for k in ("task", "attempt", "tools", "failed", "forbidden", "error", "reply")})
    # Planning time per request, the retry included: what the user waits for before the plan shows up.
    latencies = [r["latency_s"] for r in rows if r["latency_s"]]
    summary = {
        "groups": groups,
        "latency_per_request_s": {"p50": round(float(np.percentile(latencies, 50)), 3),
                                  "p95": round(float(np.percentile(latencies, 95)), 3)} if latencies else None,
        "wrong_examples": dict(sorted(wrong.items())),
    }
    results.write_summary(out_dir, summary, meta)
    return summary


def fixture_duration(task: dict[str, Any]) -> float | None:
    path = HERE / task.get("track_data", "")
    return json.loads(path.read_text())["selected_version"]["duration_s"] if path.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--from-runs", action="store_true", help="re-aggregate --out/runs.csv only")
    args = parser.parse_args()

    if args.from_runs:
        rows = results.read_tracks(args.out / "runs.csv", text=TEXT)
        meta = json.loads((args.out / "summary.json").read_text())["meta"]
        summary = aggregate(rows, meta, args.out)
        print(json.dumps(summary["groups"]["all"], indent=1, ensure_ascii=False))
        return

    config = json.loads((args.out / "raw" / "config.json").read_text())
    tasks_path = Path(config["tasks"])
    tasks = {t["id"]: t | {"_model": config["model"], "_duration": fixture_duration(t)} for t in read(tasks_path, required=("id", "gold"))}
    # After run.py --resume an attempt can appear twice; the later line replaces the earlier one.
    latest = {(r["task"], r["attempt"]): r for r in map(json.loads, (args.out / "raw" / "runs.jsonl").read_text().splitlines())}
    runs = sorted(latest.values(), key=lambda r: (list(tasks).index(r["task"]), r["attempt"]))
    rows = [check(r, tasks[r["task"]]) for r in runs]
    results.write_tracks(args.out / "runs.csv", rows)
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    results.write_tracks(args.out / "tracks.csv", [
        {"task": t, "category": rs[0]["category"], "lang": rs[0]["lang"], "attempts": len(rs),
         "correct_rate": round(float(np.mean([r["correct"] for r in rs])), 4), "pass_k": float(all(r["correct"] for r in rs)),
         "plans": " | ".join(sorted({r.get("tools") or "INVALID" for r in rs}))} for t, rs in by_task.items()])
    meta = results.run_meta(dataset=f"agent tasks {tasks_path.name}", manifest=tasks_path, models={"planner": config["llm"]},
                            coverage={"tasks": len(by_task), "attempts": len(rows),
                                      "llm_errors": sum(1 for r in runs if not r["ok"] and "Провайдер" in (r["error"] or ""))},
                            planner=config)
    summary = aggregate(rows, meta, args.out)
    print(json.dumps(summary["groups"]["all"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
