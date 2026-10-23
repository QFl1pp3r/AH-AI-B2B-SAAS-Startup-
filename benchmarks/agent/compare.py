"""Paired comparison of two agent runs on the same tasks: after minus before, with a bootstrap CI over tasks.

Each task's metric is its mean over the k attempts; tasks are the pairs, so the CI of the
difference is not blurred by how much tasks differ from each other.

    uv run python benchmarks/agent/compare.py BEFORE_DIR AFTER_DIR [--out compare.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import results, stats

METRICS = ("correct", "reply_lang_ok", "seq_ok", "no_forbidden", "unsupported_ok")
TEXT = {"task", "category", "lang", "tools", "forbidden", "error", "reply", "failed"}


def per_task(run_dir: Path) -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = defaultdict(lambda: defaultdict(list))
    for row in results.read_tracks(run_dir / "runs.csv", text=TEXT):
        task = tasks[row["task"]]
        task["category"], task["lang"] = row["category"], row["lang"]
        for key in METRICS:
            if row.get(key) is not None:
                task[key].append(row[key])
    return tasks


def compare(before: Path, after: Path) -> dict[str, Any]:
    a, b = per_task(before), per_task(after)
    shared = sorted(set(a) & set(b))
    out: dict[str, Any] = {"before": str(before), "after": str(after), "tasks": len(shared), "groups": {}}
    groups = {"all": shared}
    for field in ("lang", "category"):
        for task in shared:
            groups.setdefault(f"{field}={a[task][field]}", []).append(task)
    for name, tasks in groups.items():
        out["groups"][name] = {key: stats.paired_diff(
            [np.mean(b[t][key]) if b[t][key] else np.nan for t in tasks],
            [np.mean(a[t][key]) if a[t][key] else np.nan for t in tasks]) for key in METRICS}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = compare(args.before, args.after)
    if args.out:
        args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out["groups"]["all"], indent=1))


if __name__ == "__main__":
    main()
