"""tracks.csv and summary.json in the GuitarSet format, plus what a rerun needs to reproduce them.

Every summary carries run metadata (commit, date, device, tool backends and params, model
revisions, dataset version, manifest sha256, coverage), so a number in the README can be
traced to exactly what produced it.
"""

from __future__ import annotations

import csv
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.manifest import sha256

REPO = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def device_name(device: str = "cpu") -> str:
    """GPU model when the run used one (nvidia-smi on Colab), else the CPU."""
    if device.startswith("cuda") and shutil.which("nvidia-smi"):
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True)
        if gpu.returncode == 0 and gpu.stdout.strip():
            return gpu.stdout.strip().splitlines()[0]
    return f"cpu {platform.machine()} {platform.processor() or platform.system()}".strip()


def run_meta(*, dataset: str, manifest: Path | None = None, device: str = "cpu", tools: dict[str, dict[str, Any]] | None = None,
             models: dict[str, str] | None = None, coverage: dict[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
    """tools: {name: {"backend": ..., "params": {...}}}; models: {name: "repo@revision" or "file sha256:..."}."""
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": device_name(device),
        "tools": tools or {},
        "models": models or {},
        "dataset": dataset,
        "manifest_sha256": sha256(manifest) if manifest else None,
        "coverage": coverage,
        **extra,
    }


def write_tracks(path: Path, rows: list[dict[str, Any]]) -> Path:
    """Columns in first-seen order, so the first row's layout leads and later extras go to the end."""
    fields = list(dict.fromkeys(k for r in rows for k in r))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_tracks(path: Path, text: set[str] = frozenset()) -> list[dict[str, Any]]:
    """Read tracks.csv back: empty cells become None, numbers floats, columns in text stay strings."""
    def cast(value: str) -> Any:
        if value == "":
            return None
        try:
            return float(value)
        except ValueError:
            return value
    with path.open(newline="", encoding="utf-8") as f:
        return [{k: (v if k in text else cast(v)) for k, v in r.items()} for r in csv.DictReader(f)]


def write_summary(out_dir: Path, summary: dict[str, Any], meta: dict[str, Any]) -> Path:
    path = out_dir / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta, **summary}, indent=2, ensure_ascii=False))
    return path
