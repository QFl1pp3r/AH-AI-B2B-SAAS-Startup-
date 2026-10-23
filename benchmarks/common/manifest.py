"""Dataset manifests as JSONL: one track per line with id, audio path, references and split.

Tracks whose audio is missing stay in the manifest, so coverage is reported honestly
(as in tempo.py) instead of the dataset silently shrinking.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REQUIRED = ("id", "audio", "split")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path: Path, rows: Iterable[dict[str, Any]], required: tuple[str, ...] = REQUIRED) -> Path:
    rows = check(list(rows), required)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    return path


def read(path: Path, required: tuple[str, ...] = REQUIRED) -> list[dict[str, Any]]:
    rows = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as err:
                raise ValueError(f"{path}:{n}: not JSON: {err}") from None
    return check(rows, required, str(path))


def check(rows: list[dict[str, Any]], required: tuple[str, ...] = REQUIRED, where: str = "manifest") -> list[dict[str, Any]]:
    seen: set[str] = set()
    for n, row in enumerate(rows, 1):
        missing = [k for k in required if row.get(k) in (None, "")]
        if missing:
            raise ValueError(f"{where}: row {n} ({row.get('id', '?')}) lacks {missing}")
        if row["id"] in seen:
            raise ValueError(f"{where}: duplicate id {row['id']!r}")
        seen.add(row["id"])
    return rows


def audio_path(row: dict[str, Any], root: Path) -> Path:
    """Audio paths in a manifest are relative to the dataset root, so the same file works on Mac and Colab."""
    return Path(root) / row["audio"]


def coverage(rows: list[dict[str, Any]], root: Path, failed: Iterable[str] = ()) -> dict[str, Any]:
    missing = [r["id"] for r in rows if not audio_path(r, root).is_file()]
    failed = sorted(set(failed))
    return {"total": len(rows), "with_audio": len(rows) - len(missing), "missing_audio": missing,
            "failed": failed, "scored": len(rows) - len(missing) - len([f for f in failed if f not in missing])}
