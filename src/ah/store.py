"""Artifact store: content-addressed files plus a SQLite index of lineage.

Every result is a new artifact whose parent is the input it was made from, so
history is a tree. Rollback moves a project's head pointer to the parent;
re-running a step from an older node simply grows a new branch. Nothing is
ever overwritten.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

DEFAULT_PROJECT = "default"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    tool TEXT NOT NULL,
    backend TEXT NOT NULL,
    inputs TEXT NOT NULL,
    params TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at REAL NOT NULL,
    finished_at REAL,
    error TEXT,
    meta TEXT NOT NULL DEFAULT '{}',
    warnings TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL,
    ext TEXT NOT NULL,
    kind TEXT NOT NULL,
    role TEXT NOT NULL,
    label TEXT NOT NULL,
    parent_id TEXT REFERENCES artifacts(id),
    run_id TEXT REFERENCES runs(id),
    created_at REAL NOT NULL,
    meta TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS artifacts_parent ON artifacts(parent_id);
CREATE TABLE IF NOT EXISTS heads (
    project TEXT PRIMARY KEY,
    artifact_id TEXT NOT NULL REFERENCES artifacts(id)
);
"""


class StoreError(LookupError):
    pass


@dataclass(frozen=True)
class Artifact:
    id: str
    sha256: str
    ext: str
    kind: str
    role: str
    label: str
    parent_id: str | None
    run_id: str | None
    created_at: float
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "kind": self.kind, "role": self.role, "label": self.label,
            "parent_id": self.parent_id, "run_id": self.run_id, "sha256": self.sha256,
            "created_at": self.created_at, "meta": self.meta,
        }


@dataclass(frozen=True)
class Run:
    id: str
    tool: str
    backend: str
    inputs: dict[str, str]
    params: dict[str, Any]
    status: str
    started_at: float
    finished_at: float | None
    error: str | None
    meta: dict[str, Any]
    warnings: list[str]


def new_id(prefix: str) -> str:
    return f"{prefix}{secrets.token_hex(5)}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class ArtifactStore:
    def __init__(self, home: str | Path):
        self.home = Path(home).expanduser().resolve()
        self.blobs = self.home / "blobs"
        self.scratch = self.home / "scratch"
        self.blobs.mkdir(parents=True, exist_ok=True)
        self.scratch.mkdir(parents=True, exist_ok=True)
        self._db_path = self.home / "store.db"
        with self._connect() as db:
            db.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self._db_path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
            db.commit()
        finally:
            db.close()

    # --- blobs -----------------------------------------------------------

    def blob_path(self, sha256: str, ext: str) -> Path:
        return self.blobs / sha256[:2] / f"{sha256}{ext}"

    def path(self, artifact_id: str) -> Path:
        artifact = self.get(artifact_id)
        return self.blob_path(artifact.sha256, artifact.ext)

    def _put_blob(self, source: Path) -> tuple[str, str]:
        sha = sha256_file(source)
        ext = source.suffix.lower()
        target = self.blob_path(sha, ext)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(f".{target.name}.{os.getpid()}.part")
            shutil.copyfile(source, partial)
            partial.chmod(0o444)
            os.replace(partial, target)
        return sha, ext

    # --- artifacts -------------------------------------------------------

    def add(
        self,
        source: str | Path,
        *,
        kind: str,
        role: str,
        label: str = "",
        parent_id: str | None = None,
        run_id: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Artifact:
        if parent_id is not None:
            self.get(parent_id)
        sha, ext = self._put_blob(Path(source))
        artifact = Artifact(
            id=new_id("a"), sha256=sha, ext=ext, kind=kind, role=role,
            label=label or role, parent_id=parent_id, run_id=run_id,
            created_at=time.time(), meta=meta or {},
        )
        with self._connect() as db:
            db.execute(
                "INSERT INTO artifacts (id, sha256, ext, kind, role, label, parent_id, run_id, created_at, meta)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (artifact.id, sha, ext, kind, role, artifact.label, parent_id, run_id,
                 artifact.created_at, json.dumps(artifact.meta)),
            )
        return artifact

    def get(self, artifact_id: str) -> Artifact:
        with self._connect() as db:
            row = db.execute("SELECT * FROM artifacts WHERE id = ?", (artifact_id,)).fetchone()
        if row is None:
            raise StoreError(f"unknown artifact: {artifact_id}")
        return _artifact(row)

    def list(self, limit: int = 100) -> list[Artifact]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM artifacts ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [_artifact(row) for row in rows]

    def children(self, artifact_id: str) -> list[Artifact]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM artifacts WHERE parent_id = ? ORDER BY created_at", (artifact_id,)
            ).fetchall()
        return [_artifact(row) for row in rows]

    def lineage(self, artifact_id: str) -> list[Artifact]:
        """The artifact and its ancestors, newest first."""
        chain = [self.get(artifact_id)]
        while chain[-1].parent_id is not None:
            chain.append(self.get(chain[-1].parent_id))
        return chain

    def root(self, artifact_id: str) -> Artifact:
        return self.lineage(artifact_id)[-1]

    def tree(self, root_id: str) -> list[tuple[int, Artifact]]:
        """Depth-first (depth, artifact) pairs under a root, for display."""
        result: list[tuple[int, Artifact]] = []

        def walk(artifact: Artifact, depth: int) -> None:
            result.append((depth, artifact))
            for child in self.children(artifact.id):
                walk(child, depth + 1)

        walk(self.get(root_id), 0)
        return result

    # --- runs ------------------------------------------------------------

    def start_run(self, tool: str, backend: str, inputs: dict[str, str], params: dict[str, Any]) -> str:
        for artifact_id in inputs.values():
            self.get(artifact_id)
        run_id = new_id("r")
        with self._connect() as db:
            db.execute(
                "INSERT INTO runs (id, tool, backend, inputs, params, status, started_at)"
                " VALUES (?, ?, ?, ?, ?, 'running', ?)",
                (run_id, tool, backend, json.dumps(inputs), json.dumps(params), time.time()),
            )
        return run_id

    def finish_run(
        self, run_id: str, *, error: str | None = None,
        meta: dict[str, Any] | None = None, warnings: list[str] | None = None,
    ) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE runs SET status = ?, finished_at = ?, error = ?, meta = ?, warnings = ? WHERE id = ?",
                ("failed" if error else "done", time.time(), error,
                 json.dumps(meta or {}), json.dumps(warnings or []), run_id),
            )

    def get_run(self, run_id: str) -> Run:
        with self._connect() as db:
            row = db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise StoreError(f"unknown run: {run_id}")
        return Run(
            id=row["id"], tool=row["tool"], backend=row["backend"],
            inputs=json.loads(row["inputs"]), params=json.loads(row["params"]),
            status=row["status"], started_at=row["started_at"], finished_at=row["finished_at"],
            error=row["error"], meta=json.loads(row["meta"]), warnings=json.loads(row["warnings"]),
        )

    # --- heads -----------------------------------------------------------

    def head(self, project: str = DEFAULT_PROJECT) -> str | None:
        with self._connect() as db:
            row = db.execute("SELECT artifact_id FROM heads WHERE project = ?", (project,)).fetchone()
        return row["artifact_id"] if row else None

    def set_head(self, artifact_id: str, project: str = DEFAULT_PROJECT) -> None:
        self.get(artifact_id)
        with self._connect() as db:
            db.execute(
                "INSERT INTO heads (project, artifact_id) VALUES (?, ?)"
                " ON CONFLICT(project) DO UPDATE SET artifact_id = excluded.artifact_id",
                (project, artifact_id),
            )

    def rollback(self, project: str = DEFAULT_PROJECT) -> str:
        current = self.head(project)
        if current is None:
            raise StoreError(f"project {project!r} has no head")
        parent = self.get(current).parent_id
        if parent is None:
            raise StoreError(f"{current} is an original upload; nothing to roll back to")
        self.set_head(parent, project)
        return parent


def _artifact(row: sqlite3.Row) -> Artifact:
    return Artifact(
        id=row["id"], sha256=row["sha256"], ext=row["ext"], kind=row["kind"], role=row["role"],
        label=row["label"], parent_id=row["parent_id"], run_id=row["run_id"],
        created_at=row["created_at"], meta=json.loads(row["meta"]),
    )
