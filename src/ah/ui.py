"""Local web page: upload, run a tool on a version, compare before/after, roll back.

A small JSON API over the same Runner and ArtifactStore the CLI uses, plus a
static page in ah/static. No agent here yet. The page is for this machine:
it binds to localhost, rejects foreign Host headers (DNS rebinding) and every
POST needs the X-AH-UI header, which a cross-site form or a no-preflight fetch
cannot send.

No `from __future__ import annotations` here: FastAPI must resolve the
UploadFile annotation of the route defined inside create_app().
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from ah import audio
from ah.registry import TOOLS
from ah.runner import RunFailed, Runner
from ah.store import DEFAULT_PROJECT, ArtifactStore, StoreError

STATIC = Path(__file__).parent / "static"
LOOPBACK = ("127.0.0.1", "localhost", "::1")
CSRF_HEADER = "x-ah-ui"
UPLOAD_SUFFIXES = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3", ".m4a", ".mp4", ".aac", ".webm", ".mov"}


def peaks(path: Path, bins: int) -> list[float]:
    """Max |sample| per bin of the mono mix, for drawing a waveform."""
    data, _ = sf.read(path, dtype="float32", always_2d=True)
    mono = np.abs(data).max(axis=1)
    edges = np.linspace(0, len(mono), bins + 1, dtype=int)
    return [round(float(mono[a:b].max()), 4) if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])]


def chord_line(chords: list[dict[str, Any]]) -> list[str]:
    """Chord labels without N and without immediate repeats, as a musician would write them."""
    line: list[str] = []
    for chord in chords:
        label = chord["label"].replace(":maj", "").replace(":min", "m")
        if chord["label"] != "N" and (not line or line[-1] != label):
            line.append(label)
    return line


def analysis_for(store: ArtifactStore, artifact_id: str) -> dict[str, Any] | None:
    """The newest analyze report on this version or, failing that, on the nearest ancestor."""
    for node in store.lineage(artifact_id):
        reports = [c for c in store.children(node.id) if c.kind == "json" and c.role == "analysis"]
        if reports:
            report = json.loads(store.path(reports[-1].id).read_text())
            return {"of": node.id, "bpm": report.get("bpm"), "meter": report.get("meter"),
                    "key": (report.get("key") or {}).get("key"), "chords": chord_line(report.get("chords", []))}
    return None


def audio_target(store: ArtifactStore, artifact_id: str) -> str:
    """A JSON report (inspect, analyze) can be selected; tools act on the audio it describes."""
    artifact = store.get(artifact_id)
    return artifact_id if artifact.kind == "audio" else artifact.parent_id


def describe(store: ArtifactStore, artifact_id: str) -> dict[str, Any]:
    artifact = store.get(artifact_id)
    info = artifact.as_dict()
    info["tool"] = store.get_run(artifact.run_id).tool if artifact.run_id else "import"
    return info


def create_app(store: ArtifactStore, runner: Runner, host: str = "127.0.0.1"):
    from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile
    from fastapi.responses import FileResponse, JSONResponse, Response
    from starlette.middleware.trustedhost import TrustedHostMiddleware

    app = FastAPI(title="audio-harness", docs_url=None, redoc_url=None, openapi_url=None)
    if host in LOOPBACK:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(LOOPBACK) + ["testserver"])
    else:
        print(f"ah ui on {host}: anyone who can reach this address can run tools; there is no login", file=sys.stderr)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        if request.method == "POST":
            if request.headers.get(CSRF_HEADER) != "1":
                return Response("missing X-AH-UI header", status_code=403)
            length = request.headers.get("content-length")
            if length is None or int(length) > audio.MAX_BYTES:
                return Response("missing Content-Length or upload too large", status_code=413)
        return await call_next(request)

    @app.exception_handler(StoreError)
    async def store_error(request, err):
        return JSONResponse({"error": str(err)}, status_code=404)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/static/{name}")
    def static(name: str):
        path = (STATIC / name).resolve()
        if path.parent != STATIC.resolve() or not path.is_file():
            raise HTTPException(status_code=404)
        return FileResponse(path)

    @app.get("/api/state")
    def state(selected: str | None = None):
        head = store.head(DEFAULT_PROJECT)
        if head is None:
            return {"head": None, "selected": None, "tree": [], "tools": tools()}
        current = selected or head
        root = store.root(current)
        current_audio = audio_target(store, current)
        parent = store.get(current_audio).parent_id
        return {
            "head": head,
            "selected": current,
            "root": describe(store, root.id),
            "audio": describe(store, current_audio),
            "before": describe(store, parent) if parent else None,
            "analysis": analysis_for(store, current_audio),
            "tree": [{"depth": depth, **describe(store, a.id)} for depth, a in store.tree(root.id)],
            "tools": tools(),
        }

    def tools():
        return [{"name": name, "summary": spec.summary} for name, spec in TOOLS.items()]

    @app.get("/api/artifacts/{artifact_id}")
    def artifact(artifact_id: str):
        info = describe(store, artifact_id)
        if info["kind"] == "json":
            info["content"] = json.loads(store.path(artifact_id).read_text())
        if info["run_id"]:
            run = store.get_run(info["run_id"])
            info["run"] = {"params": run.params, "warnings": run.warnings, "status": run.status}
        return info

    @app.get("/api/artifacts/{artifact_id}/audio")
    def artifact_audio(artifact_id: str, download: bool = False):
        art = store.get(artifact_id)
        if art.kind != "audio":
            raise HTTPException(status_code=400, detail="not audio")
        name = f"{art.label or art.role}_{art.id}.wav" if download else None
        return FileResponse(store.path(artifact_id), media_type="audio/wav", filename=name)

    @app.get("/api/artifacts/{artifact_id}/peaks")
    def artifact_peaks(artifact_id: str, bins: int = 160):
        if store.get(artifact_id).kind != "audio":
            raise HTTPException(status_code=400, detail="not audio")
        return {"peaks": peaks(store.path(artifact_id), min(max(bins, 10), 2000))}

    @app.post("/api/import")
    def upload(file: UploadFile = File(...)):
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in UPLOAD_SUFFIXES:
            raise HTTPException(status_code=422, detail=f"unsupported file type {suffix or '(none)'}")
        with tempfile.TemporaryDirectory(dir=store.scratch) as tmp:
            source = Path(tmp) / f"upload{suffix}"
            with source.open("wb") as handle:
                shutil.copyfileobj(file.file, handle)
            try:
                artifact = runner.ingest(source, label=Path(file.filename).stem[:80])
            except audio.AudioError as err:
                raise HTTPException(status_code=422, detail=str(err)) from None
        return describe(store, artifact.id)

    @app.post("/api/run/{tool}")
    def run(tool: str, body: dict = Body(...)):
        if tool not in TOOLS:
            raise HTTPException(status_code=404, detail=f"unknown tool {tool}")
        params = body.get("params") or {}
        if not isinstance(params, dict):
            raise HTTPException(status_code=422, detail="params must be a JSON object")
        target = audio_target(store, body["artifact"])
        try:
            outcome = runner.run(tool, {"audio": target}, params,
                                 head_role="vocals" if tool == "separate" else None)
        except (RunFailed, ValueError) as err:
            raise HTTPException(status_code=422, detail=str(err)) from None
        audio_outs = [a for a in outcome.artifacts if a.kind == "audio"]
        shown = next((a for a in audio_outs if a.role == "vocals"), audio_outs[0] if audio_outs else outcome.artifacts[0])
        return {"run": outcome.as_dict(), "selected": shown.id}

    @app.post("/api/rollback")
    def rollback():
        return {"head": store.rollback(DEFAULT_PROJECT)}

    @app.post("/api/head")
    def make_head(body: dict = Body(...)):
        target = audio_target(store, body["artifact"])
        store.set_head(target, DEFAULT_PROJECT)
        return {"head": target}

    return app


def launch(store: ArtifactStore, runner: Runner, host: str, port: int) -> None:
    import uvicorn

    print(f"ah ui: http://{'localhost' if host in LOOPBACK else host}:{port}", file=sys.stderr)
    uvicorn.run(create_app(store, runner, host), host=host, port=port, log_level="warning")
