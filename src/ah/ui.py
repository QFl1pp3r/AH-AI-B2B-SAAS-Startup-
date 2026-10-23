"""Local web page: upload a track, tell the agent what to do, confirm its plan, compare before/after.

A small JSON API over the same Runner and ArtifactStore the CLI uses, the agent
(ah.agent) for plans, and a static page in ah/static. There are no manual tool
buttons: the CLI is the manual path. The page is for this machine:
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
from ah.agent import Agent, PlanError, analysis_for, audio_target
from ah.llm import LLM, LLMError, configured, make_llm
from ah.runner import Runner
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


def describe(store: ArtifactStore, artifact_id: str) -> dict[str, Any]:
    artifact = store.get(artifact_id)
    info = artifact.as_dict()
    info["tool"] = store.get_run(artifact.run_id).tool if artifact.run_id else "import"
    return info


def lyrics_for(store: ArtifactStore, artifact_id: str) -> str | None:
    """Id of the newest transcript of this version or its nearest ancestor. The page fetches the text itself."""
    for node in store.lineage(artifact_id):
        found = [c for c in store.children(node.id) if c.kind == "json" and c.role == "lyrics"]
        if found:
            return found[-1].id
    return None


def create_app(store: ArtifactStore, runner: Runner, host: str = "127.0.0.1", llm: LLM | None = None):
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

    agent = Agent(store, runner, llm)
    cfg = runner.settings.agent

    @app.exception_handler(StoreError)
    async def store_error(request, err):
        return JSONResponse({"detail": str(err)}, status_code=404)

    @app.exception_handler(LLMError)
    async def llm_error(request, err):
        return JSONResponse({"detail": str(err)}, status_code=502)

    @app.exception_handler(PlanError)
    async def plan_error(request, err):
        return JSONResponse({"detail": str(err)}, status_code=422)

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
            return {"head": None, "selected": None, "tree": []}
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
            "lyrics": lyrics_for(store, current_audio),
            "tree": [{"depth": depth, **describe(store, a.id)} for depth, a in store.tree(root.id)],
        }

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

    @app.get("/api/agent")
    def agent_info():
        return {"configured": llm is not None, "provider": cfg.provider, "model": cfg.model,
                "key_env": cfg.api_key_env, "base_url": cfg.base_url}

    @app.get("/api/agent/chat")
    def agent_chat(artifact: str):
        return {"plans": agent.chat(store.root(artifact).id), "busy": agent.busy.locked()}

    @app.post("/api/agent/message")
    def agent_message(body: dict = Body(...)):
        return agent.message(body["artifact"], str(body.get("text", ""))).as_dict()

    @app.get("/api/agent/plans/{plan_id}")
    def agent_plan(plan_id: str):
        plan = agent.plans.get(plan_id)
        if plan is None:
            raise HTTPException(status_code=404, detail="no such plan")
        return plan.as_dict()

    @app.post("/api/agent/plans/{plan_id}/run")
    def agent_run(plan_id: str):
        return agent.start(plan_id).as_dict()

    @app.post("/api/agent/plans/{plan_id}/cancel")
    def agent_cancel(plan_id: str):
        return agent.cancel(plan_id).as_dict()

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

    cfg = runner.settings.agent
    llm = make_llm(cfg) if configured(cfg) else None
    if llm is None:
        print(f"ah ui: agent is off: set {cfg.api_key_env} (or [agent] in ah.toml)", file=sys.stderr)
    print(f"ah ui: http://{'localhost' if host in LOOPBACK else host}:{port}", file=sys.stderr)
    uvicorn.run(create_app(store, runner, host, llm), host=host, port=port, log_level="warning")
