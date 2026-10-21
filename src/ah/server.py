"""GPU server for the remote backend: runs tools in-process with models kept loaded.

Meant for Colab behind a tunnel, so it is written for a hostile network:
bearer token from AH_SERVER_TOKEN (required, compared in constant time),
an allowlist of tools, upload size caps, one job at a time (a T4 has 16 GB),
and a fresh temp directory per request. Clients never send paths.

No `from __future__ import annotations` here: FastAPI must resolve the
UploadFile annotation of the route defined inside create_app().
"""

import hmac
import json
import os
import shutil
import tempfile
import threading
from pathlib import Path

from ah.contract import ToolRequest, execute
from ah.registry import TOOLS, instantiate
from ah.transport import pack_result

MAX_UPLOAD_BYTES = 300 * 1024 * 1024
SERVED_TOOLS = tuple(os.environ.get("AH_SERVER_TOOLS", "separate,restore,analyze").split(","))


def create_app():
    from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
    from fastapi.responses import Response

    token = os.environ.get("AH_SERVER_TOKEN", "")
    if len(token) < 24:
        raise SystemExit("set AH_SERVER_TOKEN to a random string of at least 24 characters")
    unknown = [name for name in SERVED_TOOLS if name not in TOOLS]
    if unknown:
        raise SystemExit(f"AH_SERVER_TOOLS has unknown tools: {unknown}")

    app = FastAPI(title="audio-harness GPU server", docs_url=None, redoc_url=None, openapi_url=None)
    lock = threading.Lock()
    loaded: dict = {}

    def check(authorization: str | None) -> None:
        supplied = (authorization or "").removeprefix("Bearer ").encode()
        if not hmac.compare_digest(supplied, token.encode()):
            raise HTTPException(status_code=401, detail="bad token")

    @app.middleware("http")
    async def limit_size(request: Request, call_next):
        length = request.headers.get("content-length")
        if length is None or int(length) > MAX_UPLOAD_BYTES:
            return Response("missing Content-Length or upload too large", status_code=413)
        return await call_next(request)

    @app.get("/v1/health")
    def health(authorization: str | None = Header(default=None)):
        check(authorization)
        try:
            import torch
            gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        except ImportError:
            gpu = None
        return {"tools": list(SERVED_TOOLS), "gpu": gpu, "loaded": list(loaded)}

    @app.post("/v1/tools/{name}")
    def run_tool(
        name: str,
        audio: UploadFile = File(...),
        params: str = Form("{}"),
        authorization: str | None = Header(default=None),
    ):
        check(authorization)
        if name not in SERVED_TOOLS:
            raise HTTPException(status_code=404, detail="tool not served here")
        try:
            parsed = json.loads(params)
        except json.JSONDecodeError:
            raise HTTPException(status_code=422, detail="params must be JSON") from None
        if not isinstance(parsed, dict):
            raise HTTPException(status_code=422, detail="params must be a JSON object")

        work = Path(tempfile.mkdtemp(prefix=f"ah-{name}-"))
        try:
            source = work / "input.wav"
            with source.open("wb") as handle:
                shutil.copyfileobj(audio.file, handle)
            request = ToolRequest(tool=name, inputs={"audio": str(source)}, params=parsed,
                                  out_dir=str(work / "out"))
            with lock:
                if name not in loaded:
                    loaded[name] = instantiate(TOOLS[name])
                result = execute(loaded[name], request)
            if result.error:
                raise HTTPException(status_code=500, detail=result.error)
            return Response(pack_result(result), media_type="application/zip")
        finally:
            shutil.rmtree(work, ignore_errors=True)

    return app


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port, log_level="info")
