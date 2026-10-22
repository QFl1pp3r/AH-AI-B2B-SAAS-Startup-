from __future__ import annotations

import shutil
import tempfile
import uuid
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .model import DeepFilterEngine, ModelUnavailableError

app = FastAPI(title="Audio Harness Denoise", version="0.1.0")


@lru_cache(maxsize=1)
def engine() -> DeepFilterEngine:
    return DeepFilterEngine()


@app.get("/health")
def health() -> dict[str, str]:
    """Report whether the service can load the denoising backend."""
    try:
        engine().load()
    except ModelUnavailableError as exc:
        return {"status": "degraded", "detail": str(exc)}
    return {"status": "ok", "model": "DeepFilterNet"}


@app.post("/denoise", response_class=FileResponse)
async def denoise(file: UploadFile = File(...)) -> FileResponse:
    """Denoise one uploaded audio file and return a WAV file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="A file name is required.")

    suffix = Path(file.filename).suffix.lower() or ".wav"
    with tempfile.TemporaryDirectory(prefix="audio-harness-") as temp_dir:
        workdir = Path(temp_dir)
        source = workdir / f"input{suffix}"
        output = workdir / "enhanced.wav"
        with source.open("wb") as destination:
            shutil.copyfileobj(file.file, destination)

        try:
            engine().denoise(source, output)
        except ModelUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Could not process audio: {exc}") from exc

        # FileResponse streams after the endpoint returns, so keep a durable copy.
        result_dir = Path("tools/denoise/runtime")
        result_dir.mkdir(parents=True, exist_ok=True)
        durable_output = result_dir / f"{uuid.uuid4().hex}_enhanced.wav"
        shutil.copy2(output, durable_output)

    return FileResponse(
        durable_output,
        media_type="audio/wav",
        filename=f"{Path(file.filename).stem}_enhanced.wav",
    )
