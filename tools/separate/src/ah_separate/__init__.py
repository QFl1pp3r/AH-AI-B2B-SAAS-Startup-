"""Vocal / instrumental separation with Mel-Band RoFormer (melband-roformer-infer).

The Kim vocals checkpoint (~913 MB) downloads on first use and is sha256-checked
by the library. Models are 44.1 kHz; the canonical artifacts already are, and
anything else is converted through ah.audio.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ah import audio
from ah.contract import ToolOutput, ToolResult

MODEL_SR = 44_100
DEFAULT_MODEL = "melband-roformer-kim-vocals"


class Params(BaseModel, extra="forbid"):
    model: str = DEFAULT_MODEL


class SeparateTool:
    name = "separate"

    def __init__(self) -> None:
        self.device = "cpu"
        self._sessions: dict[str, Any] = {}

    def load(self, device: str) -> None:
        self.device = device

    def _session(self, model: str):
        if model not in self._sessions:
            from mel_band_roformer import MelBandRoformerSession

            # Keep one model resident: a T4 has 16 GB and other tools share it.
            for session in self._sessions.values():
                session.close()
            self._sessions = {model: MelBandRoformerSession(model_name=model, device=self.device, progress=False).load()}
        return self._sessions[model]

    def run(self, inputs: dict[str, Path], params: dict[str, Any], out_dir: Path) -> ToolResult:
        options = Params.model_validate(params)
        data, sr = audio.decode(inputs["audio"])
        data = audio.resample(audio.to_channels(data, 2), sr, MODEL_SR)
        session = self._session(options.model)

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "in"
            folder.mkdir()
            audio.write_wav(folder / "mix.wav", data, MODEL_SR)
            manifest = session.infer(folder, store_dir=out_dir, output_format="wav_float32")

        outputs = [
            ToolOutput(role=entry["output_id"], path=entry["output_path"], kind="audio",
                       meta={"model": options.model})
            for entry in manifest
        ]
        warnings = []
        if data.shape[0] / MODEL_SR > 600:
            warnings.append("long input: on CPU separation runs well below real time")
        return ToolResult(tool=self.name, outputs=outputs,
                          meta={"model": options.model, "device": self.device}, warnings=warnings)

