"""Tools that need nothing beyond the core environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ah import audio
from ah.contract import ToolOutput, ToolResult, write_json


class InspectTool:
    name = "inspect"

    def load(self, device: str) -> None:
        pass

    def run(self, inputs: dict[str, Path], params: dict[str, Any], out_dir: Path) -> ToolResult:
        data, sr = audio.decode(inputs["audio"])
        report = audio.stats(data, sr)
        target = out_dir / "inspect.json"
        write_json(target, report)
        return ToolResult(
            tool=self.name,
            outputs=[ToolOutput(role="report", path=str(target), kind="json", meta=report)],
            meta=report,
        )


class TrimParams(BaseModel, extra="forbid"):
    start_s: float = Field(0.0, ge=0)
    end_s: float | None = Field(None, gt=0)
    silence: bool = False  # also drop leading/trailing silence inside the range
    threshold_db: float = Field(-50.0, le=0)
    pad_ms: float = Field(50.0, ge=0)  # silence kept around the detected sound
    fade_ms: float = Field(5.0, ge=0)  # only on edges that were actually cut, against clicks


class TrimTool:
    """Cuts on decoded samples, so the cut is sample-exact for any source format."""

    name = "trim"
    frame_ms = 10

    def load(self, device: str) -> None:
        pass

    def run(self, inputs: dict[str, Path], params: dict[str, Any], out_dir: Path) -> ToolResult:
        p = TrimParams(**params)
        data, sr = audio.decode(inputs["audio"])
        total = data.shape[0]
        start = min(round(p.start_s * sr), total)
        end = total if p.end_s is None else min(round(p.end_s * sr), total)
        if start >= end:
            raise ValueError(f"empty range: start {p.start_s}s, end {p.end_s}s, audio {total / sr:.3f}s")
        if p.silence:
            start, end = self._sound_bounds(data, sr, start, end, p)
        trimmed = data[start:end].copy()
        fade = min(round(p.fade_ms * sr / 1000), trimmed.shape[0] // 2)
        if fade:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)[:, None]
            if start > 0:
                trimmed[:fade] *= ramp
            if end < total:
                trimmed[-fade:] *= ramp[::-1]
        target = out_dir / "trimmed.wav"
        audio.write_wav(target, trimmed, sr)
        # offset_s maps timestamps from later tools (beats, chords) back to the source.
        meta = {
            "offset_s": round(start / sr, 6),
            "end_s": round(end / sr, 6),
            "duration_s": round((end - start) / sr, 6),
            "source_duration_s": round(total / sr, 6),
        }
        return ToolResult(
            tool=self.name,
            outputs=[ToolOutput(role="trimmed", path=str(target), kind="audio", meta=meta)],
            meta=meta,
        )

    def _sound_bounds(self, data: np.ndarray, sr: int, start: int, end: int, p: TrimParams) -> tuple[int, int]:
        """First and last frame above the threshold. Loudest channel per frame, so
        out-of-phase stereo does not cancel into fake silence."""
        frame = max(1, sr * self.frame_ms // 1000)
        segment = data[start:end]
        usable = (segment.shape[0] // frame) * frame
        tail = segment[usable:]
        frames = segment[:usable].reshape(-1, frame, segment.shape[1])
        rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1)).max(axis=1)
        if tail.size:
            rms = np.append(rms, np.sqrt(np.mean(np.square(tail, dtype=np.float64), axis=0)).max())
        loud = np.flatnonzero(rms >= 10 ** (p.threshold_db / 20))
        if loud.size == 0:
            raise ValueError(f"nothing above {p.threshold_db} dBFS in the range")
        pad = round(p.pad_ms * sr / 1000)
        first = start + loud[0] * frame - pad
        last = start + min((loud[-1] + 1) * frame, segment.shape[0]) + pad
        return max(first, start), min(last, end)
