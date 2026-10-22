"""Which tools exist, where their code lives and what they take."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

from ah.contract import Tool


@dataclass(frozen=True)
class ToolSpec:
    name: str
    module: str
    factory: str
    project: str | None  # directory under tools/ with its own uv environment; None = core env
    inputs: tuple[str, ...]
    summary: str
    gpu: bool = False
    # For the agent: output roles (audio ones can feed later steps), when to use the tool,
    # and its params as JSON-schema properties. tests/test_agent.py checks them against each Params model.
    outputs: tuple[str, ...] = ()
    hint: str = ""
    params: dict[str, Any] = field(default_factory=dict)


TOOLS: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in (
        ToolSpec(
            name="inspect", module="ah.builtin", factory="InspectTool", project=None,
            inputs=("audio",), summary="Длительность, уровни, клиппинг, тишина",
            outputs=("report",),
            hint="Measures duration, peak and RMS in dBFS, clipping and silence. Cheap; use when the user asks about loudness or clipping.",
        ),
        ToolSpec(
            name="trim", module="ah.builtin", factory="TrimTool", project=None,
            inputs=("audio",), summary="Обрезка по времени и по тишине на краях",
        ),
        ToolSpec(
            name="separate", module="ah_separate", factory="SeparateTool", project="separate",
            inputs=("audio",), summary="Вокал и инструментал (Mel-Band RoFormer)", gpu=True,
            outputs=("vocals", "instrumental"),
            hint="Splits a song into vocals and instrumental. Use before restore, denoise or analysis of one part of a mix. "
                 "Slow on CPU (about 9x real time).",
            params={"model": {"type": "string", "description": "separator registry slug; leave out for the default Kim vocals model"}},
        ),
        ToolSpec(
            name="restore", module="ah_restore", factory="RestoreTool", project="restore",
            inputs=("audio",), summary="Восстановление речи: шум, реверберация, полоса (Sidon)", gpu=True,
            outputs=("restored",),
            hint="Generative speech restoration: removes noise and reverb, restores bandwidth, outputs mono speech. "
                 "Sounds best but shifts the voice more than denoise. Speech only: on a mix run separate first and restore the vocals. "
                 "Slow on CPU (about 4x real time).",
            params={"chunk_seconds": {"type": "number", "exclusiveMinimum": 5, "maximum": 96},
                    "match_peak": {"type": "boolean", "description": "put the input's peak level back (default true)"}},
        ),
        ToolSpec(
            name="denoise", module="ah_denoise", factory="DenoiseTool", project="denoise",
            inputs=("audio",), summary="Шумоподавление речи, быстрое (DeepFilterNet3)",
            outputs=("denoised",),
            hint="Fast speech denoiser, keeps the voice closer to the original than restore. Trained on speech: music and "
                 "singing get suppressed as noise, so on a song denoise the separated vocals, not the mix. Do not denoise "
                 "before speech recognition: it made transcription worse in our benchmark.",
            params={"atten_lim_db": {"type": "number", "exclusiveMinimum": 0, "maximum": 100,
                                     "description": "suppress at most this many dB and keep the rest of the noise; leave out for full suppression"}},
        ),
        ToolSpec(
            name="analyze", module="ah_analyze", factory="AnalyzeTool", project="analyze",
            inputs=("audio",), summary="BPM, биты, сильные доли, тональность, аккорды",
            outputs=("analysis",),
            hint="Tempo (BPM), beats, downbeats, meter, key and chords (major/minor only). Works on mixes and instrumentals; "
                 "chords on a solo melody are meaningless. Key is the least reliable field.",
            params={"beats": {"type": "string", "enum": ["beat_this", "madmom"]},
                    "key": {"type": "boolean"}, "chords": {"type": "boolean"}},
        ),
    )
}


def get_spec(name: str) -> ToolSpec:
    try:
        return TOOLS[name]
    except KeyError:
        raise KeyError(f"unknown tool {name!r}; known: {', '.join(TOOLS)}") from None


def instantiate(spec: ToolSpec) -> Tool:
    """Import the tool into the current interpreter (in-process and server backends)."""
    module = importlib.import_module(spec.module)
    return getattr(module, spec.factory)()
