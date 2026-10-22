"""Which tools exist, where their code lives and what they take."""

from __future__ import annotations

import importlib
from dataclasses import dataclass

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


TOOLS: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in (
        ToolSpec(
            name="inspect", module="ah.builtin", factory="InspectTool", project=None,
            inputs=("audio",), summary="Длительность, уровни, клиппинг, тишина",
        ),
        ToolSpec(
            name="trim", module="ah.builtin", factory="TrimTool", project=None,
            inputs=("audio",), summary="Обрезка по времени и по тишине на краях",
        ),
        ToolSpec(
            name="separate", module="ah_separate", factory="SeparateTool", project="separate",
            inputs=("audio",), summary="Вокал и инструментал (Mel-Band RoFormer)", gpu=True,
        ),
        ToolSpec(
            name="restore", module="ah_restore", factory="RestoreTool", project="restore",
            inputs=("audio",), summary="Восстановление речи: шум, реверберация, полоса (Sidon)", gpu=True,
        ),
        ToolSpec(
            name="denoise", module="ah_denoise", factory="DenoiseTool", project="denoise",
            inputs=("audio",), summary="Шумоподавление речи, быстрое (DeepFilterNet3)",
        ),
        ToolSpec(
            name="analyze", module="ah_analyze", factory="AnalyzeTool", project="analyze",
            inputs=("audio",), summary="BPM, биты, сильные доли, тональность, аккорды",
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
