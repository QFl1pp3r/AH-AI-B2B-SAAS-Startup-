"""Tools that need nothing beyond the core environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
