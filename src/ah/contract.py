"""The contract between the CLI and every tool.

A tool receives input files (by role) plus a JSON object of parameters, writes
its outputs into a fresh directory and returns a ToolResult. Tools never touch
the artifact store and never see artifact ids: the runner maps ids to files and
files back to new artifacts. That keeps tools runnable on another machine
(Colab) with nothing but the files.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

from pydantic import BaseModel, Field

RESULT_FILE = "result.json"


class ToolRequest(BaseModel):
    tool: str
    inputs: dict[str, str]
    params: dict[str, Any] = Field(default_factory=dict)
    out_dir: str
    device: str = "auto"


class ToolOutput(BaseModel):
    role: str
    path: str
    kind: Literal["audio", "json"]
    meta: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    tool: str
    outputs: list[ToolOutput] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class Tool(Protocol):
    name: str

    def load(self, device: str) -> None:
        """Load weights once; called before the first run."""

    def run(self, inputs: dict[str, Path], params: dict[str, Any], out_dir: Path) -> ToolResult:
        ...


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def execute(tool: Tool, request: ToolRequest) -> ToolResult:
    """Run one request in-process and always return a result, never raise."""
    out_dir = Path(request.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        tool.load(resolve_device(request.device))
        inputs = {role: Path(path) for role, path in request.inputs.items()}
        result = tool.run(inputs, request.params, out_dir)
    except Exception as err:  # the runner turns this into a failed run record
        traceback.print_exc(file=sys.stderr)
        result = ToolResult(tool=request.tool, error=f"{type(err).__name__}: {err}")
    return result


def main(factory: Callable[[], Tool]) -> None:
    """Entry point for `python -m ah_<tool> request.json` inside a tool's own environment.

    The result goes to <out_dir>/result.json rather than stdout, because model
    libraries print progress to stdout.
    """
    if len(sys.argv) != 2:
        sys.exit("usage: python -m ah_<tool> request.json")
    request = ToolRequest.model_validate_json(Path(sys.argv[1]).read_text())
    result = execute(factory(), request)
    (Path(request.out_dir) / RESULT_FILE).write_text(result.model_dump_json(indent=2))
    sys.exit(0 if result.error is None else 1)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
