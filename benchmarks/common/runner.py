"""Tool runs for benchmarks through the real harness path, cached on disk.

Calls go through ah.runner.Runner on a throwaway store (a temp AH_HOME), so a benchmark
exercises what the CLI and the agent do: decoding to 44.1 kHz stereo, the tool's own env,
the backend from ah.toml. Outputs are copied to <cache>/<tool>/<key>/ where the key is
sha256 of (input file sha256, tool, params), so the next step of a chain, a resumed Colab
run or a rescore never runs separate twice on the same file.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ah.config import Settings, load_settings
from ah.runner import Runner
from ah.store import ArtifactStore

from common.manifest import sha256


@dataclass
class Outputs:
    tool: str
    params: dict[str, Any]
    input_sha256: str
    files: dict[str, Path]  # role -> file in the cache
    seconds: float  # the tool call only, without decoding the input into the store
    backend: str
    warnings: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    cached: bool = False


def cache_key(input_sha: str, tool: str, params: dict[str, Any]) -> str:
    return hashlib.sha256(f"{input_sha}\n{tool}\n{json.dumps(params, sort_keys=True)}".encode()).hexdigest()[:32]


class CachedRunner:
    def __init__(self, cache: Path, settings: Settings | None = None, device: str | None = None):
        self.cache = Path(cache)
        self._tmp = tempfile.TemporaryDirectory(prefix="ah-bench-")
        self.settings = settings or load_settings()
        self.settings.home = Path(self._tmp.name) / "home"
        if device:
            self.settings.device = device
        self.store = ArtifactStore(self.settings.home)
        self.runner = Runner(self.store, self.settings)

    def __enter__(self) -> CachedRunner:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._tmp.cleanup()

    def run(self, tool: str, audio: Path, params: dict[str, Any] | None = None) -> Outputs:
        params = params or {}
        input_sha = sha256(audio)
        dest = self.cache / tool / cache_key(input_sha, tool, params)
        if (dest / "outputs.json").is_file():
            return self._load(dest, cached=True)
        artifact = self.runner.ingest(audio, label=Path(audio).name)
        started = time.perf_counter()
        outcome = self.runner.run(tool, {"audio": artifact.id}, params, project=None)
        seconds = time.perf_counter() - started
        # Written next to dest and renamed at the end, so an interrupted run never leaves a half entry.
        partial = dest.with_name(dest.name + ".partial")
        shutil.rmtree(partial, ignore_errors=True)
        partial.mkdir(parents=True)
        files = {}
        for a in outcome.artifacts:
            name = f"{a.role}.{a.ext.lstrip('.')}"
            shutil.copyfile(self.store.path(a.id), partial / name)
            files[a.role] = name
        (partial / "outputs.json").write_text(json.dumps({
            "tool": tool, "params": params, "input_sha256": input_sha, "input_name": Path(audio).name, "files": files,
            "seconds": round(seconds, 3), "backend": outcome.backend, "warnings": outcome.warnings, "meta": outcome.meta,
        }, indent=1, ensure_ascii=False, default=str))
        shutil.rmtree(dest, ignore_errors=True)
        partial.rename(dest)
        return self._load(dest, cached=False)

    @staticmethod
    def _load(dest: Path, cached: bool) -> Outputs:
        raw = json.loads((dest / "outputs.json").read_text())
        return Outputs(tool=raw["tool"], params=raw["params"], input_sha256=raw["input_sha256"],
                       files={role: dest / name for role, name in raw["files"].items()}, seconds=raw["seconds"],
                       backend=raw["backend"], warnings=raw["warnings"], meta=raw["meta"], cached=cached)
