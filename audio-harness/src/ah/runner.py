"""Maps artifact ids to files, runs a tool on some backend and stores what it returns.

Backends:
- local: the tool's own uv environment in tools/<name>, as a subprocess;
- inprocess: imported into the current interpreter (Colab, the GPU server, tests);
- remote: POST to the GPU server (serve/server.py), typically Colab behind a tunnel.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from ah import audio
from ah.config import Settings
from ah.contract import RESULT_FILE, Tool, ToolRequest, ToolResult, execute
from ah.registry import ToolSpec, get_spec, instantiate
from ah.store import DEFAULT_PROJECT, Artifact, ArtifactStore
from ah.transport import unpack_result


class RunFailed(RuntimeError):
    def __init__(self, run_id: str, message: str):
        super().__init__(f"run {run_id} failed: {message}")
        self.run_id = run_id


@dataclass
class RunOutcome:
    run_id: str
    tool: str
    backend: str
    artifacts: list[Artifact]
    meta: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "tool": self.tool, "backend": self.backend,
            "artifacts": [a.as_dict() for a in self.artifacts],
            "meta": self.meta, "warnings": self.warnings,
        }


class Runner:
    def __init__(self, store: ArtifactStore, settings: Settings):
        self.store = store
        self.settings = settings
        self._loaded: dict[str, Tool] = {}

    # --- import ----------------------------------------------------------

    def ingest(self, source: str | Path, *, label: str = "", project: str = DEFAULT_PROJECT) -> Artifact:
        """Decode any supported file into a canonical WAV artifact and make it the head."""
        source = Path(source)
        data = audio.load_canonical(source)
        with tempfile.TemporaryDirectory(dir=self.store.scratch) as tmp:
            target = Path(tmp) / "audio.wav"
            audio.write_wav(target, data)
            meta = audio.stats(data, audio.CANONICAL_SR) | {"source_name": source.name}
            artifact = self.store.add(target, kind="audio", role="original", label=label or source.stem, meta=meta)
        self.store.set_head(artifact.id, project)
        return artifact

    # --- run -------------------------------------------------------------

    def run(
        self,
        tool: str,
        inputs: dict[str, str],
        params: dict[str, Any] | None = None,
        *,
        backend: str | None = None,
        project: str | None = DEFAULT_PROJECT,
        head_role: str | None = None,
    ) -> RunOutcome:
        spec = get_spec(tool)
        params = params or {}
        missing = set(spec.inputs) - set(inputs)
        if missing:
            raise ValueError(f"{tool} needs inputs: {', '.join(sorted(missing))}")
        backend = backend or self.settings.backend_for(tool)
        if spec.project is None and backend == "local":
            backend = "inprocess"

        run_id = self.store.start_run(tool, backend, inputs, params)
        work = Path(tempfile.mkdtemp(prefix=f"{run_id}-", dir=self.store.scratch))
        try:
            request = ToolRequest(
                tool=tool,
                inputs={role: str(self.store.path(artifact_id)) for role, artifact_id in inputs.items()},
                params=params,
                out_dir=str(work / "out"),
                device=self.settings.device,
            )
            if backend == "inprocess":
                result = execute(self._tool(spec), request)
            elif backend == "local":
                result = self._run_subprocess(spec, request, work)
            elif backend == "remote":
                result = self._run_remote(spec, request, work)
            else:
                raise ValueError(f"unknown backend {backend!r}")

            if result.error:
                self.store.finish_run(run_id, error=result.error, warnings=result.warnings)
                raise RunFailed(run_id, result.error)
            parent = inputs.get("audio") or next(iter(inputs.values()))
            artifacts = [self._store_output(output, parent, run_id) for output in result.outputs]
            self.store.finish_run(run_id, meta=result.meta, warnings=result.warnings)
        except RunFailed:
            raise
        except Exception as err:
            self.store.finish_run(run_id, error=f"{type(err).__name__}: {err}")
            raise RunFailed(run_id, str(err)) from err
        finally:
            shutil.rmtree(work, ignore_errors=True)

        if project is not None:
            head = next((a for a in artifacts if a.kind == "audio" and (head_role is None or a.role == head_role)), None)
            if head is not None:
                self.store.set_head(head.id, project)
        return RunOutcome(run_id=run_id, tool=tool, backend=backend, artifacts=artifacts,
                          meta=result.meta, warnings=result.warnings)

    def _store_output(self, output, parent_id: str, run_id: str) -> Artifact:
        path = Path(output.path)
        if output.kind == "audio":
            # Every stored audio artifact is canonical, whatever rate the tool or server produced.
            data = audio.load_canonical(path)
            canonical = path.with_name(f"{path.stem}.canonical.wav")
            audio.write_wav(canonical, data)
            meta = audio.stats(data, audio.CANONICAL_SR) | output.meta
            return self.store.add(canonical, kind="audio", role=output.role, parent_id=parent_id,
                                  run_id=run_id, meta=meta)
        json.loads(path.read_text())
        return self.store.add(path, kind="json", role=output.role, parent_id=parent_id,
                              run_id=run_id, meta=output.meta)

    # --- backends --------------------------------------------------------

    def _tool(self, spec: ToolSpec) -> Tool:
        if spec.name not in self._loaded:
            self._loaded[spec.name] = instantiate(spec)
        return self._loaded[spec.name]

    def _run_subprocess(self, spec: ToolSpec, request: ToolRequest, work: Path) -> ToolResult:
        request_path = work / "request.json"
        request_path.write_text(request.model_dump_json())
        override = os.environ.get(f"AH_PYTHON_{spec.name.upper()}")
        if override:
            command = [override, "-m", spec.module, str(request_path)]
        else:
            uv = shutil.which("uv")
            if uv is None:
                raise RuntimeError("uv is not installed; install it or set backend = 'inprocess'")
            project = self.settings.tools_root / spec.project
            command = [uv, "run", "--quiet", "--project", str(project), "python", "-m", spec.module, str(request_path)]
        completed = subprocess.run(command, stdout=sys.stderr, stderr=sys.stderr, check=False)
        result_path = Path(request.out_dir) / RESULT_FILE
        if not result_path.is_file():
            raise RuntimeError(f"{spec.name} exited with code {completed.returncode} and wrote no result")
        return ToolResult.model_validate_json(result_path.read_text())

    def _run_remote(self, spec: ToolSpec, request: ToolRequest, work: Path) -> ToolResult:
        remote = self.settings.remote
        if not remote.url:
            raise RuntimeError("remote backend selected but [remote].url / AH_REMOTE_URL is empty")
        if not remote.token:
            raise RuntimeError(f"remote backend needs a token in ${remote.token_env}")
        handles = {role: open(path, "rb") for role, path in request.inputs.items()}
        try:
            response = httpx.post(
                f"{remote.url.rstrip('/')}/v1/tools/{spec.name}",
                headers={"Authorization": f"Bearer {remote.token}"},
                files={role: (Path(handle.name).name, handle, "audio/wav") for role, handle in handles.items()},
                data={"params": json.dumps(request.params)},
                timeout=remote.timeout_s,
            )
        finally:
            for handle in handles.values():
                handle.close()
        if response.status_code != 200:
            raise RuntimeError(f"server answered {response.status_code}: {response.text[:500]}")
        return unpack_result(response.content, Path(request.out_dir))
