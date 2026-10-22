"""`ah` command line. Every command prints one JSON object to stdout, so the same
CLI serves people, scripts and (later) the agent's skills."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from ah import audio
from ah.config import BACKENDS, load_settings
from ah.registry import TOOLS
from ah.runner import RunFailed, Runner
from ah.store import DEFAULT_PROJECT, ArtifactStore, StoreError


def _emit(payload: Any) -> None:
    json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


def _target(store: ArtifactStore, artifact_id: str | None, project: str) -> str:
    if artifact_id:
        return artifact_id
    head = store.head(project)
    if head is None:
        raise StoreError("no artifact given and the project has no head; run `ah import FILE` first")
    return head


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ah", description="Audio harness: tools over an artifact store")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="head pointer to read and move")
    commands = parser.add_subparsers(dest="command", required=True)

    imp = commands.add_parser("import", help="add an audio file as a canonical WAV artifact")
    imp.add_argument("file")
    imp.add_argument("--label", default="")

    def tool_command(name: str) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=TOOLS[name].summary)
        sub.add_argument("artifact", nargs="?", help="artifact id (default: project head)")
        sub.add_argument("--backend", choices=BACKENDS)
        sub.add_argument("--params", default="{}", help="extra tool parameters as JSON")
        return sub

    tool_command("inspect")
    trim = tool_command("trim")
    trim.add_argument("--start", type=float, dest="start_s", help="seconds from the beginning")
    trim.add_argument("--end", type=float, dest="end_s", help="seconds from the beginning (default: to the end)")
    trim.add_argument("--silence", action="store_true", help="also drop leading and trailing silence")
    trim.add_argument("--threshold-db", type=float, help="silence threshold in dBFS (default -50)")
    sep = tool_command("separate")
    sep.add_argument("--model", help="registry slug, default melband-roformer-kim-vocals")
    sep.add_argument("--head", choices=["vocals", "instrumental"], default="vocals",
                     help="which stem becomes the project head")
    res = tool_command("restore")
    res.add_argument("--chunk-seconds", type=float)
    den = tool_command("denoise")
    den.add_argument("--atten-lim-db", type=float, help="suppress at most this many dB (default: all noise)")
    ana = tool_command("analyze")
    ana.add_argument("--beats", choices=["beat_this", "madmom"])
    ana.add_argument("--no-key", action="store_true")
    ana.add_argument("--no-chords", action="store_true")

    ls = commands.add_parser("ls", help="list recent artifacts")
    ls.add_argument("--limit", type=int, default=30)
    show = commands.add_parser("show", help="artifact, its run and lineage")
    show.add_argument("artifact", nargs="?")
    tree = commands.add_parser("tree", help="version tree from the original upload")
    tree.add_argument("artifact", nargs="?")
    head = commands.add_parser("head", help="print or move the project head")
    head.add_argument("artifact", nargs="?")
    commands.add_parser("rollback", help="move the head to its parent")
    exp = commands.add_parser("export", help="copy an artifact out of the store")
    exp.add_argument("artifact")
    exp.add_argument("out")
    commands.add_parser("tools", help="list tools and backends")

    ui = commands.add_parser("ui", help="start the Gradio page")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=7860)
    srv = commands.add_parser("serve", help="start the GPU server (for Colab)")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8765)
    return parser


def _tool_params(args: argparse.Namespace) -> dict[str, Any]:
    params = json.loads(args.params)
    if not isinstance(params, dict):
        raise ValueError("--params must be a JSON object")
    extra = {
        "model": getattr(args, "model", None),
        "start_s": getattr(args, "start_s", None),
        "end_s": getattr(args, "end_s", None),
        "silence": True if getattr(args, "silence", False) else None,
        "threshold_db": getattr(args, "threshold_db", None),
        "chunk_seconds": getattr(args, "chunk_seconds", None),
        "atten_lim_db": getattr(args, "atten_lim_db", None),
        "beats": getattr(args, "beats", None),
        "key": False if getattr(args, "no_key", False) else None,
        "chords": False if getattr(args, "no_chords", False) else None,
    }
    return params | {k: v for k, v in extra.items() if v is not None}


def main(argv: list[str] | None = None) -> int:
    # Some model libraries print emoji while loading weights. On Windows, the
    # active console encoding may be cp1251 and must not abort a model run.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    args = build_parser().parse_args(argv)
    settings = load_settings()

    if args.command == "serve":
        from ah.server import serve
        serve(args.host, args.port)
        return 0

    store = ArtifactStore(settings.home)
    runner = Runner(store, settings)
    try:
        if args.command == "import":
            _emit(runner.ingest(args.file, label=args.label, project=args.project).as_dict())
        elif args.command in TOOLS:
            artifact_id = _target(store, args.artifact, args.project)
            outcome = runner.run(
                args.command, {"audio": artifact_id}, _tool_params(args),
                backend=args.backend, project=args.project,
                head_role=getattr(args, "head", None),
            )
            _emit(outcome.as_dict())
        elif args.command == "ls":
            _emit([a.as_dict() for a in store.list(args.limit)])
        elif args.command == "show":
            artifact = store.get(_target(store, args.artifact, args.project))
            payload: dict[str, Any] = {"artifact": artifact.as_dict(), "path": str(store.path(artifact.id))}
            if artifact.run_id:
                payload["run"] = store.get_run(artifact.run_id).__dict__
            payload["lineage"] = [a.id for a in store.lineage(artifact.id)]
            if artifact.kind == "json":
                payload["content"] = json.loads(store.path(artifact.id).read_text())
            _emit(payload)
        elif args.command == "tree":
            root = store.root(_target(store, args.artifact, args.project))
            head_id = store.head(args.project)
            _emit([
                {"depth": depth, "id": a.id, "role": a.role, "kind": a.kind,
                 "tool": store.get_run(a.run_id).tool if a.run_id else "import", "head": a.id == head_id}
                for depth, a in store.tree(root.id)
            ])
        elif args.command == "head":
            if args.artifact:
                store.set_head(args.artifact, args.project)
            _emit({"project": args.project, "head": store.head(args.project)})
        elif args.command == "rollback":
            _emit({"project": args.project, "head": store.rollback(args.project)})
        elif args.command == "export":
            _export(store, args.artifact, Path(args.out))
            _emit({"exported": args.artifact, "path": str(Path(args.out).resolve())})
        elif args.command == "tools":
            _emit({name: {"summary": spec.summary, "backend": settings.backend_for(name), "gpu": spec.gpu}
                   for name, spec in TOOLS.items()})
        elif args.command == "ui":
            from ah.ui import launch
            launch(store, runner, args.host, args.port)
    except (StoreError, RunFailed, ValueError, audio.AudioError) as err:
        _emit({"error": str(err)})
        return 1
    return 0


def _export(store: ArtifactStore, artifact_id: str, out: Path) -> None:
    source = store.path(artifact_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == source.suffix.lower():
        shutil.copyfile(source, out)
        return
    ffmpeg = audio.ffmpeg_exe()
    subprocess.run([ffmpeg, "-nostdin", "-loglevel", "error", "-y", "-i", str(source), str(out)], check=True)


if __name__ == "__main__":
    sys.exit(main())
