"""Gradio page: upload, run a tool on the selected version, listen, roll back.

No agent here yet: this is the manual path through the same runner the CLI uses.
"""

from __future__ import annotations

import json
from typing import Any

from ah.config import BACKENDS
from ah.registry import TOOLS
from ah.runner import RunFailed, Runner
from ah.store import DEFAULT_PROJECT, ArtifactStore, StoreError


def _choices(store: ArtifactStore) -> list[tuple[str, str]]:
    choices = []
    for artifact in store.list(200):
        tool = store.get_run(artifact.run_id).tool if artifact.run_id else "import"
        choices.append((f"{artifact.id} · {tool} · {artifact.role} · {artifact.label}", artifact.id))
    return choices


def _tree_markdown(store: ArtifactStore, artifact_id: str | None) -> str:
    if not artifact_id:
        return "_Нет версий_"
    head = store.head(DEFAULT_PROJECT)
    lines = []
    for depth, artifact in store.tree(store.root(artifact_id).id):
        tool = store.get_run(artifact.run_id).tool if artifact.run_id else "import"
        mark = " **← head**" if artifact.id == head else ""
        indent = "\u00a0" * 6 * depth + ("└ " if depth else "")
        lines.append(f"{indent}`{artifact.id}` {tool} → {artifact.role}{mark}")
    return "  \n".join(lines)


def _details(store: ArtifactStore, artifact_id: str | None) -> tuple[Any, Any, str]:
    if not artifact_id:
        return None, None, _tree_markdown(store, None)
    artifact = store.get(artifact_id)
    path = str(store.path(artifact_id))
    if artifact.kind == "audio":
        return path, artifact.as_dict(), _tree_markdown(store, artifact_id)
    content = json.loads(store.path(artifact_id).read_text())
    return None, {"artifact": artifact.as_dict(), "content": content}, _tree_markdown(store, artifact_id)


def launch(store: ArtifactStore, runner: Runner, host: str, port: int) -> None:
    import gradio as gr

    def refresh(selected=None):
        selected = selected or store.head(DEFAULT_PROJECT)
        player, info, tree = _details(store, selected)
        return gr.update(choices=_choices(store), value=selected), player, info, tree

    def on_upload(path):
        if not path:
            return refresh()
        artifact = runner.ingest(path)
        return refresh(artifact.id)

    def on_run(tool, selected, backend, params_text):
        if not selected:
            raise gr.Error("Сначала загрузите файл или выберите версию")
        try:
            params = json.loads(params_text or "{}")
            outcome = runner.run(tool, {"audio": selected}, params, backend=backend or None,
                                 head_role="vocals" if tool == "separate" else None)
        except (RunFailed, StoreError, ValueError) as err:
            raise gr.Error(str(err)) from err
        for warning in outcome.warnings:
            gr.Warning(warning)
        audio_out = next((a for a in outcome.artifacts if a.role == "vocals"), None) if tool == "separate" else None
        audio_out = audio_out or next((a for a in outcome.artifacts if a.kind == "audio"), outcome.artifacts[0])
        return refresh(audio_out.id)

    def on_rollback():
        try:
            return refresh(store.rollback(DEFAULT_PROJECT))
        except StoreError as err:
            raise gr.Error(str(err)) from err

    def on_head(selected):
        if selected and store.get(selected).kind == "audio":
            store.set_head(selected, DEFAULT_PROJECT)
        return refresh(selected)

    with gr.Blocks(title="Audio Harness") as page:
        gr.Markdown("## Audio Harness\nКаждый шаг создаёт новую версию; откат двигает head к родителю.")
        with gr.Row():
            with gr.Column(scale=1):
                upload = gr.File(label="Загрузить аудио", type="filepath")
                version = gr.Dropdown(label="Версия", choices=_choices(store), value=store.head(DEFAULT_PROJECT))
                backend = gr.Dropdown(label="Бэкенд (пусто = из ah.toml)", choices=["", *BACKENDS], value="")
                params = gr.Textbox(label="Параметры JSON", value="{}", lines=2)
                buttons = {name: gr.Button(f"{name}: {spec.summary}") for name, spec in TOOLS.items()}
                with gr.Row():
                    rollback = gr.Button("Откат")
                    make_head = gr.Button("Сделать head")
            with gr.Column(scale=2):
                player = gr.Audio(label="Прослушивание", type="filepath", interactive=False)
                info = gr.JSON(label="Метаданные / результат")
                tree = gr.Markdown()

        outputs = [version, player, info, tree]
        upload.upload(on_upload, upload, outputs)
        version.change(lambda s: _details(store, s), version, [player, info, tree])
        for name, button in buttons.items():
            button.click(lambda s, b, p, n=name: on_run(n, s, b, p), [version, backend, params], outputs)
        rollback.click(on_rollback, None, outputs)
        make_head.click(on_head, version, outputs)
        page.load(lambda: refresh(), None, outputs)

    page.launch(server_name=host, server_port=port, allowed_paths=[str(store.blobs)])
