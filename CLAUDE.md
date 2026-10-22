# CLAUDE.md

Audio harness for an LLM agent: audio tools behind the `ah` CLI with a versioned artifact store. Read `README.md` first: it has the status table, what was verified on real weights, the decisions taken and the ordered "До MVP" task list (from the MVP plan PDF). Work it one task at a time and ask the owner before starting the next one.

## Working with the owner

- The owner is a programmer / DevOps / security engineer and writes in Russian. Answer in Russian, be blunt, suggest technologies, but leave design decisions to them.
- No eval/metrics work unless asked.
- Jamal's code: `tools/denoise/src/denoise_service`, `notebooks/`, `scripts/`, `data/splits/` and the denoise-vs-Sidon benchmark. Don't edit them unless the owner asks; keep the tool contract in `src/ah/contract.py` stable. The `ah denoise` adapter (`tools/denoise/src/ah_denoise`) and the pins in `tools/denoise/pyproject.toml` were added at the owner's request on 2026-10-22.

## Commands

```bash
uv sync --all-extras                 # core env with UI and server extras
uv run pytest -q                     # must stay green
uv sync --project tools/<name>       # a tool's own env (separate | restore | analyze)
uv run ah <command> ...              # see README for the list
```

## Conventions

- Each tool lives in `tools/<name>` with its own uv env and implements the `Tool` protocol from `src/ah/contract.py`. Tools never touch the store or artifact ids; the runner maps files in and out. New tools also need a `ToolSpec` in `src/ah/registry.py`.
- All decoding, resampling and format conversion goes through `src/ah/audio.py`. Stored audio is canonical: 44.1 kHz, stereo, float32 WAV.
- Tool results go to `<out_dir>/result.json`, never stdout (model libraries print progress there). CLI output is JSON on stdout, logs on stderr.
- Tool params are pydantic models with `extra="forbid"`.
- Pin git dependencies to a reviewed commit; never load model checkpoints from unofficial mirrors (torch.load is pickle).
- Never commit `ah.toml`, `.ah/` or tokens. Remote tokens live only in env vars.
- Match the existing style: short docstrings that explain why, sparse comments, type hints.

## Gotchas

- `tools/denoise` needs Python 3.11 (DeepFilterLib 0.5.6 has no cp312 wheels) and torch 2.3.1 / numpy<2, as in the Colab benchmark. Don't unpin them.
- separate and restore are 4–9x slower than real time on CPU; use Colab T4 (`serve/colab_gpu_server.ipynb`) for real material.
- Beat This! downloads its checkpoint from cloud.cp.jku.at, which was returning 503. analyze falls back to madmom with a warning; `AH_BEAT_THIS_CHECKPOINT` points at a local file.
- Kim RoFormer weights and madmom models are CC BY-NC-SA 4.0: non-commercial only.
- Anything transcribed from audio that reaches an agent's context is untrusted input (prompt injection).
