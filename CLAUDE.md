# CLAUDE.md

Audio harness for an LLM agent: audio tools behind the `ah` CLI with a versioned artifact store. Read `README.md` first: it has the status table, what was verified on real weights, the decisions taken and the ordered "Что дальше" list.

## Working with the owner

- The owner is a programmer / DevOps / security engineer and writes in Russian. Answer in Russian, be blunt, suggest technologies, but leave design decisions to them.
- No eval/metrics work unless asked.
- `tools/denoise` and the denoise-vs-Sidon benchmark belong to a colleague. Don't create or edit them; keep the tool contract in `src/ah/contract.py` stable.

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

- separate and restore are 4–9x slower than real time on CPU; use Colab T4 (`serve/colab_gpu_server.ipynb`) for real material.
- Beat This! downloads its checkpoint from cloud.cp.jku.at, which was returning 503. analyze falls back to madmom with a warning; `AH_BEAT_THIS_CHECKPOINT` points at a local file.
- Kim RoFormer weights and madmom models are CC BY-NC-SA 4.0: non-commercial only.
- Anything transcribed from audio that reaches an agent's context is untrusted input (prompt injection).
