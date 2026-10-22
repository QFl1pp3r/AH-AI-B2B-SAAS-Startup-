"""Run the harness agent on the benchmark tasks and record every plan it makes.

--mode plan: the real Agent.message (system prompt, tool catalog from the registry,
validate() and its one retry) with the configured LLM. Only track_data is swapped for
the task's fixture, and nothing runs: the plan is the result. Each task runs k times,
because DeepSeek in thinking mode ignores temperature.

    uv run python benchmarks/agent/run.py --mode plan --model deepseek-flash --thinking enabled \
        --out benchmarks/results/agent-plan-deepseek-flash-thinking

Writes <out>/raw/runs.jsonl (one line per attempt: plan, reply, error, every LLM call with
usage, latency and UTC time); --resume skips attempts already there. Score with score.py.
The LLM comes from ah.toml [agent] and AH_AGENT_* env, --model/--thinking override them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ah.agent
from ah import audio
from ah.agent import Agent, PlanError
from ah.config import Settings, load_settings
from ah.llm import LLMError, configured, make_llm
from ah.runner import Runner
from ah.store import ArtifactStore
from common import manifest

HERE = Path(__file__).parent


class Recorder:
    """Passes calls through and keeps what score.py needs: usage, latency, time, and the raw tool calls."""

    def __init__(self, llm):
        self.llm = llm
        self.name = llm.name
        self.calls = []

    def complete(self, system, messages, tools):
        at = time.time()
        try:
            reply = self.llm.complete(system, messages, tools)
        except LLMError as err:
            self.calls.append({"at": at, "error": str(err)})
            raise
        self.calls.append({"at": at, "usage": reply.usage, "latency_s": reply.latency_s, "text": reply.text,
                           "calls": [{"name": c.name, "input": c.input} for c in reply.calls]})
        return reply


def load_tasks(path: Path) -> list[dict]:
    tasks = manifest.read(path, required=("id", "category", "lang", "request", "track_data", "gold"))
    for task in tasks:
        task["fixture"] = json.loads((HERE / task["track_data"]).read_text())
    return tasks


def attempt(store, runner, llm, track: str, task: dict, k: int) -> dict:
    agent = Agent(store, runner, Recorder(llm))  # a fresh agent per attempt: no chat history
    out = {"task": task["id"], "attempt": k, "ok": True, "error": None, "reply": None, "steps": None}
    try:
        plan = agent.message(track, task["request"])
        out["reply"] = plan.reply
        out["steps"] = [{"tool": s.tool, "input": s.input, "params": s.params} for s in plan.steps]
    except (LLMError, PlanError) as err:
        out["ok"], out["error"] = False, str(err)
    out["llm_calls"] = agent.llm.calls
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["plan"], default="plan")
    parser.add_argument("--tasks", type=Path, default=HERE / "tasks/plan.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--thinking", choices=["enabled", "disabled"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.model:
        os.environ["AH_AGENT_MODEL"] = args.model
    if args.thinking:
        os.environ["AH_AGENT_THINKING"] = args.thinking
    cfg = load_settings().agent
    if not configured(cfg):
        sys.exit(f"нет ключа в {cfg.api_key_env}")
    llm = make_llm(cfg)
    tasks = load_tasks(args.tasks)[:args.limit]

    raw = args.out / "raw" / "runs.jsonl"
    raw.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and raw.exists():
        done = {(r["task"], r["attempt"]) for r in map(json.loads, raw.read_text().splitlines())}
    elif raw.exists():
        raw.unlink()
    (args.out / "raw" / "config.json").write_text(json.dumps({
        "llm": llm.name, "model": cfg.model, "provider": cfg.provider, "base_url": cfg.base_url,
        "thinking": cfg.thinking or "provider default", "reasoning_effort": cfg.reasoning_effort or "provider default",
        "k": args.k, "tasks": str(args.tasks), "tasks_sha256": manifest.sha256(args.tasks)}, indent=1))

    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp) / "home"
        store = ArtifactStore(home)
        runner = Runner(store, Settings(home=home))
        wav = Path(tmp) / "x.wav"
        audio.write_wav(wav, np.zeros((441, 2), np.float32))
        # One stand-in track per task; the agent reads its metadata only through track_data, swapped here.
        tracks = {t["id"]: runner.ingest(wav, label=t["fixture"]["file_name"]).id for t in tasks}
        fixtures = {tracks[t["id"]]: t["fixture"] for t in tasks}
        ah.agent.track_data = lambda _store, artifact_id: fixtures[artifact_id]

        jobs = [(t, k) for t in tasks for k in range(args.k) if (t["id"], k) not in done]
        print(f"\n{llm.name} thinking={cfg.thinking or 'default'}: {len(jobs)} attempts", file=sys.stderr)
        with ThreadPoolExecutor(args.workers) as pool, raw.open("a") as f:
            futures = [pool.submit(attempt, store, runner, llm, tracks[t["id"]], t, k) for t, k in jobs]
            for n, future in enumerate(as_completed(futures), 1):
                result = future.result()
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                if n % 10 == 0 or n == len(jobs):
                    print(f"[{n}/{len(jobs)}]", file=sys.stderr)


if __name__ == "__main__":
    main()
