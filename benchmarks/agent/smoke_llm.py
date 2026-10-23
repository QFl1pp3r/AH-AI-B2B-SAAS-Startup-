"""Live check of the planning LLM (DeepSeek by default) on the three paths the agent takes.

1. plan: one request on a fresh track.
2. history: a second request on the same track, so earlier turns go back as plain assistant text.
3. retry: the first answer is swapped for a plan validate() rejects, so the agent sends it back once.

With thinking on and tools in the request, DeepSeek wants reasoning_content from earlier assistant turns
and may answer 400 on 2 and 3; this script is how we find out. Nothing runs tools, only the planner.

    DEEPSEEK_API_KEY=... uv run python benchmarks/agent/smoke_llm.py > smoke.json
    AH_AGENT_THINKING=disabled uv run python benchmarks/agent/smoke_llm.py

AH_AGENT_* env vars override the DeepSeek defaults below (see ah.config).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from ah import audio
from ah.agent import Agent, PlanError
from ah.config import Settings, load_settings
from ah.llm import LLMError, Reply, ToolCall, configured, make_llm
from ah.runner import Runner
from ah.store import ArtifactStore

DEFAULTS = {"PROVIDER": "openai", "MODEL": "deepseek-flash", "BASE_URL": "https://api.deepseek.com",
            "API_KEY_ENV": "DEEPSEEK_API_KEY"}
REJECTED = {"reply": "x", "steps": [{"tool": "no_such_tool", "input": "track", "why": "forced retry"}]}


class Recording:
    """Logs every call; with sabotage, the first real answer is replaced by a plan validate() rejects."""

    def __init__(self, llm, sabotage=False):
        self.llm = llm
        self.name = llm.name
        self.sabotage = sabotage
        self.calls = []

    def complete(self, system, messages, tools):
        reply = self.llm.complete(system, messages, tools)
        self.calls.append({"messages": len(messages), "tools": [t["name"] for t in tools], "usage": reply.usage,
                           "latency_s": reply.latency_s, "text": reply.text[:300], "calls": [{"name": c.name, "input": c.input} for c in reply.calls]})
        if self.sabotage:
            self.sabotage = False
            return Reply(text="", calls=[ToolCall("submit_plan", REJECTED)], usage=reply.usage, latency_s=reply.latency_s)
        return reply


def scenario(name, agent, track, request):
    out = {"scenario": name, "request": request, "ok": True, "error": None, "plan": None}
    start = len(agent.llm.calls)  # plan and history share one agent, so one call log
    try:
        plan = agent.message(track, request)
        out["plan"] = {"reply": plan.reply, "steps": [{"tool": s.tool, "input": s.input, "params": s.params} for s in plan.steps]}
    except (LLMError, PlanError) as err:
        out["ok"], out["error"] = False, str(err)
    out["llm_calls"] = agent.llm.calls[start:]
    if name == "retry" and out["ok"] and len(out["llm_calls"]) != 2:
        out["ok"], out["error"] = False, f"ожидали 2 вызова LLM, было {len(out['llm_calls'])}"
    return out


def main():
    for key, value in DEFAULTS.items():
        os.environ.setdefault(f"AH_AGENT_{key}", value)
    cfg = load_settings().agent
    if not configured(cfg):
        sys.exit(f"нет ключа в {cfg.api_key_env}")
    llm = make_llm(cfg)
    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp) / "home"
        store = ArtifactStore(home)
        runner = Runner(store, Settings(home=home))
        wav = Path(tmp) / "demo.wav"
        audio.write_wav(wav, (0.3 * np.sin(2 * np.pi * 220 * np.arange(5 * 44_100) / 44_100)).astype(np.float32))
        track = runner.ingest(wav, label="demo song.wav").id
        agent = Agent(store, runner, Recording(llm))
        results = [scenario("plan", agent, track, "Отдели вокал от музыки"),
                   scenario("history", agent, track, "А теперь узнай темп и тональность")]
        results.append(scenario("retry", Agent(store, runner, Recording(llm, sabotage=True)), track, "Отдели вокал от музыки"))
    report = {"llm": llm.name, "thinking": cfg.thinking or "provider default", "reasoning_effort": cfg.reasoning_effort or "provider default",
              "ok": all(r["ok"] for r in results), "scenarios": results}
    print(json.dumps(report, ensure_ascii=False, indent=1))
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
