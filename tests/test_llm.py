"""Both providers against fake SDK clients: token usage, latency and the DeepSeek thinking knobs, no network."""

from types import SimpleNamespace

import pytest

from ah.config import AgentSettings, load_settings
from ah.llm import AnthropicLLM, OpenAICompatibleLLM

TOOLS = [{"name": "submit_plan", "description": "d", "schema": {"type": "object", "properties": {}}}]


class Recorder:
    def __init__(self, response):
        self.response = response
        self.kwargs = []

    def create(self, **kwargs):
        self.kwargs.append(kwargs)
        return self.response


def openai_llm(cfg):
    from openai.types.chat import ChatCompletion

    llm = OpenAICompatibleLLM(cfg)
    response = ChatCompletion.model_validate({
        "id": "c", "object": "chat.completion", "created": 0, "model": cfg.model,
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None, "reasoning_content": "думаю",
            "tool_calls": [{"id": "t1", "type": "function",
                            "function": {"name": "submit_plan", "arguments": '{"reply": "ok", "steps": []}'}}]}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
                  "prompt_cache_hit_tokens": 64, "prompt_cache_miss_tokens": 36,
                  "completion_tokens_details": {"reasoning_tokens": 12}},
    })
    recorder = Recorder(response)
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=recorder))
    return llm, recorder


def test_openai_compatible_reply_carries_usage_with_deepseek_cache_counts():
    pytest.importorskip("openai")
    llm, recorder = openai_llm(AgentSettings(provider="openai", model="deepseek-flash", base_url="https://api.deepseek.com"))
    reply = llm.complete("sys", [{"role": "user", "content": "hi"}], TOOLS)
    assert reply.calls[0].input == {"reply": "ok", "steps": []}
    assert reply.usage == {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
                           "prompt_cache_hit_tokens": 64, "prompt_cache_miss_tokens": 36, "reasoning_tokens": 12}
    assert reply.latency_s is not None and reply.latency_s >= 0
    assert "extra_body" not in recorder.kwargs[0]  # nothing DeepSeek-specific unless configured


def test_thinking_and_effort_go_to_the_request_body():
    pytest.importorskip("openai")
    cfg = AgentSettings(provider="openai", model="deepseek-flash", thinking="disabled", reasoning_effort="high")
    llm, recorder = openai_llm(cfg)
    llm.complete("sys", [{"role": "user", "content": "hi"}], TOOLS)
    assert recorder.kwargs[0]["extra_body"] == {"thinking": {"type": "disabled"}, "reasoning_effort": "high"}


def test_anthropic_reply_carries_usage_and_ignores_thinking():
    pytest.importorskip("anthropic")
    from anthropic.types import Message

    cfg = AgentSettings(model="claude-sonnet-5-5", base_url="http://localhost:1", thinking="disabled", reasoning_effort="max")
    llm = AnthropicLLM(cfg)
    recorder = Recorder(Message.model_validate({
        "id": "m", "type": "message", "role": "assistant", "model": cfg.model, "stop_reason": "tool_use",
        "content": [{"type": "text", "text": "план"},
                    {"type": "tool_use", "id": "t1", "name": "submit_plan", "input": {"reply": "ok", "steps": []}}],
        "usage": {"input_tokens": 30, "output_tokens": 7, "cache_read_input_tokens": 50, "cache_creation_input_tokens": 20},
    }))
    llm.client = SimpleNamespace(messages=recorder)
    reply = llm.complete("sys", [{"role": "user", "content": "hi"}], TOOLS)
    assert (reply.text, reply.calls[0].name) == ("план", "submit_plan")
    assert reply.usage["prompt_tokens"] == 100 and reply.usage["completion_tokens"] == 7
    assert reply.usage["cache_read_input_tokens"] == 50
    assert reply.latency_s is not None
    assert not {"thinking", "reasoning_effort", "extra_body"} & set(recorder.kwargs[0])


def test_thinking_settings_from_toml_and_env(tmp_path, monkeypatch):
    config = tmp_path / "ah.toml"
    config.write_text('[agent]\nprovider = "openai"\nmodel = "deepseek-flash"\nthinking = "enabled"\nreasoning_effort = "low"\n')
    monkeypatch.setenv("AH_CONFIG", str(config))
    cfg = load_settings().agent
    assert (cfg.thinking, cfg.reasoning_effort) == ("enabled", "low")
    monkeypatch.setenv("AH_AGENT_THINKING", "disabled")
    assert load_settings().agent.thinking == "disabled"
    monkeypatch.setenv("AH_AGENT_THINKING", "on")
    with pytest.raises(ValueError, match="agent.thinking"):
        load_settings()
