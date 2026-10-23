"""transcribe with a fake Whisper: hallucination filters, lyrics.json, and the text never reaching the agent."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from test_agent import ScriptedLLM, plan, store_runner, wait  # noqa: F401  (store_runner is a fixture)
from test_core import tone

from ah import registry
from ah.agent import Agent, PlanError, validate
from ah.llm import Reply
from ah.registry import ToolSpec
from ah.ui import create_app

_spec = importlib.util.spec_from_file_location(
    "ah_transcribe_under_test", Path(__file__).parents[1] / "tools/transcribe/src/ah_transcribe/__init__.py")
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

LINE = "окно открыто настежь, чайник снова стынет"


def seg(start, end, text, no_speech=0.05, logprob=-0.3, ratio=1.4):
    words = [SimpleNamespace(start=start + i * 0.3, end=start + i * 0.3 + 0.25, word=f" {w}", probability=0.9)
             for i, w in enumerate(text.split())]
    return SimpleNamespace(start=start, end=end, text=f" {text}", words=words, no_speech_prob=no_speech,
                           avg_logprob=logprob, compression_ratio=ratio)


SEGMENTS = [
    seg(0.4, 2.9, LINE),
    seg(3.0, 5.0, "Субтитры подготовил кто-то"),
    seg(5.0, 7.0, "м", no_speech=0.9, logprob=-1.6),
    seg(7.0, 9.0, "ла ла ла ла ла ла ла ла", ratio=3.1),
    seg(9.2, 11.0, "и снова утро без тебя"),
]


class FakeModel:
    def __init__(self, segments):
        self.segments = segments
        self.calls = []

    def transcribe(self, data, **kwargs):
        self.calls.append((data, kwargs))
        return iter(self.segments), SimpleNamespace(language="ru", language_probability=0.97)


class FakeTranscribe(_module.TranscribeTool):
    """Registered in-process for the agent and UI tests."""

    def _model(self, name):
        return FakeModel(SEGMENTS)


def run_fake(tmp_path, segments, params=None):
    out = tmp_path / "out"
    out.mkdir()
    tool = _module.TranscribeTool()
    model = FakeModel(segments)
    tool._model = lambda name: model
    result = tool.run({"audio": tone(tmp_path / "in.wav", sr=44_100, channels=2)}, params or {}, out)
    return result, json.loads((out / "lyrics.json").read_text()), model


def test_keeps_real_lines_and_drops_hallucinations(tmp_path):
    result, lyrics, model = run_fake(tmp_path, SEGMENTS, {"language": "ru"})
    assert [s["text"] for s in lyrics["segments"]] == [LINE, "и снова утро без тебя"]
    assert lyrics["segments"][0]["words"][0] == {"start": 0.4, "end": 0.65, "word": "окно", "p": 0.9}
    assert result.meta["dropped_segments"] == 3 and result.meta["lines"] == 2
    assert any("hallucination" in w for w in result.warnings)
    data, kwargs = model.calls[0]
    assert data.ndim == 1 and len(data) == 16_000  # 1 s of stereo 44.1 kHz -> mono 16 kHz
    assert kwargs["language"] == "ru" and kwargs["word_timestamps"] and not kwargs["condition_on_previous_text"]


def test_text_stays_out_of_meta(tmp_path):
    result, _, _ = run_fake(tmp_path, SEGMENTS)
    for meta in (result.meta, result.outputs[0].meta):
        assert all(not isinstance(v, (list, dict)) for v in meta.values())
        assert "окно" not in json.dumps(meta, ensure_ascii=False)


def test_nothing_recognized_is_a_warning_not_an_error(tmp_path):
    result, lyrics, _ = run_fake(tmp_path, [])
    assert lyrics["segments"] == [] and result.error is None
    assert "no singing or speech recognized" in result.warnings


def test_params_are_strict():
    with pytest.raises(ValueError):
        _module.Params.model_validate({"model": "tiny"})
    with pytest.raises(ValueError):
        _module.Params.model_validate({"prompt": "x"})


def test_lyrics_cannot_feed_another_step():
    with pytest.raises(PlanError, match="no audio output"):
        validate({"reply": "x", "steps": [{"tool": "transcribe", "input": "track", "why": "1"},
                                          {"tool": "analyze", "input": "1.lyrics", "why": "2"}]})


@pytest.fixture
def fake_transcribe(monkeypatch):
    spec = registry.TOOLS["transcribe"]
    monkeypatch.setitem(registry.TOOLS, "transcribe", ToolSpec(
        "transcribe", "test_transcribe", "FakeTranscribe", None, spec.inputs, spec.summary,
        outputs=spec.outputs, hint=spec.hint, params=spec.params))


def test_agent_never_sees_the_text(store_runner, fake_transcribe):  # noqa: F811
    store, runner, original = store_runner
    llm = ScriptedLLM(plan("Распознаю текст.", {"tool": "transcribe", "input": "track", "why": "текст"}),
                      Reply(text="Текст в панели."), plan("Текст уже есть, он в панели."))
    agent = Agent(store, runner, llm)
    done = wait(agent.start(agent.message(original.id, "что тут поют?").id))
    assert done.status == "done" and done.steps[0].outputs[0]["role"] == "lyrics"
    agent.message(original.id, "а теперь?")
    sent = json.dumps(llm.sent, ensure_ascii=False)
    assert "окно" not in sent and "утро" not in sent
    assert '"lines": 2' in llm.sent[1]["messages"][0]["content"]


def test_ui_finds_lyrics_for_the_audio_version(store_runner, fake_transcribe):  # noqa: F811
    store, runner, original = store_runner
    client = TestClient(create_app(store, runner, llm=ScriptedLLM()))
    assert client.get("/api/state").json()["lyrics"] is None
    lyrics = runner.run("transcribe", {"audio": original.id}).artifacts[0]
    for selected in (original.id, lyrics.id):
        assert client.get(f"/api/state?selected={selected}").json()["lyrics"] == lyrics.id
    content = client.get(f"/api/artifacts/{lyrics.id}").json()["content"]
    assert content["segments"][0]["text"] == LINE
