"""The agent benchmark without an LLM: gold plans pass the agent's own validate, fixtures match the store, scoring rules."""

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ah.agent import validate
from ah.registry import TOOLS

ROOT = Path(__file__).parents[1]
AGENT = ROOT / "benchmarks/agent"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"agent_bench_{name}", AGENT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


score = _load("score")
fixtures = _load("fixtures")
TASKS = [json.loads(line) for path in sorted((AGENT / "tasks").glob("*.jsonl")) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("task", TASKS, ids=[t["id"] for t in TASKS])
def test_gold_plans_are_plans_the_agent_would_accept(task):
    gold = task["gold"]
    assert (AGENT / task["track_data"]).is_file()
    assert set(gold.get("forbid", [])) <= set(TOOLS)
    for tools in gold["accept"] or []:
        def wired(tool):
            """The first allowed input of this tool that names an output an earlier step of this plan really has."""
            options = gold.get("tool_inputs", {}).get(tool, "track")
            for ref in [options] if isinstance(options, str) else options:
                n, _, role = ref.partition(".")
                if ref == "track" or (n.isdigit() and int(n) <= len(tools) and role in TOOLS[tools[int(n) - 1]].outputs):
                    return ref
            return options[0]

        steps = [{"tool": tool, "input": gold.get("inputs", {}).get(str(n), wired(tool)), "why": "x",
                  "params": gold.get("params", {}).get(str(n), gold.get("tool_params", {}).get(tool, {}))}
                 for n, tool in enumerate(tools, 1)]
        validate({"reply": "ok", "steps": steps})
        assert not set(tools) & set(gold.get("forbid", []))


def test_task_ids_unique_and_both_languages():
    assert len({t["id"] for t in TASKS}) == len(TASKS)
    assert {"ru", "en"} <= {t["lang"] for t in TASKS}
    for t in TASKS:
        assert (AGENT / t["track_data"]).is_file()
        for turn in t.get("history", []):
            assert set(turn) == {"request", "reply", "summary"}


def test_committed_fixtures_are_what_the_store_gives_today(tmp_path):
    def strip(data):
        return json.loads(json.dumps(data).replace(data["selected_version"]["id"], "SEL")) | {"versions": [
            {k: v for k, v in version.items() if k != "id"} for version in data["versions"]]}
    built = fixtures.build(tmp_path / "home")
    for name, data in built.items():
        committed = json.loads((AGENT / f"fixtures/{name}.json").read_text())
        assert {k: v for k, v in strip(data).items() if k != "analysis"} == \
               {k: v for k, v in strip(committed).items() if k != "analysis"}
        assert (data["analysis"] is None) == (committed["analysis"] is None)


def run(steps, reply="ok", ok=True, calls=1, attempt=0):
    usage = {"prompt_tokens": 1000, "completion_tokens": 100, "prompt_cache_hit_tokens": 800, "prompt_cache_miss_tokens": 200}
    at = datetime(2026, 10, 21, 7, tzinfo=timezone.utc).timestamp()  # Wednesday 07:00 UTC, peak
    return {"task": "t", "attempt": attempt, "ok": ok, "error": None if ok else "Модель дважды вернула негодный план",
            "reply": reply if ok else None, "steps": steps if ok else None,
            "llm_calls": [{"at": at, "usage": usage, "latency_s": 1.5}] * calls}


def task(**gold):
    return {"id": "t", "category": "c", "lang": "en", "gold": gold, "_model": "deepseek-flash"}


def step(tool, source="track", **params):
    return {"tool": tool, "input": source, "params": params}


def test_check_scores_sequence_inputs_and_params():
    gold = task(accept=[["separate", "denoise"], ["separate", "restore"]], inputs={"2": "1.vocals"},
                params={"1": {}}, forbid=["analyze"])
    good = score.check(run([step("separate"), step("denoise", "1.vocals")], "I'll split it and clean the vocals"), gold)
    assert good["correct"] == 1.0 and good["precision"] == good["recall"] == 1.0
    wired_wrong = score.check(run([step("separate"), step("denoise", "track")]), gold)
    assert wired_wrong["inputs_ok"] == 0.0 and wired_wrong["correct"] == 0.0 and wired_wrong["failed"] == "inputs_ok"
    extra = score.check(run([step("separate"), step("denoise", "1.vocals"), step("analyze", "1.instrumental")]), gold)
    assert extra["forbidden"] == "analyze" and extra["precision"] == pytest.approx(2 / 3) and extra["recall"] == 1.0
    retried = score.check(run([step("separate"), step("restore", "1.vocals")], calls=2), gold)
    assert retried["valid_first_try"] == 0.0 and retried["valid"] == 1.0 and retried["correct"] == 1.0
    failed = score.check(run(None, ok=False, calls=2), gold)
    assert failed["valid"] == 0.0 and failed["correct"] == 0.0 and failed["failed"] == "valid"


def test_tool_inputs_accept_any_order():
    gold = task(accept=[["separate", "denoise", "analyze"], ["separate", "analyze", "denoise"]],
                tool_inputs={"denoise": "1.vocals", "analyze": ["1.instrumental", "1.vocals"]})
    order = [step("separate"), step("analyze", "1.instrumental"), step("denoise", "1.vocals")]
    assert score.check(run(order), gold)["correct"] == 1.0
    assert score.check(run([step("separate"), step("analyze", "1.instrumental"), step("denoise", "track")]), gold)["inputs_ok"] == 0.0


def test_tool_params_apply_wherever_the_tool_stands():
    gold = task(accept=[["denoise", "trim"], ["trim", "denoise"]], tool_params={"trim": {"silence": True}})
    assert score.check(run([step("denoise"), step("trim", "1.denoised", silence=True)]), gold)["params_ok"] == 1.0
    assert score.check(run([step("trim", start_s=3), step("denoise", "1.trimmed")]), gold)["params_ok"] == 0.0


def test_trim_params_within_tolerance():
    gold = task(accept=[["trim"]], params={"1": {"start_s": 60, "end_s": 150}})
    assert score.check(run([step("trim", start_s=60.05, end_s=150)]), gold)["params_ok"] == 1.0
    assert score.check(run([step("trim", start_s=60, end_s=90)]), gold)["params_ok"] == 0.0
    assert score.check(run([step("trim", start_s=60)]), gold)["params_ok"] == 0.0
    assert score.check(run([step("trim", silence=True)]), task(accept=[["trim"]], params={"1": {"silence": True}}))["params_ok"] == 1.0


def test_unsupported_answer_and_ambiguous_rules():
    unsupported = task(accept=[[]], must_say_unsupported=True)
    assert score.check(run([], "MIDI баса пока нельзя сделать"), unsupported)["correct"] == 1.0
    assert score.check(run([], "Sure, done!"), unsupported)["unsupported_ok"] == 0.0
    answer = task(accept=[[]], reply_any=["124"])
    assert score.check(run([], "It is 124 BPM"), answer)["correct"] == 1.0
    assert score.check(run([step("analyze")], "Analysing"), answer)["failed"] == "seq_ok,reply_ok"
    vague = task(accept=None, max_steps=2)
    assert score.check(run([step("inspect"), step("denoise")]), vague)["correct"] == 1.0
    assert score.check(run([step("trim"), step("separate"), step("denoise", "2.vocals")]), vague)["seq_ok"] == 0.0


@pytest.mark.parametrize("reply, lang, ok", [
    ("Разделю трек на вокал и инструментал.", "ru", True),
    ("I'll split the mix into vocals and instrumental.", "en", True),
    ("124 BPM in 4/4, already in the analysis, no need to run anything.", "en", True),
    ("Taglio i primi 30 secondi e tengo il resto.", "en", False),
    ("Il tempo è 124 BPM, in 4/4 (l'analisi precedente dà anche A minore).", "en", False),
    ("Ich trenne den Track in Gesang und Instrumental.", "en", False),
    ("Splitting the track", "ru", False),
    ("Ich trenne den Track in Gesang und Instrumental.", "de", True),
    ("I'll split the track into vocals and instrumental.", "de", False),
    ("Separo la pista en voz e instrumental para que tengas el acapella.", "es", True),
    ("Taglio i primi 30 secondi e tengo il resto della traccia.", "it", True),
    ("Taglio i primi 30 secondi e tengo il resto della traccia.", "es", False),
    ("Just a trim: leading and trailing silence goes, nothing in between.", "en", True),
    ("Peak level is -0.3 dBFS, right up near full scale, but nothing clipped.", "en", True),
    ("Cortaré el silencio del principio y del final.", "en", False),
    ("One trim, cutting everything before 0:12.", "en", True),
    ("Tempo: 124 BPM, 4/4. Key: A minor (chords Am–F–C–G).", "en", True),
    ("Igen, 4/4-ben van. A korábbi elemzés szerint 124 BPM.", "en", False),
    ("Drums naar MIDI kan ik nog niet: geen van de tools doet transcriptie.", "en", False),
    ("Drums to MIDI non è ancora possibile: non ho un工具 di trascrizione.", "en", False),
])
def test_reply_language(reply, lang, ok):
    assert score.reply_lang_ok(reply, lang) is ok


def test_cost_peak_and_off_peak():
    call = run([])["llm_calls"][0]
    peak = (800 * 0.006 + 200 * 0.30 + 100 * 1.20) / 1e6
    assert score.call_cost(call, "deepseek-flash") == pytest.approx(peak)
    night = call | {"at": datetime(2026, 10, 21, 12, tzinfo=timezone.utc).timestamp()}
    weekend = call | {"at": datetime(2026, 10, 24, 7, tzinfo=timezone.utc).timestamp()}
    assert score.call_cost(night, "deepseek-flash") == pytest.approx(peak / 2)
    assert score.call_cost(weekend, "deepseek-flash") == pytest.approx(peak / 2)
    assert score.call_cost(call, "some-other-model") is None


def test_summary_pass_k_and_from_runs(tmp_path):
    gold = task(accept=[["separate"]])
    rows = [score.check(run([step("separate")], attempt=k), gold) for k in range(4)]
    rows.append(score.check(run([step("analyze")], attempt=4), gold))
    other = [r | {"task": "u"} for r in (score.check(run([step("separate")], attempt=k), gold) for k in range(5))]
    summary = score.aggregate(rows + other, {"x": 1}, tmp_path)
    assert summary["groups"]["all"]["correct"]["mean"] == 0.9
    assert summary["groups"]["all"]["pass_k"]["mean"] == 0.5
    assert summary["wrong_examples"]["c"][0]["tools"] == "analyze"
    score.results.write_tracks(tmp_path / "runs.csv", rows + other)
    rebuilt = score.aggregate(score.results.read_tracks(tmp_path / "runs.csv", text=score.TEXT), {"x": 1}, tmp_path / "b")
    assert rebuilt["groups"] == summary["groups"]


def test_compare_pairs_tasks(tmp_path):
    compare = _load("compare")
    gold = task(accept=[["separate"]])
    for name, tools in (("before", ["analyze", "separate"]), ("after", ["separate", "separate"])):
        rows = [score.check(run([step(tool)], attempt=k), gold) | {"task": f"t{n}"}
                for n in range(6) for k, tool in enumerate(tools)]
        score.results.write_tracks(tmp_path / name / "runs.csv", rows)
    out = compare.compare(tmp_path / "before", tmp_path / "after")
    assert out["tasks"] == 6
    assert out["groups"]["all"]["correct"]["diff"] == 0.5 and out["groups"]["all"]["correct"]["significant"]


def test_resume_redoes_only_attempts_the_provider_refused():
    run_module = _load("run")
    refused = {"llm_calls": [{"at": 0, "error": "Провайдер вернул 402: Insufficient Balance"}]}
    broken_json = {"llm_calls": [{"at": 0, "error": "Модель вернула вызов инструмента с битым JSON"}]}
    twice_invalid = {"llm_calls": [{"at": 0, "usage": {}}, {"at": 1, "usage": {}}]}
    assert run_module.provider_failed(refused)
    assert not run_module.provider_failed(broken_json) and not run_module.provider_failed(twice_invalid)


def test_task_history_reaches_the_agent_as_chat_turns(tmp_path):
    from test_core import tone

    from ah.agent import Agent
    from ah.config import Settings
    from ah.runner import Runner
    from ah.store import ArtifactStore

    run_module = _load("run")
    store = ArtifactStore(tmp_path / "home")
    runner = Runner(store, Settings(home=tmp_path / "home"))
    track = runner.ingest(tone(tmp_path / "in.wav")).id
    history = [{"request": "Отдели вокал", "reply": "Разделю трек.", "summary": "Готово: vocals и instrumental."}]
    agent = run_module.with_history(Agent(store, runner, None), track, {"history": history})
    assert agent._history(store.root(track).id) == [
        {"role": "user", "content": "Отдели вокал"},
        {"role": "assistant", "content": "Разделю трек.\n\nResult: Готово: vocals и instrumental."}]


def test_execute_runs_a_plan_through_the_runner_and_checks_artifacts(tmp_path):
    from test_core import tone

    from ah.config import Settings
    from common.runner import CachedRunner

    ex = _load("execute")
    clip = tone(tmp_path / "clip.wav", seconds=4.0, channels=2)
    assert ex.scaled({"params": {"start_s": 100, "end_s": 200, "silence": True}}, 0.01) == {"start_s": 1.0, "end_s": 2.0, "silence": True}
    with CachedRunner(tmp_path / "cache", Settings(home=tmp_path / "unused")) as runner:
        plan = [{"tool": "trim", "input": "track", "params": {"start_s": 100, "end_s": 300}},
                {"tool": "inspect", "input": "1.trimmed", "params": {}}]
        done = ex.run_plan(runner, plan, clip, scale=0.01)
        assert done["ok"] and [s["tool"] for s in done["steps"]] == ["trim", "inspect"]
        assert done["steps"][0]["params"] == {"start_s": 1.0, "end_s": 3.0}
        past_the_end = ex.run_plan(runner, [{"tool": "trim", "input": "track", "params": {"start_s": 500}}], clip, scale=0.01)
        assert not past_the_end["ok"] and past_the_end["steps"][0]["error"]
        out = runner.run("trim", clip, {"start_s": 1.0})
        assert ex.check_outputs("trim", {"start_s": 1.0}, clip, out.files) is None
        assert "asked" in ex.check_outputs("trim", {"start_s": 2.0}, clip, out.files)  # 3 s left, 2 s asked
        assert ex.check_outputs("separate", {}, clip, {"vocals": out.files["trimmed"]}) == "no instrumental output"


def test_execute_success_counts_only_plans_with_steps():
    ex = _load("execute")
    rows = [{"task": "a", "has_steps": True, "exec_ok": 1.0, "e2e_ok": 1.0},
            {"task": "b", "has_steps": True, "exec_ok": 0.0, "e2e_ok": 0.0},
            {"task": "c", "has_steps": False, "exec_ok": None, "e2e_ok": 1.0}]  # a right answer with no tools
    out = ex.summarize(rows)
    assert out["exec_success"]["mean"] == 0.5 and out["e2e_success"]["mean"] == round(2 / 3, 4)
