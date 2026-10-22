"""Dataset adapters and metrics, using tiny stand-ins for the official layouts."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "benchmarks" / "tempo.py"
spec = importlib.util.spec_from_file_location("tempo_benchmark", SCRIPT)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_ballroom_annotation_and_audio_match(tmp_path):
    labels = tmp_path / "labels"
    audio = tmp_path / "audio" / "Waltz"
    labels.mkdir()
    audio.mkdir(parents=True)
    (labels / "example.beats").write_text("0 1\n0.5 2\n1 3\n1.5 1\n2 2\n")
    (audio / "example.wav").touch()
    rows = bench.ballroom_rows(labels, audio.parent)
    assert len(rows) == 1
    assert rows[0]["reference_bpm"] == pytest.approx(120)
    assert rows[0]["reference_downbeats"] == [0, 1.5]
    assert rows[0]["audio"] == str((audio / "example.wav").resolve())


def test_giantsteps_v2_folds_and_zero_label(tmp_path):
    labels = tmp_path / "annotations_v2" / "tempo"
    folds = tmp_path / "splits"
    labels.mkdir(parents=True)
    folds.mkdir()
    (labels / "one.LOFI.bpm").write_text("128.0\n")
    (labels / "two.LOFI.bpm").write_text("0.0\n")
    for number in range(1, 11):
        (folds / f"fold{number:02}.txt").write_text(
            "one.LOFI.mp3\ntwo.LOFI.mp3\n" if number == 1 else "")
    rows = bench.giantsteps_rows(tmp_path, tmp_path / "audio")
    assert len(rows) == 1
    assert rows[0]["id"] == "one.LOFI"
    assert rows[0]["split"] == "test"


def test_tempo_and_event_metrics():
    assert bench.tempo_metrics(120, 120)["acc1"] is True
    assert bench.tempo_metrics(120, 60)["acc1"] is False
    assert bench.tempo_metrics(120, 60)["acc2"] is True
    assert bench.tempo_metrics(120, None)["acc2"] is False
    assert bench.event_f1([0, 0.5], [0.02, 0.49]) == pytest.approx(1)
    assert bench.event_f1([0, 0.5], [0.02]) == pytest.approx(2 / 3)


def test_run_and_resume_do_not_recompute_completed_audio(tmp_path, monkeypatch):
    calls = []

    class FakeTool:
        def load(self, device):
            assert device == "cpu"

        def run(self, inputs, params, out_dir):
            calls.append(inputs["audio"])
            (out_dir / "analysis.json").write_text(json.dumps({
                "bpm": 120, "beats": [0, 0.5, 1], "downbeats": [0]}))
            return types.SimpleNamespace(warnings=[])

    monkeypatch.setitem(sys.modules, "ah_analyze", types.SimpleNamespace(AnalyzeTool=FakeTool))
    audio = tmp_path / "track.wav"
    audio.touch()
    manifest = tmp_path / "manifest.jsonl"
    bench.write_jsonl(manifest, [{"dataset": "ballroom", "id": "track", "audio": str(audio),
                                 "split": "all", "reference_bpm": 120,
                                 "reference_beats": [0, 0.5, 1],
                                 "reference_downbeats": [0]}])
    output = tmp_path / "results"
    first = bench.evaluate(manifest, output, "beat_this", "cpu", "all", None)
    second = bench.evaluate(manifest, output, "beat_this", "cpu", "all", None, resume=True)
    assert len(calls) == 1
    assert first["metrics"]["acc1"] == second["metrics"]["acc1"] == 1
    assert first["metrics"]["beat_f1_70ms"] == 1
    assert len(bench.read_jsonl(output / "results.jsonl")) == 1
