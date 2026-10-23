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


def test_gtzan_rhythm_layout_and_empty_downbeats(tmp_path):
    annotations = tmp_path / "labels"
    jams = annotations / "jams"
    jams.mkdir(parents=True)
    audio = tmp_path / "audio" / "blues"
    audio.mkdir(parents=True)
    (audio / "blues.00000.wav").touch()
    (annotations / "stats.csv").write_text("filename,tempo mean\nblues.00000.wav,123.4\n")
    doc = {"file_metadata": {"duration": 2, "artist": "Artist A"}, "annotations": [
        {"namespace": "beat", "sandbox": {"annotation_type": "beat"},
         "data": [{"time": t} for t in (0, 0.5, 1, 1.5)]},
        {"namespace": "beat", "sandbox": {"annotation_type": "downbeat"}, "data": []},
        {"namespace": "beat", "sandbox": {"annotation_type": "8th-note"},
         "data": [{"time": 0.25}]}]}
    (jams / "blues.00000.wav.jams").write_text(json.dumps(doc))
    rows = bench.gtzan_rows(annotations, audio.parent)
    assert len(rows) == 1
    assert rows[0]["audio"] == str((audio / "blues.00000.wav").resolve())
    assert rows[0]["reference_bpm"] == pytest.approx(123.4)
    assert rows[0]["reference_downbeats"] == []
    assert rows[0]["genre"] == "blues"
    assert rows[0]["artist"] == "Artist A"


def test_tempo_and_event_metrics():
    assert bench.tempo_metrics(120, 120)["acc1"] is True
    assert bench.tempo_metrics(120, 60)["acc1"] is False
    assert bench.tempo_metrics(120, 60)["acc2"] is True
    assert bench.tempo_metrics(120, None)["acc2"] is False
    assert bench.event_f1([0, 0.5], [0.02, 0.49]) == pytest.approx(1)
    assert bench.event_f1([0, 0.5], [0.02]) == pytest.approx(2 / 3)
    assert bench.event_f1([0, 0.1], [0.06, 0.12]) == pytest.approx(1)
    assert bench.event_f1([], [0]) is None


def test_summary_coverage_and_genre_macro_ci():
    rows = [
        {"status": "ok", "id": "a", "genre": "blues", "artist": "A",
         "acc1": True, "acc2": True, "beat_f1_70ms": 1, "downbeat_f1_70ms": None},
        {"status": "ok", "id": "b", "genre": "blues", "artist": "A",
         "acc1": True, "acc2": True, "beat_f1_70ms": 1, "downbeat_f1_70ms": 0.5},
        {"status": "ok", "id": "c", "genre": "jazz", "artist": "B",
         "acc1": False, "acc2": False, "beat_f1_70ms": 0, "downbeat_f1_70ms": 0},
        {"status": "failed", "id": "d", "genre": "jazz", "artist": "C"},
    ]
    summary = bench.summarize(rows)
    assert summary["total"] == 4 and summary["scored"] == 3 and summary["failed"] == 1
    assert summary["metric_n"]["beat_f1_70ms"] == 3
    assert summary["metric_n"]["downbeat_f1_70ms"] == 2
    macro = bench.genre_macro_ci(rows, draws=100)
    assert macro["acc1"]["mean"] == pytest.approx(0.5)
    assert macro["acc1"]["tracks"] == 3
    assert macro["acc1"]["ci95"] == [0.5, 0.5]


def test_run_and_resume_do_not_recompute_completed_audio(tmp_path, monkeypatch):
    calls = []

    class FakeTool:
        def load(self, device):
            assert device == "cpu"

        def run(self, inputs, params, out_dir):
            calls.append(inputs["audio"])
            (out_dir / "analysis.json").write_text(json.dumps({
                "bpm": 120, "beats": [0, 0.5, 1, 1.5], "downbeats": [0]}))
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

    bench.write_jsonl(manifest, [{"dataset": "gtzan", "id": "track", "audio": str(audio),
                                 "split": "all", "reference_bpm": 120,
                                 "reference_beats": [0, 0.5, 1],
                                 "reference_downbeats": [], "genre": "blues", "artist": "A"}])
    gtzan = bench.evaluate(manifest, tmp_path / "gtzan_results", "beat_this", "cpu", "all", None)
    assert gtzan["metrics"]["beat_f1_70ms"] == pytest.approx(6 / 7)
    assert gtzan["metrics"]["metric_n"]["downbeat_f1_70ms"] == 0
    assert "genre_macro_ci95" in gtzan


def test_cli_coverage_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "evaluate", lambda *args: {"metrics": {"total": 2, "scored": 1}})
    prefix = ["run", "--manifest", str(tmp_path / "manifest.jsonl"),
              "--output", str(tmp_path / "out")]
    assert bench.main(prefix + ["--expect-total", "2", "--expect-scored", "1"]) == 0
    assert bench.main(prefix + ["--expect-total", "3"]) == 2
    assert bench.main(prefix + ["--expect-scored", "2"]) == 2
