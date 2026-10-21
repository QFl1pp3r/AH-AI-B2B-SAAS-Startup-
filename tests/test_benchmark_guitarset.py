"""Scoring helpers of benchmarks/guitarset_analyze.py that need neither mir_eval nor the analyze env."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "guitarset_analyze_under_test", Path(__file__).parents[1] / "benchmarks/guitarset_analyze.py")
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)


@pytest.mark.parametrize("estimate, acc1, acc2, error", [
    (120.0, 1.0, 1.0, "ok"),
    (124.0, 1.0, 1.0, "ok"),      # inside the 4% window
    (126.0, 0.0, 0.0, "other"),
    (60.0, 0.0, 1.0, "half"),
    (240.0, 0.0, 1.0, "double"),
    (180.0, 0.0, 0.0, "ternary"),  # 3/2, the swing-feel error; not an Acc2 hit
    (None, 0.0, 0.0, "other"),
])
def test_tempo_scores(estimate, acc1, acc2, error):
    assert _module.tempo_scores(estimate, 120.0) == {"tempo_acc1": acc1, "tempo_acc2": acc2}
    assert _module.tempo_error({"bpm_est": estimate, "bpm_ref": 120.0}) == error


def test_load_reference_reads_both_chord_annotations(tmp_path):
    chord = lambda source, value: {"namespace": "chord", "annotation_metadata": {"data_source": source},
                                   "data": [{"time": 0.0, "duration": 2.0, "value": value}]}
    doc = {"file_metadata": {"duration": 2.0}, "annotations": [
        {"namespace": "tempo", "annotation_metadata": {}, "data": [{"value": 90.0}]},
        {"namespace": "key_mode", "annotation_metadata": {}, "data": [{"value": "Eb:major"}]},
        {"namespace": "beat_position", "annotation_metadata": {}, "data": [
            {"time": t, "value": {"position": p}} for t, p in ((0.0, 1), (0.66, 2), (1.33, 1))]},
        chord("", "D#:maj"),
        chord("Semi-automatic chord transcription with manual verification", "D#:maj7/1"),
    ]}
    path = tmp_path / "x.jams"
    path.write_text(json.dumps(doc))
    ref = _module.load_reference(path)
    assert ref["bpm"] == 90.0 and ref["key"] == "Eb major"
    assert list(ref["downbeats"]) == [0.0, 1.33]
    assert ref["instructed"][1] == ["D#:maj"] and ref["performed"][1] == ["D#:maj7/1"]
