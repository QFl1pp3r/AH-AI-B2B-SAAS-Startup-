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


def _rows(values_by_player):
    return [{"player": p, "style": "rock", "prog": str(i % 3), "bpm": str(100 + i // 3), "kind": "comp", "x": v}
            for p, values in values_by_player.items() for i, v in enumerate(values)]


def test_bootstrap_ci_brackets_the_mean_and_is_reproducible():
    rows = _rows({f"0{p}": [float((p + i) % 2) for i in range(6)] for p in range(6)})
    ci = _module.bootstrap_ci(rows, "x")
    assert ci[0] <= _module.mean(rows, "x") <= ci[1]
    assert ci == _module.bootstrap_ci(rows, "x")


def test_bootstrap_ci_widens_when_players_differ():
    # Same overall mean 0.5. In the checkerboard every player and every sheet averages 0.5;
    # in the clustered set each player is all-0 or all-1, so rows are not independent.
    mixed = _rows({f"0{p}": [float((p + i) % 2) for i in range(6)] for p in range(6)})
    clustered = _rows({f"0{p}": [float(p % 2)] * 6 for p in range(6)})
    (lo_c, hi_c), (lo_m, hi_m) = (_module.bootstrap_ci(rows, "x") for rows in (clustered, mixed))
    assert hi_c - lo_c > 1.5 * (hi_m - lo_m)


def test_load_tracks_roundtrip(tmp_path):
    path = tmp_path / "tracks.csv"
    path.write_text("track,player,bpm,bpm_est,beat_f\n00_Rock1-130-A_comp,00,130,,0.5\n")
    row = _module.load_tracks(path)[0]
    assert (row["player"], row["bpm"], row["bpm_est"], row["beat_f"]) == ("00", "130", None, 0.5)
