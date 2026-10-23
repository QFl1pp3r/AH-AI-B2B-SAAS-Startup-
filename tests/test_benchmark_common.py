"""benchmarks/common on synthetic data: bootstrap CIs, manifests, cached tool runs, result files. No network, no weights."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from test_core import tone

from ah import registry
from ah.config import Settings
from ah.registry import ToolSpec
from common import manifest, results, speech, stats
from common.runner import CachedRunner, cache_key

ROOT = Path(__file__).parents[1]


# --- stats -------------------------------------------------------------------

def test_constant_values_give_a_point_ci_and_nan_rows_drop_out():
    assert stats.bootstrap_ci([0.7] * 20) == [0.7, 0.7]
    assert stats.mean_ci([1.0, float("nan"), 3.0]) == {"mean": 2.0, "ci95": stats.bootstrap_ci([1.0, 3.0]), "n": 2}
    assert stats.bootstrap_ci([float("nan")]) is None


def test_iid_ci_covers_the_mean_and_is_reproducible():
    values = np.random.default_rng(1).normal(0.5, 0.1, 400)
    lo, hi = stats.bootstrap_ci(values)
    assert lo < values.mean() < hi and hi - lo < 0.03
    assert stats.bootstrap_ci(values) == [lo, hi]  # seed 0 by default
    assert stats.bootstrap_ci(values, seed=1) != [lo, hi]


def test_cluster_ci_is_wider_when_rows_share_a_group():
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(8), 25)
    values = rng.normal(0, 1, 8)[groups] + rng.normal(0, 0.1, 200)  # almost all variance is between groups
    iid = stats.bootstrap_ci(values)
    clustered = stats.bootstrap_ci(values, groups=[groups])
    assert clustered[1] - clustered[0] > 3 * (iid[1] - iid[0])


def test_weights_move_the_mean():
    out = stats.mean_ci([0.0, 1.0], weights=[1.0, 3.0])
    assert out["mean"] == 0.75


def test_paired_diff_is_narrow_when_conditions_move_together():
    rng = np.random.default_rng(3)
    base = rng.uniform(0.2, 0.9, 100)  # tracks differ a lot, the condition adds a steady +0.05
    a, b = base + 0.05 + rng.normal(0, 0.01, 100), base
    paired = stats.paired_diff(a, b)
    assert paired["significant"] and paired["ci95"][0] > 0.04 and paired["ci95"][1] < 0.06
    lo_a, hi_a = stats.bootstrap_ci(a)
    lo_b, hi_b = stats.bootstrap_ci(b)
    assert hi_b > lo_a  # two separate CIs overlap, the paired one does not cross 0
    same = stats.paired_diff(base, base)
    assert same["diff"] == 0.0 and not same["significant"]


def test_two_way_bootstrap_matches_the_original_guitarset_code():
    rng = np.random.default_rng(4)
    players = rng.integers(0, 6, 120).astype(str)
    sheets = rng.integers(0, 30, 120).astype(str)
    values, base = rng.uniform(0, 1, 120), rng.uniform(1, 10, 120)
    # The pre-refactor body of guitarset_analyze.bootstrap_ci, kept here as the reference.
    _, p_idx = np.unique(players, return_inverse=True)
    _, s_idx = np.unique(sheets, return_inverse=True)
    old = np.random.default_rng(0)
    p_count = old.multinomial(6, np.full(6, 1 / 6), size=2000)
    s_count = old.multinomial(len(set(sheets)), np.full(len(set(sheets)), 1 / len(set(sheets))), size=2000)
    w = p_count[:, p_idx] * s_count[:, s_idx] * base
    total = w.sum(axis=1)
    expected = [round(float(q), 4) for q in np.percentile((w @ values)[total > 0] / total[total > 0], [2.5, 97.5])]
    assert stats.bootstrap_ci(values, groups=[players, sheets], weights=base) == expected


def test_guitarset_from_tracks_reproduces_the_committed_summary(tmp_path):
    spec = importlib.util.spec_from_file_location("guitarset_common_check", ROOT / "benchmarks/guitarset_analyze.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    committed = ROOT / "benchmarks/results/guitarset-madmom"
    module.aggregate(module.load_tracks(committed / "tracks.csv"), tmp_path)
    assert (tmp_path / "summary.json").read_text() == (committed / "summary.json").read_text()


# --- manifest ------------------------------------------------------------------

def test_manifest_roundtrip_checks_and_coverage(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"x")
    rows = [{"id": "a", "audio": "a.wav", "split": "test", "bpm": 120.0},
            {"id": "b", "audio": "b.wav", "split": "test"},
            {"id": "c", "audio": "a.wav", "split": "test"}]
    path = manifest.write(tmp_path / "m.jsonl", rows)
    assert manifest.read(path) == rows
    assert manifest.coverage(rows, tmp_path, failed=["c"]) == {
        "total": 3, "with_audio": 2, "missing_audio": ["b"], "failed": ["c"], "scored": 1}
    with pytest.raises(ValueError, match="lacks"):
        manifest.check([{"id": "x", "audio": "x.wav"}])
    with pytest.raises(ValueError, match="duplicate"):
        manifest.check(rows + rows[:1])
    path.write_text('{"id": "a", "audio": "a.wav", "split": "t"}\nnot json\n')
    with pytest.raises(ValueError, match=":2:"):
        manifest.read(path)


# --- cached runner -------------------------------------------------------------

@pytest.fixture
def halfgain(monkeypatch):
    monkeypatch.setitem(registry.TOOLS, "halfgain", ToolSpec("halfgain", "test_core", "HalfGain", None, ("audio",), "x",
                                                             outputs=("half",)))


def test_cached_runner_runs_once_per_input_and_params(tmp_path, halfgain):
    wav = tone(tmp_path / "in.wav")
    with CachedRunner(tmp_path / "cache", Settings(home=tmp_path / "unused")) as runner:
        assert runner.settings.home != tmp_path / "unused"  # throwaway store, never the user's .ah
        first = runner.run("halfgain", wav)
        again = runner.run("halfgain", wav)
    assert not first.cached and again.cached
    assert first.files == again.files and first.files["half"].is_file()
    assert first.input_sha256 == manifest.sha256(wav) and first.seconds >= 0 and first.backend == "inprocess"
    assert not list((tmp_path / "cache").rglob("*.partial"))
    # The output is canonical audio from the store, so a chain can feed it to the next tool.
    data, sr = sf.read(first.files["half"])
    assert sr == 44_100 and data.shape[1] == 2
    assert cache_key(first.input_sha256, "halfgain", {"a": 1}) != cache_key(first.input_sha256, "halfgain", {})
    assert cache_key("s", "t", {"a": 1, "b": 2}) == cache_key("s", "t", {"b": 2, "a": 1})


# --- results -------------------------------------------------------------------

def test_tracks_roundtrip_and_summary_metadata(tmp_path):
    rows = [{"track": "001", "score": 0.5, "note": "ok"}, {"track": "002", "score": None, "note": "x", "extra": 1.0}]
    results.write_tracks(tmp_path / "tracks.csv", rows)
    back = results.read_tracks(tmp_path / "tracks.csv", text={"track"})
    assert back == [{"track": "001", "score": 0.5, "note": "ok", "extra": None},
                    {"track": "002", "score": None, "note": "x", "extra": 1.0}]
    (tmp_path / "m.jsonl").write_text("{}\n")
    meta = results.run_meta(dataset="synthetic v0", manifest=tmp_path / "m.jsonl",
                            tools={"analyze": {"backend": "local", "params": {"key": True}}}, coverage={"total": 2})
    assert len(meta["git_commit"]) == 40 and meta["manifest_sha256"] == manifest.sha256(tmp_path / "m.jsonl")
    assert meta["device"].startswith("cpu")
    summary = json.loads(results.write_summary(tmp_path, {"groups": {"all": {"n": 2}}}, meta).read_text())
    assert summary["meta"]["dataset"] == "synthetic v0" and summary["groups"]["all"]["n"] == 2


# --- speech (pure parts) ----------------------------------------------------------

def test_speech_text_normalization_and_level():
    assert speech.normalize_text("Ёлка, стоит!  Там…") == "елка стоит там"
    loud = (0.5 * np.sin(np.linspace(0, 200, 16_000))).astype(np.float32)
    level = speech.normalize_level(loud)
    assert abs(20 * np.log10(speech.active_rms(level)) - speech.TARGET_DBFS) < 0.01


def test_si_sdr_and_alignment():
    pytest.importorskip("scipy")
    ref = np.random.default_rng(5).normal(0, 0.1, 16_000).astype(np.float32)
    late = np.concatenate([np.zeros(80, np.float32), ref[:-80]])
    aligned, lag = speech.align(late, ref)
    assert lag == 80
    assert speech.si_sdr(aligned, ref) > speech.si_sdr(late, ref) + 20


def test_wilson_stays_open_where_the_bootstrap_collapses():
    assert stats.bootstrap_ci([1.0] * 100) == [1.0, 1.0]
    lo, hi = stats.wilson_ci(100, 100)
    assert hi == 1.0 and 0.96 < lo < 0.965  # rule of three: about 3% errors still possible
    lo, hi = stats.wilson_ci(50, 100)
    assert lo < 0.5 < hi and abs((0.5 - lo) - (hi - 0.5)) < 1e-3


def test_by_unit_ci_takes_the_wider_interval_and_counts_units_not_rows():
    units = [u for u in range(20) for _ in range(5)]
    out = stats.by_unit_ci([1.0] * 100, units)
    assert out["mean"] == 1.0 and out["ci95_bootstrap"] == [1.0, 1.0]
    assert out["ci95"] == out["ci95_wilson"] and out["ci95"][0] < 0.85  # 20 tasks, not 100 attempts
