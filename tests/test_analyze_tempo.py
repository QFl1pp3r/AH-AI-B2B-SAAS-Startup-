"""tempo_summary is pure numpy; load it without the analyze env's heavy imports."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "ah_analyze_under_test", Path(__file__).parents[1] / "tools/analyze/src/ah_analyze/__init__.py")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
tempo_summary = _module.tempo_summary


def test_steady_120_in_four():
    beats = np.arange(0, 16, 0.5)
    summary = tempo_summary(beats, beats[::4])
    assert summary["bpm"] == pytest.approx(120.0)
    assert summary["meter"] == "4/4"
    assert summary["tempo_stability"] == pytest.approx(0.0, abs=1e-9)
    assert summary["bpm_alternatives"] == [60.0, 240.0]


def test_waltz():
    beats = np.arange(0, 12, 0.4)
    assert tempo_summary(beats, beats[::3])["meter"] == "3/4"


def test_too_few_beats():
    assert tempo_summary(np.array([0.0, 0.5]), np.array([]))["bpm"] is None
