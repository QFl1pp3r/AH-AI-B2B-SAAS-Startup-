import json

import numpy as np
import pytest

from ah import audio
from ah.builtin import TrimTool
from ah.cli import main as cli_main

SR = audio.CANONICAL_SR


def padded_tone(path, lead=0.5, body=1.0, tail=0.7):
    """Silence, a 440 Hz tone, silence; stereo with the right channel inverted."""
    t = np.arange(int(SR * body)) / SR
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    mono = np.concatenate([np.zeros(int(SR * lead)), tone, np.zeros(int(SR * tail))]).astype(np.float32)
    audio.write_wav(path, np.stack([mono, -mono], axis=1), SR)
    return path


def run(tmp_path, params):
    source = padded_tone(tmp_path / "in.wav")
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    result = TrimTool().run({"audio": source}, params, out)
    return result, audio.decode(result.outputs[0].path)[0]


def test_range_is_sample_exact(tmp_path):
    result, data = run(tmp_path, {"start_s": 0.25, "end_s": 1.75, "fade_ms": 0})
    assert data.shape == (int(1.5 * SR), 2)
    assert result.meta["offset_s"] == 0.25
    original = audio.decode(tmp_path / "in.wav")[0]
    np.testing.assert_array_equal(data, original[int(0.25 * SR):int(1.75 * SR)])


def test_silence_trim_keeps_pad_and_survives_inverted_stereo(tmp_path):
    result, data = run(tmp_path, {"silence": True, "pad_ms": 50})
    assert result.meta["offset_s"] == pytest.approx(0.45, abs=0.011)
    assert result.meta["duration_s"] == pytest.approx(1.1, abs=0.021)


def test_fade_only_on_cut_edges(tmp_path):
    _, data = run(tmp_path, {"start_s": 0.6, "fade_ms": 5})
    assert data[0, 0] == 0.0
    assert np.all(data[-int(0.7 * SR):] == 0.0)  # untouched tail, no fade applied to the end


def test_errors(tmp_path):
    with pytest.raises(ValueError, match="empty range"):
        run(tmp_path, {"start_s": 5.0})
    with pytest.raises(ValueError, match="nothing above"):
        run(tmp_path, {"end_s": 0.4, "silence": True})
    with pytest.raises(ValueError):
        run(tmp_path, {"start": 1.0})  # unknown params are rejected


def test_cli_trim(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AH_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    assert cli_main(["import", str(padded_tone(tmp_path / "in.wav"))]) == 0
    capsys.readouterr()
    assert cli_main(["trim", "--start", "0.5", "--end", "1.5"]) == 0
    artifact = json.loads(capsys.readouterr().out)["artifacts"][0]
    assert artifact["role"] == "trimmed"
    assert cli_main(["show"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert audio.info(shown["path"]).duration == pytest.approx(1.0, abs=1e-4)
