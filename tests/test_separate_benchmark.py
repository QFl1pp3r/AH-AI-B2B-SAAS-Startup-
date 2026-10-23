"""Scoring and harness integration without downloading the separator model."""

import json
import importlib.util
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ah.runner import RunOutcome, Runner
from ah.audio import ffmpeg_exe

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "separate.py"
SPEC = importlib.util.spec_from_file_location("separate_benchmark", SCRIPT)
separate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(separate)


def make_track(root):
    track = root / "test" / "song"
    track.mkdir(parents=True)
    t = np.arange(4_410) / 44_100
    vocals = np.stack([0.2 * np.sin(2 * np.pi * 440 * t)] * 2, axis=1)
    drums = np.stack([0.1 * np.sin(2 * np.pi * 100 * t)] * 2, axis=1)
    bass = np.stack([0.08 * np.sin(2 * np.pi * 70 * t)] * 2, axis=1)
    other = np.stack([0.05 * np.sin(2 * np.pi * 210 * t)] * 2, axis=1)
    for name, audio in {"vocals": vocals, "drums": drums, "bass": bass,
                        "other": other, "mixture": vocals + drums + bass + other}.items():
        sf.write(track / f"{name}.wav", audio, 44_100, subtype="FLOAT")
    return track


def test_scores_match_stems_and_detect_bad_vocals(tmp_path):
    track = make_track(tmp_path)
    instrumental = sum((sf.read(track / f"{part}.wav")[0] for part in ("drums", "bass", "other")))
    sf.write(track / "instrumental.wav", instrumental, 44_100, subtype="FLOAT")
    estimates = {"vocals": track / "vocals.wav", "instrumental": track / "instrumental.wav"}
    perfect = separate.score_track(track, estimates)
    assert perfect["vocals_sdr_db"] == 100.0
    assert perfect["vocals_si_sdr_db"] == 100.0
    assert perfect["instrumental_sdr_db"] == pytest.approx(100.0)

    contaminated = sf.read(track / "vocals.wav")[0] + sf.read(track / "drums.wav")[0]
    sf.write(track / "wrong.wav", contaminated, 44_100, subtype="FLOAT")
    wrong = separate.score_track(track, {**estimates, "vocals": track / "wrong.wav"})
    assert wrong["vocals_sdr_db"] < perfect["vocals_sdr_db"]
    assert wrong["vocals_si_sdr_db"] < perfect["vocals_si_sdr_db"]


def test_run_and_resume_do_not_invoke_separator_again(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    track = make_track(dataset)
    out = tmp_path / "results"
    calls = []

    def fake_run(self, tool, inputs, params, *, backend, project):
        calls.append(tool)
        instrumental = sum((sf.read(track / f"{part}.wav")[0] for part in ("drums", "bass", "other")))
        output = out / "instrumental.wav"
        sf.write(output, instrumental, 44_100, subtype="FLOAT")
        vocals = self.store.add(track / "vocals.wav", kind="audio", role="vocals")
        backing = self.store.add(output, kind="audio", role="instrumental")
        return RunOutcome("fake-run", tool, backend, [vocals, backing])

    monkeypatch.setattr(Runner, "run", fake_run)
    first = separate.run(dataset, out, backend="local", model="model", limit=None, resume=False)
    second = separate.run(dataset, out, backend="local", model="model", limit=None, resume=True)
    assert calls == ["separate"]
    assert first["succeeded"] == second["succeeded"] == 1
    assert first["vocals_sdr_db"]["median"] == 100.0
    assert len((out / "results.jsonl").read_text().splitlines()) == 1
    assert json.loads((out / "summary.json").read_text())["tracks"] == 1
    assert json.loads((out / "config.json").read_text())["scorer_version"] == 2


def test_compressed_musdb_decode(tmp_path):
    track = make_track(tmp_path)
    container = tmp_path / "test" / "song.stem.mp4"
    names = ("mixture", "drums", "bass", "other", "vocals")
    command = [ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error"]
    for name in names:
        command.extend(["-i", str(track / f"{name}.wav")])
    for index in range(len(names)):
        command.extend(["-map", f"{index}:a:0"])
    # Matroska preserves explicit stream order; the reader selects streams
    # independently of the container's filename extension.
    command.extend(["-c:a", "pcm_s16le", "-f", "matroska", str(container)])
    subprocess.run(command, check=True, capture_output=True)

    decoded = tmp_path / "decoded"
    decoded.mkdir()
    separate.decode_stems(container, decoded)
    assert separate.track_name(container) == "song"
    assert container in separate.tracks_in(tmp_path)
    for name in names:
        data, rate = sf.read(decoded / f"{name}.wav")
        expected, _ = sf.read(track / f"{name}.wav")
        assert rate == 44_100 and data.shape[1] == 2
        assert len(data) > 0 and np.max(np.abs(data)) > 0
        np.testing.assert_allclose(data, expected, atol=1e-4)


def test_compressed_stream_ids_follow_musdb_contract(tmp_path, monkeypatch):
    expected = ("mixture", "drums", "bass", "other", "vocals")
    assert separate.STEMS == expected
    calls = []

    def capture(command, **kwargs):
        calls.append(command)

    monkeypatch.setattr(separate.subprocess, "run", capture)
    separate.decode_stems(tmp_path / "song.stem.mp4", tmp_path)
    for stream, stem in enumerate(expected):
        assert calls[stream][calls[stream].index("-map") + 1] == f"0:a:{stream}"
        assert calls[stream][-1] == str(tmp_path / f"{stem}.wav")
