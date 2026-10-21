"""Tempo, beats, downbeats, key and chords.

- beats/downbeats: Beat This! (default, MIT, checkpoint "final0") or madmom RNN+DBN;
- key: madmom CNNKeyRecognitionProcessor (24 major/minor keys);
- chords: madmom CNN features + CRF (major/minor triads plus "N").

madmom's pretrained models are CC BY-NC-SA 4.0: fine for research and a demo,
not for a commercial product.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Literal

import numpy as np
import soundfile as sf
from pydantic import BaseModel

from ah import audio
from ah.contract import ToolOutput, ToolResult, write_json

MADMOM_SR = 44_100


class Params(BaseModel, extra="forbid"):
    beats: Literal["beat_this", "madmom"] = "beat_this"
    key: bool = True
    chords: bool = True


class AnalyzeTool:
    name = "analyze"

    def __init__(self) -> None:
        self.device = "cpu"
        self._beat_this = None

    def load(self, device: str) -> None:
        self.device = device

    # --- beats -----------------------------------------------------------

    def _beats_beat_this(self, mono: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        if self._beat_this is None:
            from beat_this.inference import Audio2Beats

            self._beat_this = Audio2Beats(checkpoint_path="final0", device=self.device, dbn=False)
        beats, downbeats = self._beat_this(mono, sr)
        return np.asarray(beats, dtype=float), np.asarray(downbeats, dtype=float)

    @staticmethod
    def _beats_madmom(wav_path: str) -> tuple[np.ndarray, np.ndarray]:
        from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor

        activations = RNNDownBeatProcessor()(wav_path)
        tracked = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)(activations)
        if len(tracked) == 0:
            return np.zeros(0), np.zeros(0)
        return tracked[:, 0], tracked[tracked[:, 1] == 1, 0]

    # --- key and chords --------------------------------------------------

    @staticmethod
    def _key(wav_path: str) -> dict[str, Any]:
        from madmom.features.key import CNNKeyRecognitionProcessor, key_prediction_to_label

        prediction = CNNKeyRecognitionProcessor()(wav_path)
        probabilities = prediction[0]
        order = np.argsort(probabilities)[::-1][:3]
        candidates = []
        for index in order:
            one_hot = np.zeros_like(prediction)
            one_hot[0, index] = 1
            candidates.append({"key": key_prediction_to_label(one_hot), "probability": round(float(probabilities[index]), 3)})
        return {"key": candidates[0]["key"], "confidence": candidates[0]["probability"], "candidates": candidates}

    @staticmethod
    def _chords(wav_path: str) -> list[dict[str, Any]]:
        from madmom.features.chords import CNNChordFeatureProcessor, CRFChordRecognitionProcessor

        features = CNNChordFeatureProcessor()(wav_path)
        segments = CRFChordRecognitionProcessor()(features)
        return [
            {"start": round(float(start), 3), "end": round(float(end), 3), "label": str(label)}
            for start, end, label in segments
        ]

    # --- run -------------------------------------------------------------

    def run(self, inputs: dict[str, Path], params: dict[str, Any], out_dir: Path) -> ToolResult:
        options = Params.model_validate(params)
        data, sr = audio.decode(inputs["audio"])
        mono = audio.to_mono(data)
        warnings: list[str] = []

        with tempfile.TemporaryDirectory() as tmp:
            # madmom reads files itself; give it 16-bit PCM at its native rate.
            madmom_wav = str(Path(tmp) / "mono.wav")
            sf.write(madmom_wav, audio.resample(mono, sr, MADMOM_SR), MADMOM_SR, subtype="PCM_16")

            if options.beats == "beat_this":
                beats, downbeats = self._beats_beat_this(mono, sr)
            else:
                beats, downbeats = self._beats_madmom(madmom_wav)
            report: dict[str, Any] = {"beat_tracker": options.beats, **tempo_summary(beats, downbeats)}
            if report["bpm"] is None:
                warnings.append("fewer than 4 beats found: no tempo estimate")
            elif report["tempo_stability"] is not None and report["tempo_stability"] > 0.08:
                warnings.append("tempo varies a lot (rubato or tracking errors); treat BPM as approximate")
            if options.key:
                report["key"] = self._key(madmom_wav)
                if report["key"]["confidence"] < 0.5:
                    warnings.append("key is uncertain; check the candidates")
            if options.chords:
                report["chords"] = self._chords(madmom_wav)

        report["beats"] = [round(float(t), 3) for t in beats]
        report["downbeats"] = [round(float(t), 3) for t in downbeats]
        target = out_dir / "analysis.json"
        write_json(target, report)
        summary = {k: report[k] for k in ("bpm", "meter") if k in report}
        if "key" in report:
            summary["key"] = report["key"]["key"]
        if "chords" in report:
            summary["chord_count"] = sum(1 for c in report["chords"] if c["label"] != "N")
        return ToolResult(
            tool=self.name,
            outputs=[ToolOutput(role="analysis", path=str(target), kind="json", meta=summary)],
            meta=summary,
            warnings=warnings,
        )


def tempo_summary(beats: np.ndarray, downbeats: np.ndarray) -> dict[str, Any]:
    """BPM from the median inter-beat interval, plus how steady it is and the likely meter."""
    beats = np.asarray(beats, dtype=float)
    downbeats = np.asarray(downbeats, dtype=float)
    if beats.size < 4:
        return {"bpm": None, "bpm_alternatives": [], "tempo_stability": None, "meter": None,
                "beat_count": int(beats.size)}
    intervals = np.diff(beats)
    bpm = 60.0 / float(np.median(intervals))
    stability = float(np.std(60.0 / intervals) / bpm)
    meter = None
    if downbeats.size >= 2:
        per_bar = [int(np.sum((beats >= a) & (beats < b))) for a, b in zip(downbeats[:-1], downbeats[1:])]
        values, counts = np.unique(per_bar, return_counts=True)
        meter = f"{int(values[np.argmax(counts)])}/4"
    return {
        "bpm": round(bpm, 2),
        # Half/double tempo is the classic octave error; give both so a person can pick.
        "bpm_alternatives": [round(bpm / 2, 2), round(bpm * 2, 2)],
        "tempo_stability": round(stability, 4),
        "meter": meter,
        "beat_count": int(beats.size),
    }
