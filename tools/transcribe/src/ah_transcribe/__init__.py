"""Lyrics and speech transcription with word timestamps (faster-whisper).

Run it on the separated vocals of a song: the backing track is what breaks ASR
on music. Do not denoise first (it made Whisper CER worse in Jamal's benchmark).
Weights are CTranslate2 model.bin files, not pickles, pinned to a reviewed
revision of the repo faster-whisper itself resolves the name to.

The text goes to the lyrics.json artifact only. Tool meta carries counts and the
language, because the agent sees meta and anything sung on the record is
untrusted input for it (prompt injection) and often copyrighted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from ah import audio
from ah.contract import ToolOutput, ToolResult, write_json

MODEL_SR = 16_000
MODELS = {
    # mobiuslabsgmbh/faster-whisper-large-v3-turbo now redirects here.
    "large-v3-turbo": ("dropbox-dash/faster-whisper-large-v3-turbo", "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"),
    "large-v3": ("Systran/faster-whisper-large-v3", "edaa852ec7e145841d8ffdb056a99866b5f0a478"),
}
# Whisper's classic output on silence and instrumental breaks: subtitle credits learned from film subtitles.
HALLUCINATIONS = ("субтитры", "редактор субтитров", "корректор", "продолжение следует", "спасибо за просмотр",
                  "подписывайтесь на канал", "thanks for watching", "subtitles by", "amara.org")


class Params(BaseModel, extra="forbid"):
    model: Literal["large-v3-turbo", "large-v3"] = "large-v3-turbo"
    language: str | None = None  # ISO 639-1, e.g. "ru"; None = detect
    vad: bool = True


def hallucinated(seg) -> bool:
    text = seg.text.strip().lower()
    if not text or any(text.startswith(p) for p in HALLUCINATIONS):
        return True
    if seg.no_speech_prob > 0.6 and seg.avg_logprob < -1.0:
        return True
    return seg.compression_ratio > 2.4  # a decoding loop repeating one phrase


def segment_dict(seg) -> dict[str, Any]:
    return {"start": round(seg.start, 2), "end": round(seg.end, 2), "text": seg.text.strip(),
            "words": [{"start": round(w.start, 2), "end": round(w.end, 2), "word": w.word.strip(),
                       "p": round(w.probability, 3)} for w in seg.words or []]}


class TranscribeTool:
    name = "transcribe"

    def __init__(self) -> None:
        self.device = "cpu"
        self._models: dict[str, Any] = {}

    def load(self, device: str) -> None:
        self.device = device

    def _model(self, name: str):
        if name not in self._models:
            from faster_whisper import WhisperModel

            repo, revision = MODELS[name]
            # Keep one model resident: a T4 has 16 GB and other tools share it.
            self._models = {name: WhisperModel(repo, revision=revision, device=self.device,
                                               compute_type="float16" if self.device == "cuda" else "int8")}
        return self._models[name]

    def run(self, inputs: dict[str, Path], params: dict[str, Any], out_dir: Path) -> ToolResult:
        options = Params.model_validate(params)
        data, sr = audio.decode(inputs["audio"])
        mono = audio.resample(audio.to_mono(data), sr, MODEL_SR)
        # condition_on_previous_text=False stops one misheard line from dragging the next ones along.
        segments, info = self._model(options.model).transcribe(
            mono, language=options.language, word_timestamps=True, vad_filter=options.vad,
            condition_on_previous_text=False, beam_size=5)
        kept, dropped = [], 0
        for seg in segments:
            if hallucinated(seg):
                dropped += 1
            else:
                kept.append(segment_dict(seg))

        lyrics = {"model": options.model, "language": info.language,
                  "language_probability": round(info.language_probability, 3),
                  "text": "\n".join(s["text"] for s in kept), "segments": kept}
        write_json(out_dir / "lyrics.json", lyrics)
        meta = {"model": options.model, "language": info.language,
                "language_probability": lyrics["language_probability"], "lines": len(kept),
                "words": sum(len(s["words"]) for s in kept), "dropped_segments": dropped}
        warnings = []
        if not kept:
            warnings.append("no singing or speech recognized")
        if dropped:
            warnings.append(f"dropped {dropped} segments that look like Whisper hallucinations (silence, loops, subtitle credits)")
        if info.language_probability < 0.5 and options.language is None:
            warnings.append(f"language detection unsure ({info.language}); pass language explicitly")
        return ToolResult(tool=self.name, outputs=[ToolOutput(role="lyrics", path=str(out_dir / "lyrics.json"),
                                                               kind="json", meta=meta)],
                          meta=meta | {"device": self.device}, warnings=warnings)
