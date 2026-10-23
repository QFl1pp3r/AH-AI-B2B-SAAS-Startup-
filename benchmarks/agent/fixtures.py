"""Build the track_data fixtures for the agent benchmark from a real store.

Each fixture is what ah.agent.track_data returns for one demo situation: a fresh song,
a song with an analysis, a song after separate, the vocals of a song selected, a voice
memo, a long podcast take. Versions and runs are written through ArtifactStore exactly
as Runner stores tool outputs; the models themselves are not run, the agent only ever
sees this metadata. Levels and durations are set to realistic values for the situation
(a 45-minute podcast would be 1 GB of canonical WAV for nothing).

    uv run python benchmarks/agent/fixtures.py   # rewrites benchmarks/agent/fixtures/*.json
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ah import audio
from ah.agent import track_data
from ah.store import ArtifactStore

OUT = Path(__file__).parent / "fixtures"


def levels(duration_s: float, peak: float, rms: float, clipped: float = 0.0, silence: float = 0.0) -> dict[str, Any]:
    return {"duration_s": duration_s, "peak_dbfs": peak, "rms_dbfs": rms, "clipped_ratio": clipped, "silence_ratio": silence,
            "sample_rate": audio.CANONICAL_SR, "channels": 2}


class Builder:
    def __init__(self, home: Path):
        self.store = ArtifactStore(home)
        self.tmp = home.parent
        self.wav = self.tmp / "blob.wav"
        audio.write_wav(self.wav, np.zeros((441, 2), np.float32))

    def original(self, name: str, meta: dict[str, Any]) -> str:
        return self.store.add(self.wav, kind="audio", role="original", label=Path(name).stem,
                              meta=meta | {"source_name": name}).id

    def run(self, tool: str, parent: str, outputs: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, str]:
        """outputs: role -> audio meta dict, or a JSON payload for report roles."""
        run_id = self.store.start_run(tool, "local", {"audio": parent}, params or {})
        made = {}
        for role, value in outputs.items():
            if role in ("analysis", "report"):
                path = self.tmp / f"{role}.json"
                path.write_text(json.dumps(value))
                made[role] = self.store.add(path, kind="json", role=role, parent_id=parent, run_id=run_id).id
            else:
                made[role] = self.store.add(self.wav, kind="audio", role=role, parent_id=parent, run_id=run_id, meta=value).id
        self.store.finish_run(run_id, meta={})
        return made


ANALYSIS = {"bpm": 124.0, "meter": "4/4", "key": {"key": "A minor", "confidence": 0.71},
            "chords": [{"start": 0.0, "end": 2.0, "label": "A:min"}, {"start": 2.0, "end": 4.0, "label": "F:maj"},
                       {"start": 4.0, "end": 6.0, "label": "C:maj"}, {"start": 6.0, "end": 8.0, "label": "G:maj"},
                       {"start": 8.0, "end": 10.0, "label": "A:min"}]}


def build(home: Path) -> dict[str, dict[str, Any]]:
    b = Builder(home)
    song = levels(212.4, -0.3, -11.8)
    fixtures: dict[str, str] = {}

    fixtures["song_fresh"] = b.original("lights_out_final_mix.wav", song)

    analyzed = b.original("night_drive_demo.mp3", song)
    b.run("analyze", analyzed, {"analysis": ANALYSIS})
    fixtures["song_analyzed"] = analyzed

    separated = b.original("rehearsal_take3.wav", levels(184.0, -1.2, -14.5))
    b.run("separate", separated, {"vocals": levels(184.0, -4.1, -21.3, silence=0.31),
                                  "instrumental": levels(184.0, -1.5, -15.2)})
    fixtures["song_separated"] = separated

    with_vocals = b.original("cover_song_raw.wav", levels(201.0, -0.8, -13.1))
    stems = b.run("separate", with_vocals, {"vocals": levels(201.0, -3.9, -20.7, silence=0.28),
                                            "instrumental": levels(201.0, -1.1, -14.0)})
    fixtures["song_vocals_selected"] = stems["vocals"]

    fixtures["voice_memo"] = b.original("voice_memo_2026-10-12.m4a", levels(95.3, -6.2, -31.5, silence=0.18))
    fixtures["podcast_long"] = b.original("interview_raw.wav", levels(2712.0, -2.0, -24.0, clipped=0.0004, silence=0.12))

    # Held-out set: names that point the wrong way, so only the user's words tell song from speech.
    fixtures["song_named_memo"] = b.original("voice_memo_0412.m4a", levels(187.5, -0.6, -12.4))
    fixtures["speech_named_song"] = b.original("final_mix_v3.wav", levels(1305.0, -3.1, -27.2, silence=0.09))

    # Second held-out set (requests from people outside the project): more states of the track, so a
    # request lands on a history of versions, a selected stem or a file that needs no separating at all.
    fixtures["live_long_clipped"] = b.original("gig_full_set_live.wav", levels(402.0, 0.0, -9.6, clipped=0.0031))
    fixtures["phone_call_mono"] = b.original("call_recording_0915.amr", levels(312.0, -9.8, -33.0, silence=0.22) | {"channels": 1})
    fixtures["instrumental_only"] = b.original("guitar_loop_no_vocals.wav", levels(48.0, -2.4, -14.9))
    fixtures["acapella_only"] = b.original("acapella_take2.wav", levels(96.0, -3.0, -19.8, silence=0.15))
    loud = b.original("master_v2_LOUD.wav", levels(198.0, 0.0, -6.1, clipped=0.012))
    fixtures["loud_master"] = loud

    trimmed_song = b.original("summer_demo.wav", levels(240.0, -0.9, -12.7))
    cut = b.run("trim", trimmed_song, {"trimmed": levels(62.0, -1.0, -12.9)}, {"start_s": 30, "end_s": 92})
    fixtures["song_trimmed_selected"] = cut["trimmed"]

    interview = b.original("zoom_interview.m4a", levels(1830.0, -4.4, -28.1, silence=0.14))
    restored = b.run("restore", interview, {"restored": levels(1830.0, -1.0, -20.3, silence=0.16) | {"channels": 1}})
    fixtures["speech_restored_selected"] = restored["restored"]

    full = b.original("band_rehearsal_0603.wav", levels(286.0, -0.5, -12.2, silence=0.04))
    part = b.run("trim", full, {"trimmed": levels(120.0, -0.6, -12.4)}, {"start_s": 60, "end_s": 180})["trimmed"]
    split = b.run("separate", part, {"vocals": levels(120.0, -4.2, -22.0, silence=0.3), "instrumental": levels(120.0, -0.9, -13.5)})
    b.run("analyze", split["instrumental"], {"analysis": ANALYSIS | {"bpm": 97.0, "key": {"key": "E minor", "confidence": 0.64}}})
    cleaned = b.run("denoise", split["vocals"], {"denoised": levels(120.0, -4.5, -22.8, silence=0.33)})
    fixtures["history_full_denoised_vocals"] = cleaned["denoised"]

    return {name: track_data(b.store, artifact) for name, artifact in fixtures.items()}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for name, data in build(Path(tmp) / "home").items():
            if (OUT / f"{name}.json").exists():
                continue  # ids are random: rewriting would change fixtures that finished runs were made with
            (OUT / f"{name}.json").write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
            print(f"\n{name}: {data['selected_version']['role']} of {data['file_name']}, {len(data['versions'])} versions")


if __name__ == "__main__":
    main()
