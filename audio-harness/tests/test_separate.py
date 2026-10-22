from pathlib import Path

import numpy as np
import pytest

from ah import audio
from ah_separate import SeparateTool


class FakeSession:
    def __init__(self, roles=("instrumental", "vocals"), stem_seconds=1.0):
        self.roles = roles
        self.stem_seconds = stem_seconds

    def infer(self, folder, store_dir, output_format):
        assert output_format == "wav_float32"
        result = []
        for role in self.roles:
            path = Path(store_dir) / f"{role}.wav"
            audio.write_wav(path, np.zeros((round(self.stem_seconds * 44_100), 2), dtype=np.float32))
            result.append({"output_id": role, "output_path": str(path)})
        return result


def run_fake(tmp_path, monkeypatch, session):
    source = tmp_path / "mix.wav"
    audio.write_wav(source, np.zeros((44_100, 2), dtype=np.float32))
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    tool = SeparateTool()
    monkeypatch.setattr(tool, "_session", lambda model: session)
    return tool.run({"audio": source}, {}, out_dir)


def test_separate_orders_stems_by_role(tmp_path, monkeypatch):
    result = run_fake(tmp_path, monkeypatch, FakeSession())
    assert [output.role for output in result.outputs] == ["vocals", "instrumental"]


@pytest.mark.parametrize("session", [FakeSession(("vocals",)), FakeSession(stem_seconds=0.5)])
def test_separate_rejects_incomplete_or_misaligned_stems(tmp_path, monkeypatch, session):
    with pytest.raises(ValueError):
        run_fake(tmp_path, monkeypatch, session)
