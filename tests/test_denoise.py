"""The denoise tool imports DeepFilterNet lazily, so the core env can load it with a fake model."""

import importlib.util
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from ah import audio

_spec = importlib.util.spec_from_file_location(
    "ah_denoise_under_test", Path(__file__).parents[1] / "tools/denoise/src/ah_denoise/__init__.py")
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module  # pydantic resolves annotations through sys.modules
_spec.loader.exec_module(_module)


def fake_tool(calls):
    tool = _module.DenoiseTool()

    def enhance(channels, atten_lim_db):
        calls.append((channels.shape, atten_lim_db))
        return channels * 0.5

    tool._enhance = enhance
    return tool


def test_denoise_runs_at_model_rate_and_keeps_length(tmp_path):
    source = tmp_path / "in.wav"
    audio.write_wav(source, np.full((44_100, 2), 0.2, dtype=np.float32))
    calls = []
    result = fake_tool(calls).run({"audio": source}, {"atten_lim_db": 12}, tmp_path)

    assert calls == [((2, 48_000), 12)]
    assert result.error is None
    out = result.outputs[0]
    assert (out.role, out.kind) == ("denoised", "audio")
    info = audio.info(out.path)
    assert (info.sample_rate, info.frames) == (48_000, 48_000)
    assert result.meta["atten_lim_db"] == 12


@pytest.mark.parametrize("params", [{"atten_lim_db": 0}, {"strength": 1}])
def test_denoise_rejects_bad_params(tmp_path, params):
    source = tmp_path / "in.wav"
    audio.write_wav(source, np.zeros((4_410, 2), dtype=np.float32))
    with pytest.raises(ValidationError):
        fake_tool([]).run({"audio": source}, params, tmp_path)


def _zip(entries):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _serve(monkeypatch, payload):
    monkeypatch.setattr(_module.urllib.request, "urlopen", lambda url: io.BytesIO(payload))


def test_fetch_model_checks_sha256_and_unpacks(tmp_path, monkeypatch):
    payload = _zip({"DeepFilterNet3/config.ini": "[df]\n", "DeepFilterNet3/checkpoints/model.ckpt": "x"})
    _serve(monkeypatch, payload)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        _module.fetch_model(tmp_path, "https://example/model.zip", "0" * 64)
    assert list(tmp_path.iterdir()) == []

    sha = _module.hashlib.sha256(payload).hexdigest()
    model_dir = _module.fetch_model(tmp_path, "https://example/model.zip", sha)
    assert (model_dir / "config.ini").read_text() == "[df]\n"


def test_safe_unzip_rejects_traversal(tmp_path):
    archive = tmp_path / "evil.zip"
    archive.write_bytes(_zip({"../escape.txt": "x"}))
    with pytest.raises(ValueError, match="escapes"):
        _module.safe_unzip(archive, tmp_path / "dest")
    assert not (tmp_path / "escape.txt").exists()
