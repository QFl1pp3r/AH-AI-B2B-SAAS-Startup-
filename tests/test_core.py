import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from ah import audio, registry
from ah.cli import main as cli_main
from ah.config import Settings
from ah.contract import ToolOutput, ToolResult
from ah.registry import ToolSpec
from ah.runner import RunFailed, Runner
from ah.store import ArtifactStore, StoreError
from ah.transport import pack_result, unpack_result


def tone(path: Path, sr: int = 22_050, seconds: float = 1.0, channels: int = 1) -> Path:
    t = np.arange(int(sr * seconds)) / sr
    data = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    if channels > 1:
        data = np.stack([data] * channels, axis=1)
    audio.write_wav(path, data, sr)
    return path


class HalfGain:
    """Test tool: writes the input at half level at 48 kHz, like Sidon would."""

    name = "halfgain"

    def load(self, device):
        pass

    def run(self, inputs, params, out_dir):
        data, sr = audio.decode(inputs["audio"])
        target = out_dir / "half.wav"
        audio.write_wav(target, audio.resample(data * 0.5, sr, 48_000), 48_000)
        return ToolResult(tool=self.name, outputs=[ToolOutput(role="half", path=str(target), kind="audio")])


class Broken:
    name = "broken"

    def load(self, device):
        pass

    def run(self, inputs, params, out_dir):
        raise RuntimeError("boom")


@pytest.fixture
def runner(tmp_path, monkeypatch):
    for name, factory in (("halfgain", "HalfGain"), ("broken", "Broken")):
        monkeypatch.setitem(registry.TOOLS, name, ToolSpec(name, __name__, factory, None, ("audio",), name))
    return Runner(ArtifactStore(tmp_path / "home"), Settings(home=tmp_path / "home"))


def test_ingest_makes_canonical_wav(runner, tmp_path):
    artifact = runner.ingest(tone(tmp_path / "in.wav", sr=22_050, channels=1))
    info = audio.info(runner.store.path(artifact.id))
    assert (info.sample_rate, info.channels) == (audio.CANONICAL_SR, 2)
    assert abs(info.duration - 1.0) < 1e-3
    assert runner.store.head() == artifact.id


def test_outputs_are_canonical_and_lineage_and_rollback(runner, tmp_path):
    original = runner.ingest(tone(tmp_path / "in.wav"))
    first = runner.run("halfgain", {"audio": original.id}).artifacts[0]
    second = runner.run("halfgain", {"audio": first.id}).artifacts[0]
    assert audio.info(runner.store.path(second.id)).sample_rate == audio.CANONICAL_SR
    assert [a.id for a in runner.store.lineage(second.id)] == [second.id, first.id, original.id]
    assert runner.store.head() == second.id
    assert runner.store.rollback() == first.id
    branch = runner.run("halfgain", {"audio": first.id}).artifacts[0]
    assert {c.id for c in runner.store.children(first.id)} == {second.id, branch.id}
    runner.store.set_head(original.id)
    with pytest.raises(StoreError):
        runner.store.rollback()


def test_failed_run_is_recorded_and_head_unchanged(runner, tmp_path):
    original = runner.ingest(tone(tmp_path / "in.wav"))
    with pytest.raises(RunFailed) as caught:
        runner.run("broken", {"audio": original.id})
    run = runner.store.get_run(caught.value.run_id)
    assert run.status == "failed" and "boom" in run.error
    assert runner.store.head() == original.id


def test_inspect_builtin(runner, tmp_path):
    original = runner.ingest(tone(tmp_path / "in.wav"))
    report = runner.run("inspect", {"audio": original.id}).artifacts[0]
    content = json.loads(runner.store.path(report.id).read_text())
    assert content["peak_dbfs"] == pytest.approx(-6.02, abs=0.05)
    assert runner.store.head() == original.id  # json outputs never move the head


def test_blobs_are_deduplicated_and_read_only(runner, tmp_path):
    a = runner.ingest(tone(tmp_path / "in.wav"))
    b = runner.ingest(tone(tmp_path / "in.wav"))
    assert a.id != b.id and a.sha256 == b.sha256
    assert not (runner.store.path(a.id).stat().st_mode & 0o222)


def test_transport_roundtrip_and_rejects_traversal(tmp_path):
    source = tone(tmp_path / "x.wav")
    packed = pack_result(ToolResult(tool="t", outputs=[ToolOutput(role="r", path=str(source), kind="audio")]))
    result = unpack_result(packed, tmp_path / "out")
    assert Path(result.outputs[0].path).read_bytes() == source.read_bytes()

    evil = io.BytesIO()
    with zipfile.ZipFile(evil, "w") as archive:
        archive.writestr("../../evil.wav", b"x")
        archive.writestr("result.json", ToolResult(
            tool="t", outputs=[ToolOutput(role="r", path="../../evil.wav", kind="audio")]).model_dump_json())
    with pytest.raises(ValueError):
        unpack_result(evil.getvalue(), tmp_path / "out2")


def test_cli_end_to_end(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AH_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    assert cli_main(["import", str(tone(tmp_path / "in.wav"))]) == 0
    imported = json.loads(capsys.readouterr().out)
    assert cli_main(["inspect"]) == 0
    run = json.loads(capsys.readouterr().out)
    assert run["artifacts"][0]["parent_id"] == imported["id"]
    assert cli_main(["rollback"]) == 1  # an upload has no parent
    assert "error" in json.loads(capsys.readouterr().out)


def test_server_auth_and_run(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from ah import server

    monkeypatch.setenv("AH_SERVER_TOKEN", "t" * 32)
    monkeypatch.setattr(server, "SERVED_TOOLS", ("inspect",))
    client = TestClient(server.create_app())
    wav = tone(tmp_path / "in.wav").read_bytes()
    assert client.post("/v1/tools/inspect", files={"audio": ("a.wav", wav)},
                       headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.post("/v1/tools/separate", files={"audio": ("a.wav", wav)},
                       headers={"Authorization": "Bearer " + "t" * 32}).status_code == 404
    ok = client.post("/v1/tools/inspect", files={"audio": ("a.wav", wav)},
                     headers={"Authorization": "Bearer " + "t" * 32})
    assert ok.status_code == 200
    result = unpack_result(ok.content, tmp_path / "out")
    assert result.outputs[0].role == "report"
