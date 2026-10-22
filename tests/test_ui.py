"""The local web page's API: upload, agent plan and run, before/after, rollback, and the localhost guards."""

import time

import pytest
from fastapi.testclient import TestClient
from test_agent import ScriptedLLM, plan
from test_core import runner, tone  # noqa: F401  (runner is a fixture)

from ah import registry
from ah.agent import chord_line
from ah.llm import Reply
from ah.registry import ToolSpec
from ah.ui import create_app

POST = {"X-AH-UI": "1"}


@pytest.fixture
def llm():
    return ScriptedLLM()


@pytest.fixture
def client(runner, llm, monkeypatch):  # noqa: F811
    monkeypatch.setitem(registry.TOOLS, "halfgain", ToolSpec("halfgain", "test_core", "HalfGain", None, ("audio",),
                                                            "halfgain", outputs=("half",)))
    return TestClient(create_app(runner.store, runner, llm=llm))


def upload(client, path, name="take.wav"):
    with path.open("rb") as handle:
        return client.post("/api/import", files={"file": (name, handle, "audio/wav")}, headers=POST)


def test_page_and_empty_state(client):
    assert "AUDIO HARNESS" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/..%2Fui.py").status_code == 404
    assert client.get("/api/state").json()["head"] is None
    assert client.get("/api/agent").json()["configured"] is True


def test_guards_block_cross_site_requests(client, tmp_path):
    with tone(tmp_path / "a.wav").open("rb") as handle:
        assert client.post("/api/import", files={"file": ("a.wav", handle)}).status_code == 403
    assert client.get("/api/state", headers={"Host": "evil.example"}).status_code == 400
    assert upload(client, tone(tmp_path / "b.wav"), name="b.exe").status_code == 422


def test_upload_ask_agent_run_compare_and_roll_back(client, llm, tmp_path):
    original = upload(client, tone(tmp_path / "in.wav")).json()
    assert original["label"] == "take" and original["tool"] == "import"
    assert len(client.get(f"/api/artifacts/{original['id']}/peaks?bins=50").json()["peaks"]) == 50
    assert client.get(f"/api/artifacts/{original['id']}/audio").headers["content-type"] == "audio/wav"

    llm.replies += [plan("Сделаю тише", {"tool": "halfgain", "input": "track", "why": "тише"}), Reply(text="Стало тише.")]
    proposed = client.post("/api/agent/message", json={"artifact": original["id"], "text": "тише"}, headers=POST).json()
    assert proposed["status"] == "proposed"
    assert client.post(f"/api/agent/plans/{proposed['id']}/run", headers=POST).json()["status"] == "running"
    for _ in range(200):
        done = client.get(f"/api/agent/plans/{proposed['id']}").json()
        if done["status"] != "running":
            break
        time.sleep(0.05)
    assert done["status"] == "done" and done["summary"] == "Стало тише."
    chat = client.get(f"/api/agent/chat?artifact={original['id']}").json()
    assert [p["id"] for p in chat["plans"]] == [proposed["id"]] and chat["busy"] is False

    half = done["selected"]
    state = client.get(f"/api/state?selected={half}").json()
    assert state["head"] == half and state["before"]["id"] == original["id"]
    assert state["root"]["label"] == "take" and len(state["tree"]) == 2

    assert client.post("/api/rollback", headers=POST).json()["head"] == original["id"]
    assert client.post("/api/rollback", headers=POST).status_code == 404  # an original has no parent
    assert client.post("/api/head", json={"artifact": half}, headers=POST).json()["head"] == half


def test_agent_off_is_a_readable_error(runner, tmp_path):  # noqa: F811
    client = TestClient(create_app(runner.store, runner))
    original = upload(client, tone(tmp_path / "in.wav")).json()
    assert client.get("/api/agent").json()["configured"] is False
    response = client.post("/api/agent/message", json={"artifact": original["id"], "text": "тише"}, headers=POST)
    assert response.status_code == 502 and "не настроен" in response.json()["detail"]


def test_chord_line_drops_no_chord_and_repeats():
    chords = [{"label": "C:maj"}, {"label": "C:maj"}, {"label": "N"}, {"label": "A:min"}, {"label": "F:maj"}]
    assert chord_line(chords) == ["C", "Am", "F"]
