"""The local web page's API: upload, run, before/after, rollback, and the localhost guards."""

import pytest
from fastapi.testclient import TestClient
from test_core import runner, tone  # noqa: F401  (runner is a fixture)

from ah.ui import chord_line, create_app

POST = {"X-AH-UI": "1"}


@pytest.fixture
def client(runner):  # noqa: F811
    return TestClient(create_app(runner.store, runner))


def upload(client, path, name="take.wav"):
    with path.open("rb") as handle:
        return client.post("/api/import", files={"file": (name, handle, "audio/wav")}, headers=POST)


def test_page_and_empty_state(client):
    assert "AUDIO HARNESS" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/..%2Fui.py").status_code == 404
    state = client.get("/api/state").json()
    assert state["head"] is None and {t["name"] for t in state["tools"]} >= {"separate", "denoise", "analyze"}


def test_guards_block_cross_site_requests(client, tmp_path):
    with tone(tmp_path / "a.wav").open("rb") as handle:
        assert client.post("/api/import", files={"file": ("a.wav", handle)}).status_code == 403
    assert client.get("/api/state", headers={"Host": "evil.example"}).status_code == 400
    assert upload(client, tone(tmp_path / "b.wav"), name="b.exe").status_code == 422


def test_upload_run_compare_and_roll_back(client, tmp_path):
    original = upload(client, tone(tmp_path / "in.wav")).json()
    assert original["label"] == "take" and original["tool"] == "import"
    assert len(client.get(f"/api/artifacts/{original['id']}/peaks?bins=50").json()["peaks"]) == 50
    assert client.get(f"/api/artifacts/{original['id']}/audio").headers["content-type"] == "audio/wav"

    report = client.post("/api/run/inspect", json={"artifact": original["id"]}, headers=POST).json()["selected"]
    # A JSON report is selected now; the next tool must act on the audio it describes.
    half = client.post("/api/run/halfgain", json={"artifact": report}, headers=POST).json()["selected"]
    state = client.get(f"/api/state?selected={half}").json()
    assert state["head"] == half and state["before"]["id"] == original["id"]
    assert state["tree"][0]["id"] == original["id"] and len(state["tree"]) == 3

    assert client.post("/api/rollback", headers=POST).json()["head"] == original["id"]
    assert client.post("/api/rollback", headers=POST).status_code == 404  # an original has no parent
    assert client.post("/api/head", json={"artifact": half}, headers=POST).json()["head"] == half


def test_failed_tool_is_a_readable_error(client, tmp_path):
    original = upload(client, tone(tmp_path / "in.wav")).json()
    response = client.post("/api/run/broken", json={"artifact": original["id"]}, headers=POST)
    assert response.status_code == 422 and "boom" in response.json()["detail"]


def test_chord_line_drops_no_chord_and_repeats():
    chords = [{"label": "C:maj"}, {"label": "C:maj"}, {"label": "N"}, {"label": "A:min"}, {"label": "F:maj"}]
    assert chord_line(chords) == ["C", "Am", "F"]
