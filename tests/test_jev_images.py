"""Image protocol and policy checks; no external API, browser, or SUMO needed."""

import json
from types import SimpleNamespace

from PIL import Image
import pytest
import requests

from jevlight.controller import JevLightController
from jevlight.images import camera_snapshot
from jevlight.jev_client import JevClient, JevResult
from tests.test_jevlight_controller import make_observation, phase_answer


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "camera.jpg"
    Image.new("RGB", (1024, 768), "gray").save(path)
    return path


def test_official_image_request_uses_multipart_and_bypasses_sdk(monkeypatch, image_path):
    client = JevClient(transport="official", mock=True, model="jev-1.14.0", api_key="test")
    client.mock = False
    client._sdk_client = object()  # Would fail if the legacy SDK path were used.
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(
            status_code=200, headers={}, raise_for_status=lambda: None,
            json=lambda: {"answers": {"phase_i0": phase_answer("NTST")},
                          "model": "jev-1.14.0", "usage": {"input_tokens": 111, "output_tokens": 2}},
        )

    monkeypatch.setattr(requests, "post", post)
    result = client.evaluate(
        {"camera_image": "unique-camera"},
        {"phase_i0": {"type": "choice", "instructions": "Choose from camera_image", "criteria": {"NTST": "go"}}},
        assets={"unique-camera": image_path},
    )
    url, kwargs = calls[0]
    assert url.endswith("/v1/systemone")
    assert kwargs["headers"]["TypeSafe-Danger-Zone-Images"] == "true"
    assert kwargs["headers"]["Authorization"] == "Bearer test"
    assert "Content-Type" not in kwargs["headers"]
    assert "json" not in kwargs
    parts = kwargs["files"]
    assert parts[0][0] == "request"
    assert parts[0][1][2] == "application/json"
    payload = json.loads(parts[0][1][1])
    assert payload["model"] == "jev-1.14.0"
    assert payload["assets"] == {"unique-camera": {"type": "image", "part": "image-0"}}
    assert parts[1][0] == "image-0"
    assert parts[1][1][1] == image_path.read_bytes()
    assert parts[1][1][2] == "image/jpeg"
    prepared = requests.Request("POST", url, **{k: v for k, v in kwargs.items() if k != "timeout"}).prepare()
    assert prepared.headers["Content-Type"].startswith("multipart/form-data; boundary=")
    assert result.answers["phase_i0"]["choice"] == "NTST"
    assert result.input_tokens == 111


def test_region_rejection_is_not_retried(monkeypatch, image_path):
    client = JevClient(transport="official", mock=True, api_key="test", max_retries=8)
    client.mock = False
    calls = []
    response = requests.Response()
    response.status_code = 451
    response.url = "https://api.typesafe.ai/v1/systemone"

    def post(*args, **kwargs):
        calls.append(1)
        return response

    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr("jevlight.jev_client.time.sleep", lambda _: pytest.fail("permanent rejection retried"))
    result = client.evaluate({"image": "ref"}, {}, assets={"ref": image_path})
    assert "451" in result.error
    assert len(calls) == 1


def test_ssl_outage_retries_with_bounded_backoff(monkeypatch, image_path):
    client = JevClient(transport="official", mock=True, api_key="test", max_retries=8)
    client.mock = False
    calls, sleeps = [], []

    def post(*args, **kwargs):
        calls.append(1)
        if len(calls) <= 8:
            raise requests.exceptions.SSLError("unexpected EOF")
        return SimpleNamespace(status_code=200, headers={}, raise_for_status=lambda: None,
                               json=lambda: {"model": "jev-1.14.0", "answers": {}})

    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr("jevlight.jev_client.time.sleep", sleeps.append)
    result = client.evaluate({"image": "ref"}, {}, assets={"ref": image_path})
    assert result.error is None
    assert len(calls) == 9
    assert sleeps == [1, 2, 4, 8, 16, 30, 30, 30]


@pytest.mark.parametrize("size", [(256, 256), (255, 1024), (1569, 512)])
def test_bad_image_dimensions_do_not_make_a_request(monkeypatch, tmp_path, size):
    path = tmp_path / "bad.png"
    Image.new("RGB", size).save(path)
    client = JevClient(transport="official", mock=True, max_retries=0)
    client.mock = False
    monkeypatch.setattr(requests, "post", lambda *a, **kw: pytest.fail("invalid image uploaded"))
    result = client.evaluate({"image": "ref"}, {}, assets={"ref": path})
    assert "dimensions" in result.error


def test_asset_keys_do_not_count_as_references(image_path):
    with pytest.raises(ValueError, match="unreferenced"):
        JevClient._image_parts({"state": {"ref": 1}, "questions": {}}, {"ref": image_path})


@pytest.mark.parametrize("transport", ["mcp", "local"])
def test_image_transport_must_be_explicit(transport, image_path):
    with pytest.raises(ValueError, match="official"):
        JevClient(transport=transport, mock=True).evaluate({"x": "ref"}, {}, assets={"ref": image_path})


class ImageStub(JevClient):
    def __init__(self, answer=None, error=None):
        super().__init__(transport="official", mock=True)
        self.calls = []
        self.answer = answer
        self.error = error

    def evaluate(self, state, questions, assets=None):
        self.calls.append((state, questions, assets))
        key = next(iter(questions))
        return JevResult(answers={key: self.answer}, error=self.error)


def test_image_policy_includes_empty_junctions_without_numeric_oracle(image_path):
    observation = make_observation(empty=True, queue_by_phase={"ETWT": 999})
    client = ImageStub(phase_answer("NTST"))
    controller = JevLightController(client, 15, packaging="per_intersection")
    actions, traces = controller.decide_images([observation], {"i0": image_path}, 3, "Prevent spillback.")
    assert actions == {"i0": 1}
    assert len(client.calls) == 1
    state, questions, assets = client.calls[0]
    assert set(state) == {"intersection_id", "camera_image", "current_phase", "control_interval_s"}
    assert state["camera_image"] in assets
    assert list(questions) == ["phase_i0"]
    assert "Prevent spillback." in questions["phase_i0"]["instructions"]
    assert "999" not in json.dumps(state)
    assert traces[0]["input_mode"] == "image"
    assert traces[0]["fallback"] is None
    assert traces[0]["mock"] is True


@pytest.mark.parametrize("answer", [None, "invalid", {"type": "choice", "choice": "BAD"},
                                  phase_answer("NTST", confidence="NaN")])
def test_invalid_vision_answer_keeps_previous_signal(answer, image_path):
    client = ImageStub(answer)
    controller = JevLightController(client, 15, packaging="per_intersection")
    actions, traces = controller.decide_images([make_observation()], {"i0": image_path}, 0)
    assert actions == {"i0": 0}
    assert traces[0]["fallback"] is not None


def test_camera_snapshot_preserves_native_coordinates_without_subscriptions():
    vehicle = SimpleNamespace(
        getIDList=lambda: ["car"], getPosition3D=lambda _: (100, 200, 0),
        getColor=lambda _: (255, 0, 0, 255), getAngle=lambda _: 90,
        getSpeed=lambda _: 2, getVehicleClass=lambda _: "bus",
        subscribe=lambda *a: pytest.fail("must not overwrite SUMOEnv subscriptions"),
    )
    conn = SimpleNamespace(
        vehicle=vehicle, simulation=SimpleNamespace(getTime=lambda: 10),
        trafficlight=SimpleNamespace(getIDList=lambda: ["i0"], getRedYellowGreenState=lambda _: "Gr"),
    )
    assert camera_snapshot(conn) == {
        "t": 10.0, "vehicles": [["car", 100, 200, 0, 90, 2, 255, 0, 0, 1]], "tls": {"i0": "Gr"},
    }
