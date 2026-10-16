"""Unit tests for the local transport, the local Jev server, and the
jevlight/cojevlight agent modes (no network beyond loopback, no SUMO)."""

from __future__ import annotations

import threading
from typing import Any, Dict, List

import requests

from jevlight.controller import (
    CONGESTION_QUESTION_PREFIX,
    JevLightController,
    MaxPressureController,
    cojevlight_phase_question,
    jevlight_phase_question,
    phase_pressure,
)
from jevlight.jev_client import (
    DEFAULT_LOCAL_BASE_URL,
    JEV_API_KEY_ENV_VARS,
    JevClient,
    JevResult,
)
from jevlight.local_server import (
    MockBackend,
    VllmBackend,
    choice_prompt,
    label_logprob,
    make_server,
    noul_prompt,
    softmax_scores,
)
from tests.test_jevlight_controller import (
    StubJevClient,
    make_observation,
    phase_answer,
)


def completions_response(top_logprobs: Dict[str, float], text: str = "") -> Dict[str, Any]:
    return {
        "model": "tev1-test",
        "choices": [
            {
                "text": text,
                "logprobs": {"top_logprobs": [top_logprobs]},
            }
        ],
        "usage": {"prompt_tokens": 111, "completion_tokens": 1},
    }


# ---------------------------------------------------------------------- #
# Client: local transport
# ---------------------------------------------------------------------- #


class TestLocalTransportClient:
    def make_client(self, **kwargs):
        defaults = dict(
            transport="local",
            base_url="http://127.0.0.1:9",
            api_key=None,
        )
        defaults.update(kwargs)
        return JevClient(**defaults)

    def test_no_api_key_required_and_default_base_url(self, monkeypatch):
        for name in JEV_API_KEY_ENV_VARS:
            monkeypatch.delenv(name, raising=False)
        client = JevClient(transport="local")
        assert client.base_url == DEFAULT_LOCAL_BASE_URL

    def test_env_base_url(self, monkeypatch):
        monkeypatch.setenv("JEV_LOCAL_BASE_URL", "http://gpu-host:9000/")
        client = JevClient(transport="local")
        assert client.base_url == "http://gpu-host:9000"

    def test_hosted_transports_still_require_a_key(self, monkeypatch):
        for name in JEV_API_KEY_ENV_VARS:
            monkeypatch.delenv(name, raising=False)
        for transport in ("mcp", "official"):
            try:
                JevClient(transport=transport)
            except ValueError as exc:
                assert "API key" in str(exc)
            else:
                raise AssertionError(f"{transport} accepted a missing key")

    def test_posts_wire_format_without_model_or_auth(self, monkeypatch):
        # Regression: a hosted key in the environment must not leak to a
        # third-party local server as a Bearer token.
        monkeypatch.setenv("JEV_API_KEY", "jev_secret")
        captured: List[Dict[str, Any]] = []

        def fake_post(url, headers=None, json=None, timeout=None):
            captured.append({"url": url, "headers": headers, "json": json})
            return type(
                "R",
                (),
                {
                    "status_code": 200,
                    "headers": {},
                    "raise_for_status": lambda self: None,
                    "json": lambda self: {
                        "model": "tev1",
                        "answers": {"q1": {"type": "noul", "noul": 0.5}},
                        "usage": {"input_tokens": 10, "output_tokens": 2},
                    },
                },
            )()

        monkeypatch.setattr(requests, "post", fake_post)
        client = self.make_client()
        result = client.evaluate({"a": 1}, {"q1": {"type": "noul", "instructions": "x"}})

        assert len(captured) == 1
        call = captured[0]
        assert call["url"] == "http://127.0.0.1:9/v1/systemone"
        assert "Authorization" not in call["headers"]
        assert "model" not in call["json"]
        assert call["json"]["state"] == {"a": 1}
        assert result.answers["q1"]["noul"] == 0.5
        assert result.model == "tev1"
        assert result.input_tokens == 10
        assert result.error is None

    def test_model_sent_when_set_and_auth_when_keyed(self, monkeypatch):
        captured: List[Dict[str, Any]] = []

        def fake_post(url, headers=None, json=None, timeout=None):
            captured.append({"headers": headers, "json": json})
            response = completions_response({})
            response.update(
                {"answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}}
            )
            return type(
                "R",
                (),
                {
                    "status_code": 200,
                    "headers": {},
                    "raise_for_status": lambda self: None,
                    "json": lambda self: response,
                },
            )()

        monkeypatch.setattr(requests, "post", fake_post)
        client = self.make_client(model="laya", api_key="secret")
        client.evaluate({}, {})
        assert captured[0]["json"]["model"] == "laya"
        assert captured[0]["headers"]["Authorization"] == "Bearer secret"

    def test_retries_transient_failure_then_succeeds(self, monkeypatch):
        monkeypatch.setattr("jevlight.jev_client.time.sleep", lambda _: None)
        calls = {"n": 0}

        def fake_post(url, headers=None, json=None, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise requests.ConnectionError("refused")
            body = {
                "answers": {"q": {"type": "noul", "noul": 1.0}},
                "usage": {},
            }
            return type(
                "R",
                (),
                {
                    "status_code": 200,
                    "headers": {},
                    "raise_for_status": lambda self: None,
                    "json": lambda self: body,
                },
            )()

        monkeypatch.setattr(requests, "post", fake_post)
        client = self.make_client(max_retries=2)
        result = client.evaluate({}, {"q": {"type": "noul"}})
        assert calls["n"] == 2
        assert result.answers["q"]["noul"] == 1.0

    def test_error_result_when_retries_exhausted(self, monkeypatch):
        monkeypatch.setattr("jevlight.jev_client.time.sleep", lambda _: None)

        def fake_post(url, headers=None, json=None, timeout=None):
            raise requests.ConnectionError("refused")

        monkeypatch.setattr(requests, "post", fake_post)
        client = self.make_client(max_retries=1)
        result = client.evaluate({}, {})
        assert result.error is not None
        assert result.rate_limited is False


# ---------------------------------------------------------------------- #
# Local server: prompts, logprob math, backends
# ---------------------------------------------------------------------- #


class TestLogprobMath:
    def test_label_variants_match(self):
        table = {" A": -0.1, "B.": -2.0, "C": -4.0}
        assert label_logprob(table, "A") == -0.1
        assert label_logprob(table, "B") == -2.0
        assert label_logprob(table, "D") is None

    def test_softmax_normalizes(self):
        probabilities = softmax_scores({"A": -0.1, "B": -2.0})
        assert abs(sum(probabilities.values()) - 1.0) < 1e-9
        assert probabilities["A"] > 0.8

    def test_missing_label_floored_below_weakest(self):
        probabilities = softmax_scores({"A": -0.1, "B": None})
        assert probabilities["A"] > 0.99
        assert probabilities["B"] < 0.01

    def test_choice_prompt_lists_lettered_options(self):
        prompt, option_ids = choice_prompt(
            {"queue": 3}, "pick a phase", {"ETWT": "east-west", "NTST": "north-south"}
        )
        assert option_ids == ["ETWT", "NTST"]
        assert "STATE:" in prompt and "QUESTION:" in prompt
        assert "A. ETWT — east-west" in prompt
        assert "B. NTST — north-south" in prompt
        assert prompt.rstrip().endswith("Answer with one letter:")

    def test_noul_prompt_ends_with_yes_no(self):
        assert noul_prompt({}, "congested?").endswith("Answer Yes or No:")


class TestVllmBackend:
    def make_backend(self):
        return VllmBackend(base_url="http://127.0.0.1:9", model="tev1")

    def test_choice_answer_from_label_logprobs(self, monkeypatch):
        backend = self.make_backend()
        monkeypatch.setattr(
            backend,
            "_complete",
            lambda prompt: completions_response({" A": -0.1, "B": -2.0}),
        )
        answer = backend._answer_choice(
            {},
            {
                "type": "choice",
                "instructions": "pick",
                "criteria": {"ETWT": "a", "NTST": "b"},
            },
        )
        assert answer["type"] == "choice"
        assert answer["choice"] == "ETWT"
        assert abs(sum(answer["probabilities"].values()) - 1.0) < 1e-9
        assert answer["confidence"] == max(answer["probabilities"].values())

    def test_noul_answer(self, monkeypatch):
        backend = self.make_backend()
        monkeypatch.setattr(
            backend,
            "_complete",
            lambda prompt: completions_response({" Yes": -0.3, " No": -1.7}),
        )
        answer = backend._answer_noul({}, {"type": "noul", "instructions": "q"})
        assert answer["type"] == "noul"
        expected = softmax_scores({"yes": -0.3, "no": -1.7})["yes"]
        assert abs(answer["noul"] - expected) < 1e-9

    def test_generated_text_fallback_when_labels_missing(self, monkeypatch):
        backend = self.make_backend()
        monkeypatch.setattr(
            backend,
            "_complete",
            lambda prompt: completions_response({"x": -0.5}, text=" B."),
        )
        answer = backend._answer_choice(
            {},
            {
                "type": "choice",
                "instructions": "pick",
                "criteria": {"ETWT": "a", "NTST": "b"},
            },
        )
        assert answer["choice"] == "NTST"

    def test_score_answer_carries_legend(self, monkeypatch):
        backend = self.make_backend()
        monkeypatch.setattr(
            backend,
            "_complete",
            lambda prompt: completions_response({"A": -0.2, "B": -3.0}),
        )
        question = {
            "type": "score",
            "instructions": "grade",
            "criteria": {"low": "l", "high": "h"},
        }
        answer = backend._answer_choice({}, question)
        assert answer["type"] == "score"
        assert answer["score"] == "low"
        assert answer["legend"] == {"low": "l", "high": "h"}

    def test_list_shaped_top_logprobs(self):
        backend = self.make_backend()
        completion = {
            "logprobs": {
                "top_logprobs": [
                    [{"token": "A", "logprob": -0.4}, {"token": "B", "logprob": -1.4}]
                ]
            }
        }
        table = backend._top_logprobs(completion)
        assert table == {"A": -0.4, "B": -1.4}
        assert label_logprob(table, "A") == -0.4

    def test_evaluate_answers_every_question_and_sums_usage(self, monkeypatch):
        backend = self.make_backend()

        def fake_post(url, headers=None, json=None, timeout=None):
            prompt = (json or {}).get("prompt", "")
            table = (
                {" Yes": -0.3, " No": -1.7}
                if "Yes or No" in prompt
                else {" A": -0.1, "B": -2.0}
            )
            return type(
                "R",
                (),
                {
                    "status_code": 200,
                    "headers": {},
                    "raise_for_status": lambda self: None,
                    "json": lambda self: completions_response(table),
                },
            )()

        monkeypatch.setattr(requests, "post", fake_post)
        questions = {
            "phase_i0": {
                "type": "choice",
                "instructions": "pick",
                "criteria": {"ETWT": "a", "NTST": "b"},
            },
            "congested_i0": {"type": "noul", "instructions": "jam?"},
        }
        outcome = backend.evaluate({}, questions)
        assert set(outcome["answers"]) == {"phase_i0", "congested_i0"}
        assert outcome["model"] == "tev1-test"
        assert outcome["usage"]["input_tokens"] == 222
        assert outcome["usage"]["output_tokens"] == 2


# ---------------------------------------------------------------------- #
# Local server: HTTP round trip
# ---------------------------------------------------------------------- #


class TestSystemOneHttp:
    def start_server(self):
        server = make_server(MockBackend(), host="127.0.0.1", port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, f"http://127.0.0.1:{server.server_address[1]}"

    def test_health_and_models(self):
        server, base = self.start_server()
        try:
            health = requests.get(f"{base}/health", timeout=5).json()
            assert health["status"] == "ok"
            models = requests.get(f"{base}/v1/models", timeout=5).json()
            assert models["data"][0]["id"]
        finally:
            server.shutdown()
            server.server_close()

    def test_systemone_round_trip_matches_mock_jev(self):
        server, base = self.start_server()
        try:
            state = {
                "scenario": "test",
                "intersections": {
                    "i0": {
                        "lanes": {
                            "ET": {
                                "queue": 5,
                                "approaching_by_segment": [0, 0, 0],
                                "average_waiting_time_s": 10,
                            },
                            "WT": {
                                "queue": 4,
                                "approaching_by_segment": [0, 0, 0],
                                "average_waiting_time_s": 10,
                            },
                        }
                    }
                },
            }
            questions = {
                "phase_i0": {
                    "type": "choice",
                    "instructions": "pick the phase",
                    "criteria": {
                        "ETWT": "east-west through",
                        "NTST": "north-south through",
                        "ELWL": "left turns",
                        "NLSL": "left turns ns",
                    },
                },
                "congested_i0": {"type": "noul", "instructions": "jammed?"},
            }
            response = requests.post(
                f"{base}/v1/systemone", json={"state": state, "questions": questions},
                timeout=5,
            )
            response.raise_for_status()
            body = response.json()
            assert body["answers"]["phase_i0"]["choice"] == "ETWT"
            assert body["answers"]["congested_i0"]["noul"] == 0.25
            assert body["usage"]["input_tokens"] == 1024
        finally:
            server.shutdown()
            server.server_close()

    def test_bad_request_and_unknown_route(self):
        server, base = self.start_server()
        try:
            bad = requests.post(
                f"{base}/v1/systemone",
                data=b"not json",
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
            assert bad.status_code == 400
            missing = requests.post(f"{base}/v1/other", json={}, timeout=5)
            assert missing.status_code == 404
        finally:
            server.shutdown()
            server.server_close()


# ---------------------------------------------------------------------- #
# Agent modes
# ---------------------------------------------------------------------- #


class TestAgentModeQuestions:
    def test_jevlight_prompt_is_per_intersection(self):
        observation = make_observation()
        question = jevlight_phase_question(observation, 15)
        assert question["type"] == "choice"
        assert "signal control agent for intersection `$`" in (
            question["instructions"]
        )
        assert set(question["criteria"]) == {
            "ETWT", "NTST", "ELWL", "NLSL",
        }

    def test_cojevlight_prompt_uses_direct_neighbours(self):
        observation = make_observation()
        question = cojevlight_phase_question(
            observation, 15, "$"
        )
        assert "cooperating with its direct neighbours" in (
            question["instructions"]
        )
        assert "neighbours" in question["instructions"]


class TestAgentModeController:
    def make_controller(self, agent_mode, **kwargs):
        client = StubJevClient(
            results=[
                JevResult(answers={
                    "phase_i0": phase_answer("ETWT"),
                    "phase_i1": phase_answer("NTST"),
                }),
                JevResult(answers={"phase_i0": phase_answer("ETWT")}),
            ]
        )
        # Mirror the runner's per-agent defaults for the speculative Noul.
        kwargs.setdefault("speculative", False)
        controller = JevLightController(
            client,
            15,
            packaging="per_intersection",
            agent_mode=agent_mode,
            **kwargs,
        )
        return controller, client

    def test_jevlight_defaults_to_local_phase_only(self):
        controller, client = self.make_controller("jevlight")
        observations = [make_observation("i0"), make_observation("i1", empty=True)]
        controller.decide(observations, step=0)
        # Only the active intersection issues a request; its state is the
        # bare intersection with only its phase question.
        assert len(client.requests) == 1
        request = client.requests[0]
        assert "intersections" not in request["state"]
        assert set(request["questions"]) == {"phase_i0"}
        assert "signal control agent for intersection" in (
            request["questions"]["phase_i0"]["instructions"]
        )

    def test_cojevlight_sends_one_target_per_request(self):
        controller, client = self.make_controller("cojevlight")
        observations = [make_observation("i0"), make_observation("i1")]
        controller.decide(observations, step=0)
        assert len(client.requests) == 2
        for request in client.requests:
            state = request["state"]
            assert "intersections" not in state
            assert state["cooperative_view"] == {}
            assert set(request["questions"]) == {f"phase_{state['intersection_id']}"}
            assert "cooperating with its direct neighbours" in next(iter(request["questions"].values()))["instructions"]

    def test_trace_records_agent_mode(self):
        controller, _client = self.make_controller("jevlight")
        _actions, traces = controller.decide([make_observation("i0")], step=0)
        assert traces[0]["agent_mode"] == "jevlight"

    def test_pre_release_mode_names_stay_aliases(self):
        # ``llmlight`` / ``collmlight`` normalize to the renamed modes.
        for old_name, canonical in (
            ("llmlight", "jevlight"),
            ("collmlight", "cojevlight"),
        ):
            controller = JevLightController(
                StubJevClient(), 15, agent_mode=old_name
            )
            assert controller.agent_mode == canonical

    def test_invalid_agent_mode_rejected(self):
        try:
            JevLightController(StubJevClient(), 15, agent_mode="gptlight")
        except ValueError as exc:
            assert "agent_mode" in str(exc)
        else:
            raise AssertionError("invalid agent_mode accepted")


class TestMaxPressureController:
    def test_pressure_counts_queue_and_approaching(self):
        observation = make_observation(queue_by_phase={"ETWT": 3})
        observation.lanes["NT"].llmlight_cells = [1, 2, 0, 0]
        # ETWT: queues 3+3; NTST: approaching 3 on NT plus queue 0 on ST.
        assert phase_pressure(observation, "ETWT") == 6.0
        assert phase_pressure(observation, "NTST") == 3.0
        assert phase_pressure(observation, "ELWL") == 0.0

    def test_decide_picks_argmax_pressure_without_any_client(self):
        controller = MaxPressureController(15)
        observation = make_observation(
            intersection_id="i0", queue_by_phase={"ETWT": 5, "NTST": 2}
        )
        actions, traces = controller.decide([observation], step=0)
        # PHASES = ["WT_ET", "NT_ST", "WL_EL", "NL_SL"] -> ETWT is index 0.
        assert actions == {"i0": 0}
        trace = traces[0]
        assert trace["controller"] == "maxpressure"
        assert trace["signal"] == "ETWT"
        assert trace["pressures"]["ETWT"] == 10.0
        assert trace["fallback"] is None

    def test_decide_covers_every_intersection(self):
        controller = MaxPressureController(15)
        observations = [
            make_observation("i0", queue_by_phase={"NTST": 1}),
            make_observation("i1", queue_by_phase={"ELWL": 4}),
        ]
        actions, traces = controller.decide(observations, step=3)
        assert set(actions) == {"i0", "i1"}
        assert traces[0]["signal"] == "NTST"
        assert traces[1]["signal"] == "ELWL"
        assert all(t["step"] == 3 for t in traces)

    def test_usage_summary_reports_zero_requests(self):
        summary = MaxPressureController(15).usage_summary()
        assert summary["requests"] == 0
        assert summary["failures"] == 0
        assert summary["latency_mean_s"] is None
