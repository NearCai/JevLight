"""Unit tests for the JevLight controller and client (no network, no SUMO)."""

from __future__ import annotations

import json

from typing import Any, List, Mapping

import pytest
from types import SimpleNamespace
from pathlib import Path

from jevlight.controller import (
    AGENT_MODE_DEFAULTS,
    CONGESTION_QUESTION_PREFIX,
    JevLightController,
    PHASE_QUESTION_PREFIX,
    congestion_question,
    jev_intersection_state,
    jevlight_phase_question,
    phase_criteria,
    cojevlight_phase_question,
)
from jevlight.jev_client import (
    JevClient,
    JevResult,
    resolve_jev_api_key,
)
from jevlight.observation import (
    CANONICAL_PHASES,
    IntersectionObservation,
    LaneObservation,
    rank_phases,
    ALL_LANES,
    attach_lane_relations,
    collect_observations,
)


PHASES = ["WT_ET", "NT_ST", "WL_EL", "NL_SL"]


def make_lane(queue=0, cells=(0, 0, 0, 0), waiting=0.0, occupancy=0.0):
    lane = LaneObservation()
    lane.queue = float(queue)
    # rank_phases reads the ten uniform queued/moving cells, not the scalar
    # queue; queued vehicles sit in segment 0 (nearest the junction).
    lane.queue_cells[0] = float(queue)
    lane.llmlight_cells = list(cells)
    lane.avg_wait = float(waiting)
    lane.occupancy = float(occupancy)
    return lane


def make_observation(
    intersection_id="i0",
    queue_by_phase=None,
    waiting=0.0,
    phases=PHASES,
    empty=False,
    current_phase="WT_ET",
):
    lanes = {lane: make_lane() for lane in (
        "NT", "NL", "NR", "ST", "SL", "SR",
        "ET", "EL", "ER", "WT", "WL", "WR",
    )}
    for phase, queue in (queue_by_phase or {}).items():
        for lane_name in (phase[:2], phase[2:]):
            lanes[lane_name].queue = float(queue)
            lanes[lane_name].queue_cells[0] = float(queue)
    for lane in lanes.values():
        lane.avg_wait = waiting
    return IntersectionObservation(
        intersection_id=intersection_id,
        lanes=lanes,
        phases=list(phases),
        neighbours=[],
        empty=empty,
        current_phase=current_phase,
    )


class StubJevClient(JevClient):
    """Captures requests and replays scripted results."""

    def __init__(self, results: List[JevResult] = None):
        super().__init__(mock=True)
        self.results = list(results or [])
        self.requests: List[Mapping[str, Any]] = []

    def evaluate(self, state, questions):
        self.requests.append({"state": state, "questions": dict(questions)})
        if self.results:
            return self.results.pop(0)
        return JevResult()


def phase_answer(signal, confidence=0.9, probabilities=None):
    return {
        "type": "choice",
        "choice": signal,
        "probabilities": probabilities
        or {name: 0.25 for name in CANONICAL_PHASES},
        "confidence": confidence,
    }


class TestStateAndQuestions:
    def test_intersection_state_shape(self):
        observation = make_observation(queue_by_phase={"ETWT": 4}, waiting=12.0)
        state = jev_intersection_state(observation, 15)
        assert state["intersection_id"] == "i0"
        assert state["phase_duration_s"] == 15
        assert state["current_phase"] == "ETWT"
        assert set(state["lanes"]) == {
            "NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL",
        }
        assert state["lanes"]["ET"]["queue"] == 4
        assert state["lanes"]["WT"]["queue"] == 4
        assert state["lanes"]["ET"]["average_waiting_time_s"] == 12.0
        assert state["lanes"]["ET"]["approaching_by_segment"] == [0, 0, 0]

    def test_segment_merging_matches_llmlight_cells(self):
        observation = make_observation()
        observation.lanes["NT"].llmlight_cells = [1, 2, 3, 4]
        state = jev_intersection_state(observation, 15)
        assert state["lanes"]["NT"]["approaching_by_segment"] == [1, 2, 7]

    def test_phase_criteria_are_canonical_and_described(self):
        observation = make_observation(phases=PHASES)
        criteria = phase_criteria(observation)
        assert set(criteria) == set(CANONICAL_PHASES)
        for description in criteria.values():
            assert "Activates" in description and "releases" in description

    def test_phase_question_wire_format(self):
        observation = make_observation()
        question = jevlight_phase_question(observation, 30, "intersections.i0")
        assert question["type"] == "choice"
        assert "intersections.i0" in question["instructions"]
        assert set(question["criteria"]) == set(CANONICAL_PHASES)

    def test_congestion_question_is_noul(self):
        question = congestion_question("intersections.i0")
        assert question["type"] == "noul"
        assert "intersections.i0" in question["instructions"]


class TestNetworkPackaging:
    def test_one_request_with_all_active_questions(self):
        observations = [
            make_observation("i0", queue_by_phase={"ETWT": 3}),
            make_observation("i1", queue_by_phase={"NTST": 5}, empty=True),
        ]
        client = StubJevClient(
            results=[
                JevResult(
                    answers={
                        f"{PHASE_QUESTION_PREFIX}i0": phase_answer("ETWT"),
                        f"{CONGESTION_QUESTION_PREFIX}i0": {
                            "type": "noul",
                            "noul": 0.4,
                        },
                    },
                    input_tokens=500,
                    output_tokens=10,
                )
            ]
        )
        controller = JevLightController(client, 15, packaging="network", agent_mode="jevlight", speculative=False)
        actions, traces = controller.decide(observations, step=0)

        assert len(client.requests) == 1
        request = client.requests[0]
        # The shared state carries every intersection, empty or not.
        assert set(request["state"]["intersections"]) == {"i0", "i1"}
        # Questions exist only for the active intersection.
        assert set(request["questions"]) == {
            f"{PHASE_QUESTION_PREFIX}i0",
        }
        # The chosen phase maps back onto the SUMO phase list index.
        assert actions == {"i0": 0, "i1": PHASES.index("NT_ST")}
        trace = next(t for t in traces if t["intersection"] == "i0")
        assert trace["signal"] == "ETWT"
        assert trace["confidence"] == 0.9
        assert trace["congestion_noul"] == 0.4
        assert trace["fallback"] is None
        empty_trace = next(t for t in traces if t["intersection"] == "i1")
        assert empty_trace["fallback"] == "empty_junction_local_ranking"
        assert controller.total_requests == 1
        assert controller.total_input_tokens == 500

    def test_empty_junction_uses_exact_local_ranking(self):
        observation = make_observation(
            "i0", queue_by_phase={"NLSL": 7}, empty=True
        )
        controller = JevLightController(StubJevClient(), 15)
        actions, _ = controller.decide([observation], step=0)
        expected = rank_phases(observation, 15)[0][0]
        assert actions["i0"] == PHASES.index("NL_SL")
        assert expected == "NLSL"

    def test_speculative_disabled_drops_noul(self):
        observation = make_observation("i0", queue_by_phase={"ETWT": 1})
        client = StubJevClient(
            results=[
                JevResult(
                    answers={f"{PHASE_QUESTION_PREFIX}i0": phase_answer("ETWT")}
                )
            ]
        )
        controller = JevLightController(
            client, 15, speculative=False
        )
        controller.decide([observation], step=0)
        assert set(client.requests[0]["questions"]) == {
            f"{PHASE_QUESTION_PREFIX}i0"
        }


class TestAnswerGating:
    def base_controller(self, client, **kwargs):
        return JevLightController(client, 15, **kwargs)

    def test_request_error_keeps_previous_signal(self):
        observation = make_observation("i0", queue_by_phase={"ETWT": 2})
        client = StubJevClient(results=[JevResult(error="boom")])
        controller = self.base_controller(client, fallback="previous")
        controller.previous_signals["i0"] = "NLSL"
        actions, traces = controller.decide([observation], step=1)
        assert actions["i0"] == PHASES.index("NL_SL")
        trace = traces[0]
        assert trace["fallback"] == "missing_answer"
        assert trace["signal"] == "NLSL"
        assert controller.total_failures == 1

    def test_rate_limited_result_counts_and_falls_back(self):
        observation = make_observation("i0", queue_by_phase={"ETWT": 2})
        client = StubJevClient(
            results=[
                JevResult(
                    error="jev_decide error: Too many requests.",
                    rate_limited=True,
                )
            ]
        )
        controller = self.base_controller(client)
        controller.previous_signals["i0"] = "NTST"
        actions, traces = controller.decide([observation], step=0)
        assert actions["i0"] == PHASES.index("NT_ST")
        assert traces[0]["fallback"] == "missing_answer"
        assert traces[0]["rate_limited"] is True
        assert controller.total_rate_limited == 1

    def test_invalid_choice_falls_back(self):
        observation = make_observation("i0")
        client = StubJevClient(
            results=[
                JevResult(
                    answers={
                        f"{PHASE_QUESTION_PREFIX}i0": phase_answer("ZZZZ")
                    }
                )
            ]
        )
        controller = self.base_controller(client)
        controller.previous_signals["i0"] = "NTST"
        actions, traces = controller.decide([observation], step=0)
        assert actions["i0"] == PHASES.index("NT_ST")
        assert traces[0]["fallback"] == "invalid_choice"

    def test_low_confidence_falls_back_to_ranking(self):
        observation = make_observation("i0", queue_by_phase={"ELWL": 6})
        client = StubJevClient(
            results=[
                JevResult(
                    answers={
                        f"{PHASE_QUESTION_PREFIX}i0": phase_answer(
                            "ETWT", confidence=0.35
                        )
                    }
                )
            ]
        )
        controller = self.base_controller(
            client, min_confidence=0.6, fallback="ranking"
        )
        actions, traces = controller.decide([observation], step=0)
        # The valid but unconfident ETWT answer is rejected; the local
        # ranking prefers the ELWL pressure.
        assert actions["i0"] == PHASES.index("WL_EL")
        assert traces[0]["fallback"] == "low_confidence"
        assert traces[0]["confidence"] == 0.35

    def test_confident_answer_within_threshold_is_used(self):
        observation = make_observation("i0", queue_by_phase={"ELWL": 6})
        client = StubJevClient(
            results=[
                JevResult(
                    answers={
                        f"{PHASE_QUESTION_PREFIX}i0": phase_answer(
                            "ETWT", confidence=0.61
                        )
                    }
                )
            ]
        )
        controller = self.base_controller(client, min_confidence=0.6)
        actions, traces = controller.decide([observation], step=0)
        assert actions["i0"] == PHASES.index("WT_ET")
        assert traces[0]["fallback"] is None


class TestPerIntersectionPackaging:
    def test_one_request_per_active_intersection(self):
        observations = [
            make_observation("i0", queue_by_phase={"ETWT": 2}),
            make_observation("i1", queue_by_phase={"NTST": 3}),
        ]
        client = StubJevClient(
            results=[
                JevResult(answers={f"{PHASE_QUESTION_PREFIX}i0": phase_answer("ETWT")}),
                JevResult(answers={f"{PHASE_QUESTION_PREFIX}i1": phase_answer("NTST")}),
            ]
        )
        controller = JevLightController(
            client, 15, packaging="per_intersection", speculative=False
        )
        actions, traces = controller.decide(observations, step=0)
        assert len(client.requests) == 2
        per_request_ids = [
            set(request["questions"]) for request in client.requests
        ]
        assert per_request_ids == [
            {f"{PHASE_QUESTION_PREFIX}i0"},
            {f"{PHASE_QUESTION_PREFIX}i1"},
        ]
        for request in client.requests:
            state = request["state"]
            assert "intersections" not in state
            assert "lanes" in state
        assert actions == {"i0": 0, "i1": 1}
        assert len(traces) == 2
        assert controller.total_requests == 2


class TestMockTransport:
    def test_mock_uses_state_pressure(self):
        client = JevClient(mock=True)
        state = {
            "intersections": {
                "i0": {
                    "lanes": {
                        "NT": {"queue": 9, "approaching_by_segment": [0, 0, 0],
                               "average_waiting_time_s": 0, "occupancy": 0},
                        "ST": {"queue": 9, "approaching_by_segment": [0, 0, 0],
                               "average_waiting_time_s": 0, "occupancy": 0},
                    }
                }
            }
        }
        questions = {f"{PHASE_QUESTION_PREFIX}i0": jevlight_phase_question(
            make_observation(), 15, "intersections.i0"
        )}
        result = client.evaluate(state, questions)
        answer = result.answers[f"{PHASE_QUESTION_PREFIX}i0"]
        assert answer["choice"] == "NTST"
        assert answer["probabilities"]["NTST"] == 0.25


class TestMcpParsing:
    """Parsing logic for the community jev_decide tool payload."""

    @staticmethod
    def decide_result(text: str, is_error: bool = False) -> dict:
        return {
            "content": [{"type": "text", "text": text}],
            "isError": is_error,
        }

    def make_client(self):
        return JevClient(
            transport="mcp", api_key="jev_test", mock=False
        )

    def test_official_shape_payload(self):
        client = self.make_client()
        payload = (
            '{"model": "jev-1.13.0", "answers": {"phase": '
            '{"type": "choice", "choice": "ETWT", '
            '"probabilities": {"ETWT": 0.8}, "confidence": 0.7}}, '
            '"usage": {"input_tokens": 42, "output_tokens": 8}}'
        )
        result = client._parse_decide_result(
            self.decide_result(payload), started=0.0
        )
        assert result.model == "jev-1.13.0"
        assert result.input_tokens == 42
        assert result.answers["phase"]["choice"] == "ETWT"
        assert result.error is None

    def test_bare_answers_payload(self):
        client = self.make_client()
        payload = (
            '{"phase": {"type": "choice", "choice": "NLSL", '
            '"confidence": 0.66}, "congested": {"type": "noul", "noul": 0.9}}'
        )
        result = client._parse_decide_result(
            self.decide_result(payload), started=0.0
        )
        assert result.answers["phase"]["choice"] == "NLSL"
        assert result.answers["congested"]["noul"] == 0.9

    def test_unparsable_payload_reports_error(self):
        client = self.make_client()
        result = client._parse_decide_result(
            self.decide_result("not json"), started=0.0
        )
        assert result.error.startswith("unparsable jev_decide content")

    def test_rate_limit_marker_detection(self):
        assert JevClient._is_rate_limited("Too many requests. Please try again later.")
        assert JevClient._is_rate_limited("HTTP 429: rate limit exceeded")
        assert not JevClient._is_rate_limited("jev_decide error: bad state")


class TestApiKeyResolution:
    def test_explicit_wins(self, monkeypatch):
        monkeypatch.setenv("JEV_API_KEY", "env-key")
        assert resolve_jev_api_key("explicit-key") == "explicit-key"

    def test_env_order(self, monkeypatch):
        monkeypatch.setenv("JEV_API_KEY", "community-key")
        assert resolve_jev_api_key() == "community-key"
        monkeypatch.delenv("JEV_API_KEY")
        monkeypatch.setenv("TYPESAFE_API_KEY", "official-key")
        assert resolve_jev_api_key() == "official-key"

    def test_missing_returns_none(self, monkeypatch):
        monkeypatch.delenv("JEV_API_KEY", raising=False)
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        assert resolve_jev_api_key() is None

    def test_client_requires_key_without_mock(self, monkeypatch):
        monkeypatch.delenv("JEV_API_KEY", raising=False)
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        with pytest.raises(ValueError):
            JevClient()


class TestConfigValidation:
    def test_invalid_packaging_rejected(self):
        with pytest.raises(ValueError):
            JevLightController(StubJevClient(), 15, packaging="bogus")

    def test_invalid_fallback_rejected(self):
        with pytest.raises(ValueError):
            JevLightController(StubJevClient(), 15, fallback="bogus")

    def test_confidence_bounds_enforced(self):
        with pytest.raises(ValueError):
            JevLightController(StubJevClient(), 15, min_confidence=1.5)

    def test_invalid_transport_rejected(self):
        with pytest.raises(ValueError):
            JevClient(transport="carrier-pigeon", api_key="k")


class FakeLaneNetwork:
    """SUMO's lane connection surface with independently specified routes."""

    def __init__(self, routes):
        self.routes = routes

    def getLane(self, lane_id):
        connections = [
            SimpleNamespace(
                getDirection=lambda direction=direction: direction,
                getToLane=lambda target=target: SimpleNamespace(getID=lambda: target),
            )
            for direction, target in self.routes.get(lane_id, [])
        ]
        return SimpleNamespace(getOutgoing=lambda: connections)


def connected_observations():
    a, b, c = [make_observation(name) for name in ("a", "b", "c")]
    a.neighbours = ["c"]  # Geometry would incorrectly suggest c.
    a.lanes["ET"].physical_lanes = ["a_in_0"]
    a.lanes["EL"].physical_lanes = ["a_in_0"]  # Shared physical lane.
    b.lanes["WT"].physical_lanes = ["b_in_0"]
    b.lanes["WL"].physical_lanes = ["b_in_0"]
    c.lanes["WR"].physical_lanes = ["c_in_0"]
    net = FakeLaneNetwork({
        "a_in_0": [("s", "b_in_0"), ("s", "b_in_0"), ("l", "external_0")],
        "c_in_0": [("r", "b_in_0")],
    })
    attach_lane_relations(SimpleNamespace(sumo_net=net), [a, b, c])
    return a, b, c


class TestCooperativeInput:
    def test_topology_preserves_branching_turns_boundaries_and_reciprocity(self):
        a, b, c = connected_observations()
        assert a.neighbours == ["b"]
        assert a.topology_available
        assert a.lane_relations["ET"]["downstream"] == [
            {"intersection_id": "b", "lane": "WL"},
            {"intersection_id": "b", "lane": "WT"},
        ]
        # A left-turn has an external destination, never its straight receiver.
        assert a.lane_relations["EL"]["downstream"] == []
        assert b.lane_relations["WT"]["upstream"] == [
            {"intersection_id": "a", "lane": "ET"},
            {"intersection_id": "c", "lane": "WR"},
        ]
        assert c.lane_relations["WR"]["downstream"] == a.lane_relations["ET"]["downstream"]

    def test_missing_network_is_unknown_not_empty_known_topology(self):
        obs = make_observation()
        attach_lane_relations(SimpleNamespace(), [obs])
        assert not obs.topology_available
        assert obs.lane_relations == {}

    def test_cold_start_does_not_invent_cooperation_from_topology(self):
        observations = connected_observations()
        client = StubJevClient()
        controller = JevLightController(client, 15)
        controller.decide(observations, 0)
        assert controller.packaging == "per_intersection"
        assert len(client.requests) == 3
        for request in client.requests:
            state = request["state"]
            target = state["intersection_id"]
            assert state["cooperative_view"] == {}
            assert state["cooperation_source"] == "observed_vehicle_transitions"
            for field in ("lane_relations", "topology_available", "related_intersections", "neighbours", "intersections"):
                assert field not in state
            assert set(request["questions"]) == {f"phase_{target}"}
            assert set(request["questions"][f"phase_{target}"]["criteria"]) == set(CANONICAL_PHASES)
            assert set(state["lanes"]) == set(ALL_LANES)

    @pytest.mark.parametrize("mode", ["cojevlight", "collmlight"])
    def test_network_packaging_is_rejected_for_cooperative_mode(self, mode):
        with pytest.raises(ValueError, match="requires per_intersection"):
            JevLightController(StubJevClient(), 15, agent_mode=mode, packaging="network")

    def test_learned_view_uses_latest_state_filters_empty_and_excludes_history(self):
        a, b, c = connected_observations()
        a.lanes["ET"].occupancy = b.lanes["WT"].occupancy = .2
        a.lanes["ET"].vehicles = {"vehicle_fixture_927": 0}
        client = StubJevClient()
        controller = JevLightController(client, 15)
        controller.decide([a, b, c], 0)
        a.lanes["ET"].vehicles = {}
        b.lanes["WT"].vehicles = {"vehicle_fixture_927": 0}
        b.lanes["WT"].queue = 17
        controller.decide([a, b, c], 1)
        latest = client.requests[-3:]
        state = next(r["state"] for r in latest if r["state"]["intersection_id"] == "a")
        entry, = state["cooperative_view"]["ET"]["downstream"]
        assert (entry["intersection_id"], entry["lane"]) == ("b", "WT")
        assert entry["state"]["queue"] == 17
        assert set(entry) == {"intersection_id", "lane", "current_phase", "state"}
        assert state["cooperative_view"]["ET"]["upstream"] == []
        assert controller.trajectory_relations.matched_transitions == 1
        for request in latest:
            for field in ("history", "sampled_transition_count", "selected_phase", "vehicle_fixture_927"):
                assert field not in json.dumps(request)
        fresh = StubJevClient()
        JevLightController(fresh, 15).decide([a, b, c], 1)
        assert fresh.requests[0]["state"]["cooperative_view"]["ET"]["downstream"] == []
        b.lanes["WT"].occupancy = 0
        b.empty = True
        controller.decide([a, b, c], 2)
        state = next(r["state"] for r in client.requests[-2:] if r["state"]["intersection_id"] == "a")
        assert state["cooperative_view"]["ET"]["downstream"] == []
        # Filtering current state does not erase the learned association.
        assert controller.trajectory_relations.related(("a", "ET"), "downstream") == [("b", "WT")]

    @pytest.mark.parametrize("steps", [-1, 1, 5])
    def test_history_cannot_be_reenabled(self, steps):
        with pytest.raises(ValueError, match="current state only"):
            JevLightController(StubJevClient(), 15, cojev_history_steps=steps)

    def test_zero_history_and_local_mode_keep_requests_local(self):
        obs = make_observation()
        client = StubJevClient()
        controller = JevLightController(client, 15, speculative=False, cojev_history_steps=0)
        controller.decide([obs], 0)
        controller.decide([obs], 1)
        assert "history" not in client.requests[-1]["state"]
        local_client = StubJevClient()
        JevLightController(local_client, 15, agent_mode="jevlight", packaging="per_intersection", speculative=False).decide([obs], 0)
        assert "lane_relations" not in local_client.requests[0]["state"]
        assert set(local_client.requests[0]["state"]["lanes"]) == set(obs.lanes) - {"NR", "SR", "ER", "WR"}

    def test_direction_definitions_and_switch_cost_are_consistent(self):
        obs = make_observation()
        criteria = phase_criteria(obs)
        for phase, descriptions in {
            "ETWT": ("ET = through lanes arriving from the east", "WT = through lanes arriving from the west"),
            "NTST": ("NT = through lanes arriving from the north", "ST = through lanes arriving from the south"),
            "ELWL": ("EL = left-turn lanes arriving from the east", "WL = left-turn lanes arriving from the west"),
        }.items():
            for description in descriptions:
                assert description in criteria[phase]
        state = jev_intersection_state(obs, 15)
        assert (state["yellow_time_s"], state["green_time_if_switch_s"]) == (5, 10)
        question = cojevlight_phase_question(obs, 15, "$")
        for field in ("cooperative_view", "latest current state", "yellow_time_s", "occupancy"):
            assert field in question["instructions"]
        assert "no reasoning text" in question["instructions"]

    def test_movement_statistics_cover_all_physical_lanes(self):
        intersection = SimpleNamespace(inter_id="a", inter_name="a", adjacency_info={})
        env = SimpleNamespace(
            list_intersection=[intersection],
            intersection_dict={"a": {"roads": {"in": {
                "type": "incoming", "location": "North", "length": 90,
                "lanes": {"go_straight": [0, 1]},
            }}}},
            dic_traffic_env_conf={"YELLOW_TIME": 4},
            waiting_vehicle_list={"x": 10, "y": 30},
            get_lane_waiting_vehicle_count=lambda: {"in_0": 1, "in_1": 1},
            get_lane_vehicles=lambda: {"in_0": ["x"], "in_1": ["y"]},
            get_vehicle_distance=lambda: {"x": 88, "y": 87},
            get_vehicle_speed=lambda: {"x": 0, "y": 0},
        )
        obs = collect_observations(env)[0]
        assert obs.lanes["NT"].queue == 2
        assert obs.lanes["NT"].avg_wait == 20
        assert obs.lanes["NT"].occupancy == .1  # 2 / (2 lanes * 10 vehicles).
        assert obs.yellow_time == 4
        assert obs.lanes["NT"].physical_lanes == ["in_0", "in_1"]


@pytest.mark.parametrize("relative_path, junction_count", [
    ("data/Jinan/3_4/roadnet_3_4.net.xml", 12),
    ("data/Hangzhou/4_4/roadnet_4_4.net.xml", 16),
    ("data/NewYork/28_7/roadnet_28_7.net.xml", 196),
])
def test_real_sumo_four_phase_directions_and_cooperative_connections(relative_path, junction_count):
    """Verify named green phases against lane geometry, without starting SUMO."""
    import xml.etree.ElementTree as ET
    import sumolib
    from utils.sumo_env import Intersection

    path = Path(__file__).resolve().parents[1] / relative_path
    net = sumolib.net.readNet(str(path))
    observations = []
    lane_codes = {}
    for tls in net.getTrafficLights():
        node = net.getNode(tls.getID())
        intersection = Intersection.__new__(Intersection)
        x, y = node.getCoord()
        intersection.point = {"x": x, "y": y}
        intersection.incoming_roads = {edge.getID(): edge for edge in node.getIncoming()}
        intersection.outgoing_roads = {edge.getID(): edge for edge in node.getOutgoing()}
        intersection.road_id_2_orient = {}
        intersection._calculate_road_orientations()
        lanes = {name: LaneObservation() for name in ALL_LANES}
        for edge in node.getIncoming():
            side = intersection.road_id_2_orient["incoming"][edge.getID()]
            for physical in edge.getLanes():
                lane_codes[physical.getID()] = side
                for connection in physical.getOutgoing():
                    suffix = {"s": "T", "l": "L", "L": "L", "r": "R", "R": "R"}.get(connection.getDirection())
                    if suffix:
                        lane = lanes[side + suffix]
                        if physical.getID() not in lane.physical_lanes:
                            lane.physical_lanes.append(physical.getID())
                            lane.road_length = edge.getLength()
        observations.append(IntersectionObservation(tls.getID(), lanes, list(CANONICAL_PHASES)))

    # XML's actual green connection indices are independent of phase-name parsing.
    root = ET.parse(path).getroot()
    controlled = {}
    for connection in root.findall("connection"):
        suffix = {"s": "T", "l": "L", "L": "L"}.get(connection.get("dir"))
        if connection.get("tl") and suffix:
            physical = f'{connection.get("from")}_{connection.get("fromLane")}'
            key = (connection.get("tl"), int(connection.get("linkIndex")))
            controlled.setdefault(key, set()).add(lane_codes[physical] + suffix)
    checked = 0
    for logic in root.findall("tlLogic"):
        for phase in logic.findall("phase"):
            name = phase.get("name")
            if name not in CANONICAL_PHASES:
                continue
            actual = set()
            for index, signal in enumerate(phase.get("state")):
                if signal in "Gg":
                    actual.update(controlled.get((logic.get("id"), index), set()))
            assert actual == {name[:2], name[2:]}
            checked += 1
    assert checked == junction_count * 4

    attach_lane_relations(SimpleNamespace(sumo_net=net), observations)
    assert len(observations) == junction_count
    by_id = {obs.intersection_id: obs for obs in observations}
    for obs in observations:
        assert obs.topology_available and obs.neighbours
        for lane, sides in obs.lane_relations.items():
            for ref in sides["downstream"]:
                receiver = by_id[ref["intersection_id"]]
                assert {"intersection_id": obs.intersection_id, "lane": lane} in receiver.lane_relations[ref["lane"]]["upstream"]
                # Every declared route must have an actual lane connection of
                # the source movement's turn type into the receiver group.
                assert any(
                    connection.getToLane().getID() in receiver.lanes[ref["lane"]].physical_lanes
                    and {"s": "T", "l": "L", "L": "L", "r": "R", "R": "R"}.get(connection.getDirection()) == lane[1]
                    for physical in obs.lanes[lane].physical_lanes
                    for connection in net.getLane(physical).getOutgoing()
                )

    for obs in observations:
        obs.empty = False
    client = StubJevClient()
    JevLightController(client, 15).decide(observations, 0)
    assert len(client.requests) == junction_count
    for request in client.requests:
        state = request["state"]
        assert "intersections" not in state
        assert len(request["questions"]) == 1
        assert state["cooperative_view"] == {}
        assert "lane_relations" not in state



@pytest.mark.parametrize("agent_mode", ["jevlight", "cojevlight"])
@pytest.mark.parametrize("congestion_probability", [0.0, 1.0])
def test_optional_congestion_shares_request_and_only_phase_answer_controls_signal(agent_mode, congestion_probability):
    observation = make_observation("i0", queue_by_phase={"ETWT": 1})
    client = StubJevClient(results=[JevResult(answers={
        "phase_i0": phase_answer("ETWT"),
        "congested_i0": {"type": "noul", "noul": congestion_probability},
    })])
    controller = JevLightController(client, 15, agent_mode=agent_mode,
                                    **{**AGENT_MODE_DEFAULTS[agent_mode], "speculative": True})
    actions, traces = controller.decide([observation], step=0)
    assert len(client.requests) == 1
    assert set(client.requests[0]["questions"]) == {"phase_i0", "congested_i0"}
    assert actions["i0"] == PHASES.index("WT_ET")
    assert traces[0]["congestion_noul"] == congestion_probability
    assert traces[0]["fallback"] is None


@pytest.mark.parametrize("agent_mode", ["jevlight", "cojevlight"])
def test_default_modes_send_only_phase_question(agent_mode):
    observation = make_observation("i0", queue_by_phase={"ETWT": 1})
    client = StubJevClient(results=[JevResult(answers={"phase_i0": phase_answer("ETWT")})])
    controller = JevLightController(client, 15, agent_mode=agent_mode,
                                    **AGENT_MODE_DEFAULTS[agent_mode])
    actions, traces = controller.decide([observation], step=0)
    assert set(client.requests[0]["questions"]) == {"phase_i0"}
    assert actions["i0"] == PHASES.index("WT_ET")
    assert traces[0]["congestion_noul"] is None
