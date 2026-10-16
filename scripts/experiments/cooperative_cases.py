"""Synthetic vehicle transitions and current traffic for Co input validation."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import sumolib

from jevlight.controller import JevLightController
from jevlight.jev_client import JevResult
from jevlight.observation import ALL_LANES, CONTROLLED_LANES, IntersectionObservation, LaneObservation, attach_lane_relations
from scripts.experiments.preparation_cases import FLOWS, PHASES, ROOT, SCENARIOS, source_hashes


class CaptureClient:
    """Capture construction only; responses are placeholders, never model results."""
    transport = "synthetic_capture"
    model = "placeholder_not_inference"
    def __init__(self):
        self.requests = []

    def evaluate(self, state, questions):
        self.requests.append(deepcopy({"state": state, "questions": questions}))
        return JevResult(answers={qid: ({"type": "noul", "noul": 0.5} if definition["type"] == "noul" else
            {"type": "choice", "choice": next(iter(definition["criteria"])),
             "probabilities": {key: 0.25 for key in definition["criteria"]}, "confidence": 0.25})
            for qid, definition in questions.items()})


def network_observations(path):
    """Build movement mappings from the same road orientations as SUMO runtime."""
    from utils.sumo_env import Intersection

    network = sumolib.net.readNet(str(path))
    observations = []
    for tls in network.getTrafficLights():
        node = network.getNode(tls.getID())
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
                for connection in physical.getOutgoing():
                    suffix = {"s": "T", "l": "L", "L": "L", "r": "R", "R": "R"}.get(connection.getDirection())
                    if suffix:
                        lane = lanes[side + suffix]
                        if physical.getID() not in lane.physical_lanes:
                            lane.physical_lanes.append(physical.getID())
                            lane.road_length = edge.getLength()
        observations.append(IntersectionObservation(tls.getID(), lanes, PHASES))
    attach_lane_relations(SimpleNamespace(sumo_net=network), observations)
    return observations


def warm_synthetic_relations(controller, templates):
    """Exercise the real learner with explicitly synthetic, physically valid trips.

    Topology is used only to construct this offline fixture. Runtime association
    learning receives vehicle snapshots and never reads these topology refs.
    """
    before = deepcopy(templates)
    after = deepcopy(templates)
    sources = {obs.intersection_id: obs for obs in before}
    destinations = {obs.intersection_id: obs for obs in after}
    for obs in before + after:
        for lane in obs.lanes.values():
            lane.vehicles = {}
    count = 0
    for obs in templates:
        for lane in CONTROLLED_LANES:
            for ref in obs.lane_relations.get(lane, {}).get("downstream", []):
                if ref["lane"] not in CONTROLLED_LANES:
                    continue
                vehicle = f"synthetic_transition_{count}"
                sources[obs.intersection_id].lanes[lane].vehicles[vehicle] = 0
                destinations[ref["intersection_id"]].lanes[ref["lane"]].vehicles[vehicle] = 0
                count += 1
    controller.trajectory_relations.observe(before, 0)
    controller.trajectory_relations.observe(after, 1)


def cases(speculative=False):
    for traffic_id, (_, directory, route) in FLOWS.items():
        templates = network_observations(next((ROOT / directory).glob("*.net.xml")))
        for index, scenario in enumerate(SCENARIOS):
            observations = []
            for template in templates:
                observation = deepcopy(template)
                queue = 0 if scenario == "empty" else 150 if scenario == "high_load" else 22
                for name in ALL_LANES:
                    lane_queue = queue
                    if scenario == "directional_backlog" and name not in ("NT", "ST"):
                        lane_queue = 0
                    if scenario == "turn_conflict" and name not in ("NL", "EL", "WT", "ST"):
                        lane_queue = 0
                    if scenario == "phase_change" and name not in ("NL", "SL"):
                        lane_queue = 0
                    cells = [0, 0, 0, 0] if scenario == "empty" else [100, 150, 200, 250] if scenario == "high_load" else [5, 3, 2, 1]
                    lane = observation.lanes[name]
                    lane.queue = lane_queue
                    lane.avg_wait = 0 if not lane_queue else 9999 if scenario == "long_wait" else 90
                    lane.llmlight_cells = cells
                    lane.occupancy = min(1, lane_queue / 50)
                observation.empty = scenario == "empty"
                observation.current_phase = PHASES[index % 4]
                observations.append(observation)
            client = CaptureClient()
            controller = JevLightController(client, 15, packaging="per_intersection", speculative=speculative,
                max_workers=1, agent_mode="cojevlight", cojev_history_steps=0)
            warm_synthetic_relations(controller, templates)
            controller.trajectory_relations.observe(observations, 2)
            controller._prepare_cooperative_states(observations)
            # Capture all targets separately, even empty ones for schema coverage.
            controller._decide_per_intersection(observations, 2)
            # Keep the largest serialized request per scenario as a representative;
            # this is not a claim about the longest model-specific token encoding.
            request = max(client.requests, key=lambda row: len(json.dumps(row)))
            yield {"case_id": f"cojevlight:{traffic_id}:{scenario}", "traffic_id": traffic_id,
                "agent_mode": "cojevlight", "scenario": scenario,
                "source": "synthetic vehicle transitions on real SUMO topology, then synthetic current traffic; largest JSON center-lane view; not real trajectories, inference or traffic results",
                "intersection_count": 1 + len({entry["intersection_id"]
                    for sides in request["state"]["cooperative_view"].values()
                    for entries in sides.values() for entry in entries}),
                "network_intersection_count": len(templates), "packaging": "per_intersection", "history_steps": 0,
                "traffic_file": f"{directory}/{route}", "request": request}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/decision_models/preparation/cooperative_cases")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "results"):
        parser.error("output must be inside results/")
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "prompt_cases.jsonl.tmp"
    with temporary.open("w") as stream:
        for case in cases():
            stream.write(json.dumps(case, ensure_ascii=False) + "\n")
    temporary.replace(output / "prompt_cases.jsonl")
    (output / "benchmark_source_hashes.json").write_text(json.dumps(source_hashes(), indent=2) + "\n")


if __name__ == "__main__":
    main()
