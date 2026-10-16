"""Intersection observations and local phase ranking.

Extracted from ChatLight's ``models/llmlight_controller.py`` and
``models/llmlight_prompts.py`` (the LLMLight/CoLLMLight migration), keeping
only what JevLight needs: the observation dataclasses, the SUMO snapshot
collector, and the exact CoLLMLight waiting-time-reduction ranking used for
empty junctions and fallbacks.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


CANONICAL_PHASES = ("ETWT", "NTST", "ELWL", "NLSL")
CONTROLLED_LANES = ("NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL")
ALL_LANES = (
    "NT", "NL", "NR", "ST", "SL", "SR",
    "ET", "EL", "ER", "WT", "WL", "WR",
)
LOCATION_SHORT = {"North": "N", "South": "S", "East": "E", "West": "W"}
MOVEMENT_SUFFIX = {"go_straight": "T", "turn_left": "L", "turn_right": "R"}
COLLMLIGHT_SEGMENTS = 10
COLLMLIGHT_CAR_SPACING = 9.0


def normalize_phase(phase: str) -> str:
    return re.sub(r"[^A-Z]", "", str(phase).upper())


def phase_lanes(phase: str) -> Tuple[str, ...]:
    normalized = normalize_phase(phase)
    return tuple(normalized[index : index + 2] for index in range(0, len(normalized), 2))


def canonical_phase_name(phase: str) -> str:
    """Normalize SUMO's pair order (``WT_ET``) to source order (``ETWT``)."""
    return "".join(sorted(phase_lanes(phase)))


@dataclass
class LaneObservation:
    """Both source observation views for one conceptual movement lane."""

    queue: float = 0.0
    approaching: float = 0.0
    avg_wait: float = 0.0
    occupancy: float = 0.0
    cells: List[float] = field(
        default_factory=lambda: [0.0] * COLLMLIGHT_SEGMENTS
    )
    queue_cells: List[float] = field(
        default_factory=lambda: [0.0] * COLLMLIGHT_SEGMENTS
    )
    llmlight_cells: List[float] = field(default_factory=lambda: [0.0] * 4)
    vehicles: Dict[str, int] = field(default_factory=dict)
    positions: Dict[str, float] = field(default_factory=dict)
    road_length: float = 0.0
    physical_lanes: List[str] = field(default_factory=list)


@dataclass
class IntersectionObservation:
    intersection_id: str
    lanes: Dict[str, LaneObservation]
    phases: List[str]
    neighbours: List[str] = field(default_factory=list)
    empty: bool = True
    current_phase: Optional[str] = None
    lane_relations: Dict[str, Dict[str, List[Dict[str, Any]]]] = field(default_factory=dict)
    topology_available: bool = False
    yellow_time: float = 5.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _movement_indices(road: Mapping[str, Any], movement: str) -> Sequence[int]:
    lanes = road.get("lanes", {})
    indices = lanes.get(movement)
    if indices is None:
        indices = road.get(movement)
    return indices or ()


def _llmlight_cell(lane_position: float, road_length: float) -> int:
    if lane_position <= road_length / 10:
        return 0
    if lane_position <= road_length / 3:
        return 1
    if lane_position <= road_length * 2 / 3:
        return 2
    return 3


def collect_observations(env: Any) -> List[IntersectionObservation]:
    """Build the two source observations from one cached SUMO snapshot.

    LLMLight uses four non-uniform distance cells, later presented as three
    segments. CoLLMLight uses ten uniform moving/queued cells, vehicle identity,
    per-lane position and road length. Movement occupancy aggregates storage
    across all its physical lanes using 9 metres per vehicle.
    """
    lane_queues = env.get_lane_waiting_vehicle_count()
    lane_vehicles = env.get_lane_vehicles()
    vehicle_distances = env.get_vehicle_distance()
    vehicle_speeds = env.get_vehicle_speed()
    waiting_times = env.waiting_vehicle_list
    observations: List[IntersectionObservation] = []
    intersection_ids = [intersection.inter_id for intersection in env.list_intersection]

    for intersection_index, intersection in enumerate(env.list_intersection):
        metadata = env.intersection_dict[intersection.inter_name]
        lanes = {lane: LaneObservation() for lane in ALL_LANES}
        for road_id, road in metadata["roads"].items():
            if road.get("type") != "incoming":
                continue
            direction = LOCATION_SHORT.get(road.get("location"))
            if direction is None:
                continue
            road_length = float(road.get("length", 0.0) or 0.0)
            for movement, suffix in MOVEMENT_SUFFIX.items():
                lane_name = f"{direction}{suffix}"
                physical_lanes = [
                    f"{road_id}_{index}" for index in _movement_indices(road, movement)
                ]
                lane_observation = lanes[lane_name]
                lane_observation.road_length = road_length
                lane_observation.queue = float(
                    sum(lane_queues.get(lane_id, 0) for lane_id in physical_lanes)
                )

                lane_observation.physical_lanes.extend(physical_lanes)
                physical_waits: List[float] = []
                for physical_lane in physical_lanes:
                    for vehicle_id in lane_vehicles.get(physical_lane, ()):
                        lane_position = road_length - float(
                            vehicle_distances.get(vehicle_id, 0.0)
                        )
                        segment_length = road_length / COLLMLIGHT_SEGMENTS if road_length else 1.0
                        segment = int(lane_position // segment_length)
                        lane_observation.vehicles[vehicle_id] = segment
                        lane_observation.positions[vehicle_id] = lane_position
                        speed = float(vehicle_speeds.get(vehicle_id, 0.0) or 0.0)
                        if vehicle_id in waiting_times:
                            physical_waits.append(float(waiting_times[vehicle_id]))
                        if 0 <= segment < COLLMLIGHT_SEGMENTS:
                            if speed > 0.1:
                                lane_observation.cells[segment] += 1
                            else:
                                lane_observation.queue_cells[segment] += 1
                        if speed > 0.1:
                            lane_observation.llmlight_cells[
                                _llmlight_cell(lane_position, road_length)
                            ] += 1
                lane_observation.avg_wait = (
                    sum(physical_waits) / len(physical_waits) if physical_waits else 0.0
                )

                lane_observation.approaching = float(sum(lane_observation.cells))
                capacity = (road_length // COLLMLIGHT_CAR_SPACING) * len(physical_lanes)
                lane_observation.occupancy = (
                    len(lane_observation.positions) / capacity if capacity else 0.0
                )

        adjacency = intersection.adjacency_info.get(
            "adjacency_row", [intersection_index]
        )
        neighbours = [
            intersection_ids[int(neighbour_index)]
            for neighbour_index in adjacency
            if 0 <= int(neighbour_index) < len(intersection_ids)
            and int(neighbour_index) != intersection_index
        ]
        phases = list(metadata.get("control_phases", CANONICAL_PHASES))
        current_phase = phases[0] if phases else None
        if hasattr(env, "get_current_local_action") and phases:
            current_action = int(env.get_current_local_action(intersection.inter_id))
            if 0 <= current_action < len(phases):
                current_phase = phases[current_action]
        empty = sum(lanes[lane].occupancy for lane in CONTROLLED_LANES) == 0
        observations.append(
            IntersectionObservation(
                intersection_id=intersection.inter_id,
                lanes=lanes,
                phases=phases,
                neighbours=neighbours,
                empty=empty,
                current_phase=current_phase,
                yellow_time=float(env.dic_traffic_env_conf.get("YELLOW_TIME", 5.0)),
            )
        )
    attach_lane_relations(env, observations)
    return observations


def attach_lane_relations(env: Any, observations: Sequence[IntersectionObservation]) -> None:
    """Resolve possible inter-junction movements from SUMO lane connections.

    A source movement's outgoing physical lane is another junction's incoming
    lane. That lane can serve several destination movements; these are possible
    destinations, not known routing proportions. Include right-turn feeders.
    No geometric-neighbour or compass-based guesses are used.
    """
    net = getattr(env, "sumo_net", None)
    if net is None:
        for observation in observations:
            observation.neighbours = []
            observation.lane_relations = {}
            observation.topology_available = False
        return
    destinations: Dict[str, List[Tuple[IntersectionObservation, str]]] = {}
    for observation in observations:
        observation.lane_relations = {
            lane: {"upstream": [], "downstream": []} for lane in ALL_LANES
        }
        observation.topology_available = True
        for lane_name, lane in observation.lanes.items():
            for physical_lane in lane.physical_lanes:
                destinations.setdefault(physical_lane, []).append((observation, lane_name))

    suffixes = {"s": "T", "l": "L", "L": "L", "r": "R", "R": "R"}
    for source in observations:
        for lane_name, lane in source.lanes.items():
            for physical_lane in lane.physical_lanes:
                for connection in net.getLane(physical_lane).getOutgoing():
                    if suffixes.get(connection.getDirection()) != lane_name[1]:
                        continue
                    receiving_lane = connection.getToLane().getID()
                    for destination, destination_movement in destinations.get(receiving_lane, ()):
                        if destination.intersection_id == source.intersection_id:
                            continue
                        downstream = {
                            "intersection_id": destination.intersection_id,
                            "lane": destination_movement,
                        }
                        upstream = {"intersection_id": source.intersection_id, "lane": lane_name}
                        if downstream not in source.lane_relations[lane_name]["downstream"]:
                            source.lane_relations[lane_name]["downstream"].append(downstream)
                        if upstream not in destination.lane_relations[destination_movement]["upstream"]:
                            destination.lane_relations[destination_movement]["upstream"].append(upstream)
    for observation in observations:
        observation.neighbours = sorted({
            ref["intersection_id"]
            for relations in observation.lane_relations.values()
            for refs in relations.values()
            for ref in refs
        })
        for relations in observation.lane_relations.values():
            for refs in relations.values():
                refs.sort(key=lambda ref: (ref["intersection_id"], ref["lane"]))


def _effective_range(
    lane: LaneObservation,
    phase_duration: float,
    car_speed: float = 11.11,
) -> int:
    if lane.road_length <= 0:
        return COLLMLIGHT_SEGMENTS - 1
    distance = car_speed * phase_duration
    segment_length = lane.road_length / COLLMLIGHT_SEGMENTS
    return min(int(distance // segment_length), COLLMLIGHT_SEGMENTS - 1)


def rank_phases(
    observation: IntersectionObservation,
    phase_duration: float,
    car_speed: float = 11.11,
) -> List[Tuple[str, float]]:
    """Exact CoLLMLight local waiting-time-reduction ranking."""
    lane_values: Dict[str, float] = {}
    for lane_name in CONTROLLED_LANES:
        lane = observation.lanes[lane_name]
        lane_range = _effective_range(lane, phase_duration, car_speed)
        moving = sum(lane.cells[: lane_range + 1])
        queued = sum(lane.queue_cells[: lane_range + 1])
        lane_values[lane_name] = (
            queued * lane.avg_wait
            + queued * phase_duration
            + moving * phase_duration
        )
    phase_values = {
        phase: lane_values[phase[:2]] + lane_values[phase[2:]]
        for phase in CANONICAL_PHASES
    }
    return sorted(phase_values.items(), key=lambda item: item[1], reverse=True)
