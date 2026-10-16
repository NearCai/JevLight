from types import SimpleNamespace
from jevlight.cooperation import TrajectoryRelations
from jevlight.observation import CONTROLLED_LANES, LaneObservation


def test_missing_neighbour_observation_remains_unknown():
    tracker = TrajectoryRelations(CONTROLLED_LANES)
    tracker.downstream_counts[("center", "NT")][("missing", "ST")] = 1
    tracker.downstream_counts[("center", "NT")][("known", "SL")] = 1
    center = SimpleNamespace(intersection_id="center", lanes={"NT": LaneObservation(occupancy=.2)})
    states = {"known": {"current_phase": "NTST", "lanes": {}}}
    assert tracker.view(center, states, include_empty_neighbours=False) == {
        "NT": {"upstream": [], "downstream": []}}
