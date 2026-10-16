"""Vehicle-transition memory for Co mode; model views contain current state only."""

from copy import deepcopy

# CoLLMLight run_CoLLMlight.py / CoLLMLightAgent.py defaults (pinned in docs).
COMMUNICATION_THRESHOLD = 0.0
CONNECTION_THRESHOLD = 0.05


class TrajectoryRelations:
    """Match observed departures/arrivals by vehicle ID, as in CoLLMLight.

    Counts are cumulative within this controller/run. A relation qualifies when
    its fraction of the corresponding source/destination counts exceeds 5%.
    IDs and historical snapshots are internal; only current lane data is exposed.
    """

    def __init__(self, tracked_lanes, connection_threshold=CONNECTION_THRESHOLD):
        from collections import defaultdict, Counter
        self.tracked_lanes = frozenset(tracked_lanes)
        self.connection_threshold = connection_threshold
        self.previous = {}
        self.last_release = {}
        self.upstream_counts = defaultdict(Counter)
        self.downstream_counts = defaultdict(Counter)
        self.last_step = None
        self.matched_transitions = 0
        self.ambiguous_events = 0

    def observe(self, observations, step):
        if self.last_step is not None:
            if step < self.last_step:
                raise ValueError('Trajectory steps must increase; create a new controller for a new run')
            if step == self.last_step:
                return
        current = {}
        for observation in observations:
            for name, lane in observation.lanes.items():
                for vehicle in lane.vehicles:
                    current.setdefault(vehicle, set()).add((observation.intersection_id, name))
        # All departures precede arrivals, including transitions in one sample.
        for vehicle, before in self.previous.items():
            after = current.get(vehicle, set())
            if before == after:
                continue
            if len(before) == 1:
                self.last_release[vehicle] = next(iter(before))
            else:
                # Shared physical lanes do not reveal the vehicle's turn intent.
                self.last_release[vehicle] = None
                self.ambiguous_events += 1
        for vehicle, after in current.items():
            if after == self.previous.get(vehicle):
                continue
            origin = self.last_release.pop(vehicle, None)
            if len(after) != 1:
                self.ambiguous_events += 1
                continue
            destination = next(iter(after))
            if (origin is None or origin[0] == destination[0]
                    or origin[1] not in self.tracked_lanes
                    or destination[1] not in self.tracked_lanes):
                continue
            self.upstream_counts[destination][origin] += 1
            self.downstream_counts[origin][destination] += 1
            self.matched_transitions += 1
        self.previous = current
        self.last_step = step

    def related(self, center, side):
        counts = (self.upstream_counts if side == 'upstream' else self.downstream_counts).get(center, {})
        total = sum(counts.values())
        return sorted(key for key, count in counts.items()
                      if total and count / total > self.connection_threshold)

    def view(self, observation, states, *, include_empty_neighbours, occupancy_threshold=0.0):
        """A current upstream/downstream view rooted at each nonempty center lane."""
        view = {}
        for lane in sorted(self.tracked_lanes):
            central = observation.lanes.get(lane)
            if central is None or central.occupancy <= 0:
                continue
            view[lane] = {}
            for side in ('upstream', 'downstream'):
                entries = []
                for identity, name in self.related((observation.intersection_id, lane), side):
                    neighbour = states.get(identity)
                    if neighbour is None or name not in neighbour['lanes']:
                        continue  # Missing observations remain unknown, never invented zeros.
                    state = neighbour['lanes'][name]
                    if not include_empty_neighbours and state['occupancy'] <= occupancy_threshold:
                        continue
                    entries.append({'intersection_id': identity, 'lane': name,
                                    'current_phase': neighbour['current_phase'], 'state': deepcopy(state)})
                view[lane][side] = entries
        return view
