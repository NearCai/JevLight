"""JevLight controller: signal control decisions via the Jev decision model.

Jev is not a chat model.  One request sends a ``state`` plus a map of typed
questions and receives one typed answer per question, evaluated in parallel
against the same state.

JevLight maps signal control onto that interface:

- The **state** is the per-intersection observation in LLMLight semantics
  (queue, three-segment approaching vehicles, average waiting time, occupancy)
  rendered as JSON. Cooperative requests include the target and only its
  direct upstream/downstream neighbours, all at the current decision step.
- The **question** for each active intersection is one ``Choice`` whose
  criteria are the signal phases available at that intersection, plus one
  optional ``Noul`` congestion judgment for logging in both modes.
  Both questions share the latest current state in one request.
- The answer's ``choice`` is the phase to activate.  ``probabilities`` and
  ``confidence`` are recorded per decision and can gate a fallback.

Fallbacks follow the migrated ChatLight source conventions: empty junctions
never call the API and use the exact CoLLMLight local ranking; failed or
unparsable requests keep the previous signal (LLMLight/CoLLMLight behavior)
or fall back to the local ranking; answers below ``min_confidence`` take the
same fallback.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from jevlight.jev_client import JevClient, JevResult
from jevlight.cooperation import COMMUNICATION_THRESHOLD, TrajectoryRelations
from jevlight.observation import (
    ALL_LANES,
    CONTROLLED_LANES,
    IntersectionObservation,
    canonical_phase_name,
    rank_phases,
)


JEV_PACKAGINGS = ("network", "per_intersection")
JEV_FALLBACKS = ("previous", "ranking")
# Agent modes: ``jevlight`` is a per-intersection agent and ``cojevlight``
# adds direct upstream/downstream context — LLMLight's and CoLLMLight's
# request patterns and prompt wording on the same Jev wire format. The
# pre-release names ``llmlight`` / ``collmlight`` stay accepted as aliases.
AGENT_MODES = ("jevlight", "cojevlight")
AGENT_MODE_ALIASES = {"llmlight": "jevlight", "collmlight": "cojevlight"}
PHASE_QUESTION_PREFIX = "phase_"
CONGESTION_QUESTION_PREFIX = "congested_"
PROMPT_VERSION = "incoming-side-shared-instructions-v7"
COOPERATIVE_PROMPT_VERSION = "incoming-side-trajectory-lanes-v6"

# Packaging and speculative-question defaults per agent mode: jevlight is a
# per-intersection agent; both modes disable the optional congestion judgment by default.
# cojevlight decides
# with its own current state and direct neighbours per request.
AGENT_MODE_DEFAULTS = {
    "jevlight": {"packaging": "per_intersection", "speculative": False},
    "cojevlight": {"packaging": "per_intersection", "speculative": False},
}


# ---------------------------------------------------------------------- #
# State and question construction
# ---------------------------------------------------------------------- #


def jev_intersection_state(
    observation: IntersectionObservation,
    phase_duration: float,
    include_right_turns: bool = False,
) -> Dict[str, Any]:
    """Render one intersection observation as a Jev-friendly JSON state.

    The lane semantics follow LLMLight: ``queue`` counts vehicles already
    waiting at the stop line, ``approaching_by_segment`` counts moving
    vehicles in three distance segments (segment 1 nearest the junction).
    """
    lanes: Dict[str, Any] = {}
    for lane_name in (ALL_LANES if include_right_turns else CONTROLLED_LANES):
        lane = observation.lanes[lane_name]
        cells = lane.llmlight_cells
        lanes[lane_name] = {
            "queue": int(lane.queue),
            "approaching_by_segment": [
                int(cells[0]),
                int(cells[1]),
                int(cells[2] + cells[3]),
            ],
            "average_waiting_time_s": round(float(lane.avg_wait), 1),
            "occupancy": round(float(lane.occupancy), 3),
            "road_length_m": round(float(lane.road_length), 1),
        }
        if include_right_turns:
            physical = set(lane.physical_lanes)
            lanes[lane_name]["shared_movements"] = [
                name for name, other in observation.lanes.items()
                if name != lane_name and physical.intersection(other.physical_lanes)
            ]
    current_phase = (
        canonical_phase_name(observation.current_phase)
        if observation.current_phase
        else None
    )
    duration = (
        int(phase_duration) if float(phase_duration).is_integer() else phase_duration
    )
    return {
        "prompt_version": PROMPT_VERSION,
        "intersection_id": observation.intersection_id,
        "phase_duration_s": duration,
        "current_phase": current_phase,
        "yellow_time_s": observation.yellow_time,
        "green_time_if_switch_s": max(0.0, duration - observation.yellow_time),
        "available_phases": [canonical_phase_name(phase) for phase in observation.phases],
        "lanes": lanes,
        "neighbours": list(observation.neighbours),
    }


def phase_criteria(
    observation: IntersectionObservation,
) -> Dict[str, str]:
    """Candidate-specific movement meanings; shared policy belongs in instructions."""
    location_words = {
        "N": "arriving from the north",
        "S": "arriving from the south",
        "E": "arriving from the east",
        "W": "arriving from the west",
    }
    movement_words = {"T": "through", "L": "left-turn"}
    criteria: Dict[str, str] = {}
    for phase in observation.phases:
        canonical = canonical_phase_name(phase)
        if canonical in criteria:
            continue
        parts = []
        for index in range(0, len(canonical), 2):
            code = canonical[index : index + 2]
            parts.append(
                f"{code} = {movement_words.get(code[1], code[1])} lanes "
                f"{location_words.get(code[0], code[0])}"
            )
        criteria[canonical] = (
            f"Activates {canonical[:2]} and {canonical[2:]}; "
            f"releases {', '.join(parts)}."
        )
    return criteria


def jevlight_phase_question(
    observation: IntersectionObservation,
    phase_duration: float,
    state_path: str = "$",
) -> Dict[str, Any]:
    """JevLight prompt: one agent per intersection, local view only.

    Wording follows LLMLight's original per-intersection signal-control
    prompt (the intersection's own lanes and current pressure, no network
    context); the wire format stays one Jev ``choice`` question.
    """
    criteria = phase_criteria(observation)
    duration = (
        int(phase_duration) if float(phase_duration).is_integer() else phase_duration
    )
    instructions = (
        f"You are the signal control agent for intersection `{state_path}`. "
        f"The intersection state at `{state_path}` lists every "
        "signal-controlled lane with queued vehicles, approaching vehicles "
        "by distance segment (segment 1 is nearest the junction), and "
        "average waiting time. Select the signal phase to activate for the "
        f"next {duration} seconds to minimize the intersection's queue "
        "length and vehicle waiting time. Movements whose lanes hold "
        "heavier queues, more approaching vehicles, or longer waits need "
        "the green signal sooner; the current phase's lanes have already "
        "been moving. When comparing phases, prefer the phase whose lanes "
        "hold the largest combined queue, approaching, and waiting-time pressure."
    )
    return {
        "type": "choice",
        "instructions": instructions,
        "criteria": criteria,
    }


def cojevlight_phase_question(
    observation: IntersectionObservation,
    phase_duration: float,
    state_path: str,
) -> Dict[str, Any]:
    """Choose one target phase using its current state and direct neighbours."""
    criteria = phase_criteria(observation)
    duration = (
        int(phase_duration) if float(phase_duration).is_integer() else phase_duration
    )
    instructions = (
        "You are the signal control agent for one intersection, cooperating with its direct neighbours. "
        f"For the target at `{state_path}` ($ is the state root), choose one "
        f"provided phase for the next {duration}-second decision interval. "
        "Lane letters N/S/E/W denote the incoming side, not travel heading; "
        "T/L/R mean through/left/right. NT travels north to south, ST south "
        "to north, ET east to west, WT west to east. A phase releases only "
        "its named movements. Right turns remain local context only. "
        "cooperative_view is organized by each nonempty central T/L lane, with "
        "upstream and downstream lists of (intersection_id, lane, current_phase, state). "
        "Associations are learned by matching vehicle IDs leaving one movement "
        "and arriving at another, retaining cumulative shares strictly above 5%. "
        "Only straight/left movements are shared, and neighbour occupancy must "
        "be greater than 0. No whole-neighbour topology or rankings are included. "
        "An empty or missing list can mean unobserved connectivity or filtered "
        "lanes; it does not establish free downstream space. These sampled "
        "associations do not guarantee destinations of currently waiting vehicles. "
        "Assess local queues and waiting times, then connected upstream demand "
        "and downstream receiving space using occupancy (vehicles divided by "
        "physical-lane storage capacity; 1 means estimated full, values may exceed 1). "
        "shared_movements identifies overlapping physical lanes: do not add "
        "their counts as if they were disjoint queues. "
        "Avoid releases into nearly full downstream lanes; favor clearing receivers "
        "that unblock queued upstream traffic when downstream space permits. "
        "Do not let long waits grow without considering a feasible release. "
        "Moving counts in approaching_by_segment cover 0-10%, 10-33%, 33-100% "
        "of road_length_m back from the stop line; nearer arrivals matter sooner. "
        "Use only the latest current state of the target and connected junctions. "
        "Current state and learned associations do not provide other agents' next actions; do not "
        "assume joint optimization or a promised green wave. "
        "Compare local relief, spillback and upstream relief for each candidate "
        "using the same current-state decision procedure. "
        "local_phase_priority estimates local waiting-time relief in seconds "
        "using ranking_assumed_speed_m_s; it ignores downstream blockage and "
        "switching loss, so it is a reference ranking, not a mandatory answer. "
        "Keeping current_phase allows the full interval of green; switching "
        "first consumes yellow_time_s, leaving green_time_if_switch_s. "
        "Current green does not imply a cleared queue. Select exactly one "
        "provided option using the service's choice format; no reasoning text "
        "or extra fields."
    )
    return {
        "type": "choice",
        "instructions": instructions,
        "criteria": criteria,
    }


def congestion_question(state_path: str) -> Dict[str, Any]:
    """Speculative congestion Noul, logged for analysis (near-zero cost)."""
    return {
        "type": "noul",
        "instructions": (
            f"Is intersection `{state_path}` currently under heavy congestion, "
            "meaning a large share of its signal-controlled lanes hold long "
            "vehicle queues or extreme waiting times?"
        ),
    }


# ---------------------------------------------------------------------- #
# MaxPressure baseline (no decision model)
# ---------------------------------------------------------------------- #


def phase_pressure(observation: IntersectionObservation, phase: str) -> float:
    """Classic max-pressure score of one canonical phase.

    Pressure = queued + approaching vehicles on the two lanes the phase
    serves (LLMLight's MaxPressure semantics, matching the mock Jev's
    ``queue + approaching`` scoring).
    """
    pressure = 0.0
    for lane_name in (phase[:2], phase[2:]):
        lane = observation.lanes[lane_name]
        pressure += float(lane.queue)
        pressure += float(sum(lane.llmlight_cells))
    return pressure


class MaxPressureController:
    """Non-LLM MaxPressure baseline: no Jev, no requests, pure local state.

    Exposes the same ``decide``/``usage_summary`` surface as
    :class:`JevLightController` so the runner's control loop is unchanged;
    every decision is the argmax-pressure phase of the intersection's own
    observation.
    """

    def __init__(self, phase_duration: float):
        self.phase_duration = phase_duration

    @staticmethod
    def _phase_to_action(observation: IntersectionObservation, phase: str) -> int:
        for index, available in enumerate(observation.phases):
            if canonical_phase_name(available) == canonical_phase_name(phase):
                return index
        return 0

    def decide(
        self,
        observations: Sequence[IntersectionObservation],
        step: int,
    ) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
        actions: Dict[str, int] = {}
        traces: List[Dict[str, Any]] = []
        for observation in observations:
            scores: Dict[str, float] = {}
            for phase in observation.phases:
                canonical = canonical_phase_name(phase)
                if canonical not in scores:
                    scores[canonical] = phase_pressure(observation, canonical)
            best = max(sorted(scores), key=scores.get)
            actions[observation.intersection_id] = self._phase_to_action(
                observation, best
            )
            traces.append({
                "step": step,
                "intersection": observation.intersection_id,
                "controller": "maxpressure",
                "agent_mode": "maxpressure",
                "packaging": "none",
                "transport": "local",
                "model": "maxpressure",
                "signal": best,
                "pressures": scores,
                "fallback": None,
            })
        return actions, traces

    def usage_summary(self) -> Dict[str, Any]:
        return {
            "requests": 0,
            "failures": 0,
            "rate_limited": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_mean_s": None,
            "latency_max_s": None,
        }


# ---------------------------------------------------------------------- #
# Controller
# ---------------------------------------------------------------------- #


class JevLightController:
    """LLMLight-style control loop with Jev as the decision model.

    ``decide`` mirrors ``LLMLightController.decide``: observations in, one
    action index per intersection (into ``observation.phases``) and one trace
    record per decision out.  Empty junctions use the exact CoLLMLight local
    ranking without an API call.
    """

    def __init__(
        self,
        client: JevClient,
        phase_duration: float,
        packaging: Optional[str] = None,
        fallback: str = "previous",
        min_confidence: float = 0.0,
        speculative: bool = False,
        max_workers: int = 8,
        agent_mode: str = "cojevlight",
        cojev_history_steps: int = 0,
    ):
        agent_mode = AGENT_MODE_ALIASES.get(agent_mode, agent_mode)
        if agent_mode not in AGENT_MODES:
            raise ValueError(
                f"agent_mode must be one of {AGENT_MODES}, got {agent_mode}"
            )
        packaging = packaging or AGENT_MODE_DEFAULTS[agent_mode]["packaging"]
        if agent_mode == "cojevlight" and packaging != "per_intersection":
            raise ValueError("cojevlight requires per_intersection packaging (target and direct neighbours only)")
        if packaging not in JEV_PACKAGINGS:
            raise ValueError(
                f"packaging must be one of {JEV_PACKAGINGS}, got {packaging}"
            )
        if fallback not in JEV_FALLBACKS:
            raise ValueError(
                f"fallback must be one of {JEV_FALLBACKS}, got {fallback}"
            )
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        self.client = client
        self.phase_duration = phase_duration
        self.packaging = packaging
        self.fallback = fallback
        self.min_confidence = float(min_confidence)
        self.speculative = speculative
        self.max_workers = max(1, int(max_workers))
        self.agent_mode = agent_mode
        if cojev_history_steps != 0:
            raise ValueError("cojev_history_steps must be 0: cooperative decisions use current state only")
        self.trajectory_relations = TrajectoryRelations(CONTROLLED_LANES) if agent_mode == "cojevlight" else None
        self.cooperative_states: Dict[str, Any] = {}
        self.previous_signals: Dict[str, str] = {}
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
        self.total_failures = 0
        self.total_rate_limited = 0
        self.latencies: List[float] = []

    # -- helpers ------------------------------------------------------- #

    def _phase_question(
        self,
        observation: IntersectionObservation,
        state_path: str,
    ) -> Dict[str, Any]:
        """The phase Choice question worded for the active agent mode."""
        if self.agent_mode == "jevlight":
            return jevlight_phase_question(
                observation, self.phase_duration, state_path
            )
        return cojevlight_phase_question(
            observation,
            self.phase_duration,
            state_path,
        )

    @staticmethod
    def _phase_to_action(observation: IntersectionObservation, phase: str) -> int:
        for index, available in enumerate(observation.phases):
            if canonical_phase_name(available) == canonical_phase_name(phase):
                return index
        return 0

    def _previous_signal(self, observation: IntersectionObservation) -> str:
        signal = self.previous_signals.get(observation.intersection_id)
        if signal is None and observation.current_phase is not None:
            signal = canonical_phase_name(observation.current_phase)
        available = {
            canonical_phase_name(phase) for phase in observation.phases
        }
        if signal in available:
            return signal
        return canonical_phase_name(observation.phases[0])

    def _fallback_signal(
        self,
        observation: IntersectionObservation,
        reason: str,
    ) -> Tuple[str, str]:
        if self.fallback == "ranking":
            return rank_phases(observation, self.phase_duration)[0][0], reason
        return self._previous_signal(observation), reason

    def _record_result(self, result: JevResult) -> None:
        self.total_requests += 1
        self.total_input_tokens += result.input_tokens
        self.total_output_tokens += result.output_tokens
        self.latencies.append(result.latency)
        if result.error:
            self.total_failures += 1
        if result.rate_limited:
            self.total_rate_limited += 1

    def _select_signal(
        self,
        observation: IntersectionObservation,
        answer: Optional[Any],
    ) -> Tuple[str, Optional[str], Optional[float], Optional[Dict[str, float]]]:
        """Apply the answer with validity and confidence gating.

        Returns ``(signal, fallback_reason, confidence, probabilities)``.
        ``fallback_reason`` is ``None`` when the Jev answer is used directly.
        """
        if answer is None or answer.get("type") != "choice":
            signal, reason = self._fallback_signal(observation, "missing_answer")
            return signal, reason, None, None
        signal = answer.get("choice")
        probabilities = {
            str(key): float(value)
            for key, value in (answer.get("probabilities") or {}).items()
        }
        confidence = answer.get("confidence")
        confidence_value = (
            float(confidence) if confidence is not None else None
        )
        available = {
            canonical_phase_name(phase) for phase in observation.phases
        }
        if signal not in available:
            fallback_signal, reason = self._fallback_signal(
                observation, "invalid_choice"
            )
            return fallback_signal, reason, confidence_value, probabilities
        if (
            confidence_value is not None
            and confidence_value < self.min_confidence
        ):
            fallback_signal, reason = self._fallback_signal(
                observation, "low_confidence"
            )
            return fallback_signal, reason, confidence_value, probabilities
        return signal, None, confidence_value, probabilities

    def _empty_signal(self, observation: IntersectionObservation) -> str:
        ranking = rank_phases(observation, self.phase_duration)
        return ranking[0][0]

    def _trace_base(
        self,
        observation: IntersectionObservation,
        step: int,
    ) -> Dict[str, Any]:
        return {
            "step": step,
            "intersection": observation.intersection_id,
            "controller": "jevlight",
            "agent_mode": self.agent_mode,
            "packaging": self.packaging,
            "transport": self.client.transport,
            "model": self.client.model or "server-default",
            "prompt_version": COOPERATIVE_PROMPT_VERSION if self.agent_mode == "cojevlight" else PROMPT_VERSION,
        }

    def _trace_decision(
        self,
        observation: IntersectionObservation,
        step: int,
        question_id: str,
        answer: Optional[Any],
        result: JevResult,
    ) -> Tuple[str, Dict[str, Any]]:
        signal, fallback, confidence, probabilities = self._select_signal(
            observation, answer
        )
        congestion = result.answers.get(
            f"{CONGESTION_QUESTION_PREFIX}{observation.intersection_id}"
        )
        trace = self._trace_base(observation, step)
        trace.update({
            "question_id": question_id,
            "answer": answer,
            "signal": signal,
            "probabilities": probabilities,
            "confidence": confidence,
            "congestion_noul": (
                congestion.get("noul") if congestion else None
            ),
            "fallback": fallback,
            "error": result.error,
            "rate_limited": result.rate_limited,
            "request_latency": result.latency,
            "request_input_tokens": result.input_tokens,
            "request_output_tokens": result.output_tokens,
        })
        return signal, trace

    # -- decision paths ------------------------------------------------ #

    def _prepare_cooperative_states(self, observations: Sequence[IntersectionObservation]) -> None:
        self.cooperative_states = {}
        for observation in observations:
            state = jev_intersection_state(observation, self.phase_duration, include_right_turns=True)
            state["prompt_version"] = COOPERATIVE_PROMPT_VERSION
            state.pop("neighbours", None)
            state["cooperation_source"] = "observed_vehicle_transitions"
            available = set(state["available_phases"])
            state["ranking_assumed_speed_m_s"] = 11.11
            state["local_phase_priority"] = [
                {"phase": phase, "estimated_waiting_time_reduction_s": round(score, 1)}
                for phase, score in rank_phases(observation, self.phase_duration)
                if phase in available
            ]
            self.cooperative_states[observation.intersection_id] = state

    def _decide_network(
        self,
        observations: Sequence[IntersectionObservation],
        active: Sequence[IntersectionObservation],
        step: int,
    ) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
        if self.agent_mode == "cojevlight":
            raise ValueError("cojevlight does not support network packaging")
        state: Dict[str, Any] = {
            "scenario": (
                "Traffic signal control. The state holds every signalized "
                "intersection of the road network. Movement codes name the "
                "incoming side: ET is through from east to west; WL is "
                "left-turn from west toward north on a regular junction. "
                "Only the provided phase criteria are selectable. Right-turn "
                "movements supply context, not extra phase choices."
            ),
            "intersections": {
                observation.intersection_id: jev_intersection_state(observation, self.phase_duration)
                for observation in observations
            },
        }
        questions: Dict[str, Any] = {}
        for observation in active:
            state_path = f"intersections[{json.dumps(observation.intersection_id)}]"
            questions[f"{PHASE_QUESTION_PREFIX}{observation.intersection_id}"] = (
                self._phase_question(
                    observation, state_path
                )
            )
            if self.speculative:
                questions[
                    f"{CONGESTION_QUESTION_PREFIX}{observation.intersection_id}"
                ] = congestion_question(state_path)

        result = self.client.evaluate(state, questions)
        self._record_result(result)
        actions: Dict[str, int] = {}
        traces: List[Dict[str, Any]] = []
        for observation in active:
            question_id = (
                f"{PHASE_QUESTION_PREFIX}{observation.intersection_id}"
            )
            signal, trace = self._trace_decision(
                observation, step, question_id, result.answers.get(question_id), result
            )
            actions[observation.intersection_id] = self._phase_to_action(
                observation, signal
            )
            traces.append(trace)
        return actions, traces

    def _decide_per_intersection(
        self,
        active: Sequence[IntersectionObservation],
        step: int,
    ) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
        requests: List[Tuple[IntersectionObservation, Any, Dict[str, Any]]] = []
        for observation in active:
            state = jev_intersection_state(observation, self.phase_duration)
            if self.agent_mode == "cojevlight":
                state = dict(self.cooperative_states[observation.intersection_id])
                state["cooperative_view"] = self.trajectory_relations.view(
                    observation, self.cooperative_states, include_empty_neighbours=False,
                    occupancy_threshold=COMMUNICATION_THRESHOLD,
                )
            questions: Dict[str, Any] = {
                f"{PHASE_QUESTION_PREFIX}{observation.intersection_id}": (
                    self._phase_question(observation, "$")
                )
            }
            if self.speculative:
                questions[
                    f"{CONGESTION_QUESTION_PREFIX}{observation.intersection_id}"
                ] = congestion_question("$")
            requests.append((observation, state, questions))

        if getattr(self.client, 'batch_size', 0):
            results = self.client.evaluate_batch([
                {'state': state, 'questions': questions} for _, state, questions in requests])
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(
                    executor.map(
                        lambda item: self.client.evaluate(item[1], item[2]),
                        requests,
                    )
                )
        if len(results) != len(requests):
            raise ValueError('batch result count mismatch')

        actions: Dict[str, int] = {}
        traces: List[Dict[str, Any]] = []
        for (observation, _state, _), result in zip(requests, results):
            self._record_result(result)
            question_id = f"{PHASE_QUESTION_PREFIX}{observation.intersection_id}"
            signal, trace = self._trace_decision(
                observation, step, question_id, result.answers.get(question_id), result
            )
            actions[observation.intersection_id] = self._phase_to_action(
                observation, signal
            )
            traces.append(trace)
        return actions, traces

    def decide(
        self,
        observations: Sequence[IntersectionObservation],
        step: int,
    ) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
        if self.agent_mode == "cojevlight":
            self.trajectory_relations.observe(observations, step)
            self._prepare_cooperative_states(observations)
        active = [
            observation for observation in observations if not observation.empty
        ]
        actions: Dict[str, int] = {}
        traces: List[Dict[str, Any]] = []

        for observation in observations:
            if observation.empty:
                signal = self._empty_signal(observation)
                actions[observation.intersection_id] = self._phase_to_action(
                    observation, signal
                )
                trace = self._trace_base(observation, step)
                trace.update({
                    "signal": signal,
                    "fallback": "empty_junction_local_ranking",
                })
                traces.append(trace)

        if active:
            if self.packaging == "network":
                active_actions, active_traces = self._decide_network(
                    observations, active, step
                )
            else:
                active_actions, active_traces = self._decide_per_intersection(
                    active, step
                )
            actions.update(active_actions)
            traces.extend(active_traces)

        for observation in observations:
            action = actions[observation.intersection_id]
            signal = next(
                (
                    canonical_phase_name(phase)
                    for index, phase in enumerate(observation.phases)
                    if index == action
                ),
                canonical_phase_name(observation.phases[0]),
            )
            self.previous_signals[observation.intersection_id] = signal
        return actions, traces

    def usage_summary(self) -> Dict[str, Any]:
        latencies = self.latencies
        return {
            "requests": self.total_requests,
            "failures": self.total_failures,
            "rate_limited": self.total_rate_limited,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "latency_mean_s": (
                sum(latencies) / len(latencies) if latencies else None
            ),
            "latency_max_s": max(latencies) if latencies else None,
        }
