"""Local Laya typed-decision agent for JevLight.

Laya is used as a drop-in, offline replacement for the hosted Jev endpoint.  A
single process-wide checkpoint is shared by all intersections so a 24-intersection
CityFlow run does not allocate 24 copies of the 421M-parameter encoder.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any

from utils.my_utils import get_state_detail


class LayaAgent:
    """Select a signal phase and green duration with one local Laya forward pass."""

    _model = None
    _model_key = None
    _model_lock = threading.Lock()
    _infer_lock = threading.Lock()
    _warmed_keys = set()

    PHASE_LANES = (("ET", "WT"), ("NT", "ST"), ("EL", "WL"), ("NL", "SL"))

    def __init__(
        self,
        dic_agent_conf,
        dic_traffic_env_conf,
        intersection=None,
        inter_name="0",
        phase_num=4,
        **_,
    ):
        if intersection is None:
            raise ValueError("LayaAgent requires intersection road metadata")
        self.roads = intersection["roads"]
        self.inter_name = inter_name
        self.phase_num = int(phase_num)
        self.phase_options = [f"phase_{i + 1}" for i in range(self.phase_num)]
        durations = dic_traffic_env_conf.get(
            "ACTION_DURATION", {i: value for i, value in enumerate((15, 20, 25, 30, 35, 40))}
        )
        self.duration_options = [int(durations[key]) for key in sorted(durations, key=int)]
        self.model_name = dic_agent_conf.get("LAYA_MODEL", "convaiinnovations/laya-typed-decisions")
        self.device = dic_agent_conf.get("LAYA_DEVICE") or None
        self.max_len = int(dic_agent_conf.get("LAYA_MAX_LEN", 1024))
        self.use_guardrail = bool(dic_agent_conf.get("LAYA_GUARDRAIL", True))
        self.min_confidence = float(dic_agent_conf.get("LAYA_MIN_CONFIDENCE", 0.0))
        self.last_response = None
        self.last_error = None
        self.last_guardrail = None
        self.last_state = None
        self.last_confidence = None
        self.last_probabilities = None

        if not self.duration_options:
            raise ValueError("ACTION_DURATION must contain at least one duration")
        self._ensure_model(self.model_name, self.device)

    @classmethod
    def _ensure_model(cls, model_name=None, device=None):
        model_name = model_name or os.getenv("LAYA_MODEL", "convaiinnovations/laya-typed-decisions")
        device = device or os.getenv("LAYA_DEVICE") or None
        key = (model_name, device)
        with cls._model_lock:
            if cls._model is None or cls._model_key != key:
                from laya import Agent

                cls._model = Agent(model_name, device=device)
                cls._model_key = key
                cls._warmed_keys.discard(key)
        return cls._model

    @classmethod
    def warmup(cls, model_name=None, device=None):
        """Run one representative forward pass before CityFlow starts.

        The first CUDA call includes kernel/context initialization and is much
        slower than steady-state inference.  Warming up with the same four- and
        six-option shapes used by the controller removes that cold-start spike
        from the first live intersection decision.
        """
        model = cls._ensure_model(model_name, device)
        key = (model_name or os.getenv("LAYA_MODEL", "convaiinnovations/laya-typed-decisions"),
               device or os.getenv("LAYA_DEVICE") or None)
        with cls._infer_lock:
            if key in cls._warmed_keys:
                return None
            questions = {
                "phase": {
                    "type": "choice",
                    "instructions": "Choose the best traffic signal phase.",
                    "criteria": {f"phase_{i}": f"Phase {i}" for i in range(1, 5)},
                },
                "duration": {
                    "type": "choice",
                    "instructions": "Choose the green duration in seconds.",
                    "criteria": {f"duration_{s}": f"Hold green for {s} seconds"
                                 for s in (15, 20, 25, 30, 35, 40)},
                },
            }
            try:
                result = model.system_one(
                    '{"task":"traffic_signal_control","warmup":true}', questions
                )
            except Exception as exc:
                # A warmup failure must not prevent the normal fallback path
                # from starting the experiment; the next live call will retry.
                return f"{type(exc).__name__}: {exc}"
            cls._warmed_keys.add(key)
            return result

    def choose_action(self, env, state=None, count=None):
        self.last_error = None
        self.last_guardrail = None
        self.last_confidence = None
        self.last_probabilities = None
        detail, incoming, mean_speed = get_state_detail(self.roads, env)
        fallback = self._fallback(detail)
        state_payload = self._state_payload(detail, incoming, mean_speed, state, count)
        questions = {
            "phase": {
                "type": "choice",
                "instructions": (
                    "Choose the signal phase with the greatest urgent service need. "
                    "Prioritize existing queue vehicles and waiting time, then close arrivals; "
                    "avoid starvation and unnecessary switching."
                ),
                "criteria": {
                    f"phase_{i + 1}": description
                    for i, description in enumerate(self._phase_descriptions()[: self.phase_num])
                },
            },
            "duration": {
                "type": "choice",
                "instructions": (
                    "Choose the green duration for the selected phase. Use a longer duration "
                    "for a large backlog and a shorter duration for a light queue."
                ),
                "criteria": {
                    f"duration_{seconds}": f"Hold green for {seconds} seconds"
                    for seconds in self.duration_options
                },
            },
        }
        try:
            # Laya's model is shared across intersections; serializing the forward pass
            # keeps CUDA state deterministic while state collection remains concurrent-safe.
            with self._infer_lock:
                response = self._ensure_model(self.model_name, self.device).system_one(
                    state_payload, questions
                )
            self.last_response = response
            phase_answer = response["answers"]["phase"]
            duration_answer = response["answers"]["duration"]
            confidences = [phase_answer.get("confidence"), duration_answer.get("confidence")]
            valid_confidences = [float(x) for x in confidences if x is not None]
            self.last_confidence = min(valid_confidences) if valid_confidences else None
            self.last_probabilities = {
                "phase": phase_answer.get("probabilities"),
                "duration": duration_answer.get("probabilities"),
            }
            if self.min_confidence and (
                self.last_confidence is None or self.last_confidence < self.min_confidence
            ):
                raise ValueError(
                    f"Laya confidence {self.last_confidence!r} below threshold "
                    f"{self.min_confidence:.3f}"
                )
            phase = self._match(phase_answer["choice"], self.phase_options)
            duration_names = [f"duration_{seconds}" for seconds in self.duration_options]
            duration = self._match(duration_answer["choice"], duration_names)
            if phase is None or duration is None:
                raise ValueError(f"Laya returned an unknown option: {response!r}")
            return self._apply_guardrail(phase, duration, detail)
        except Exception as exc:  # local inference failures retain a safe controller path
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.last_response = None
            return fallback

    def _state_payload(self, detail, incoming, mean_speed, state, count):
        payload = {
            "task": "traffic_signal_control",
            "intersection": self.inter_name,
            "decision_step": count,
            "current_phase": self._current_phase(state),
            "phase_elapsed_seconds": self._phase_elapsed(state),
            "mean_approaching_speed_mps": round(float(mean_speed), 3),
            "outgoing_lanes": detail,
            "incoming_lanes": incoming,
            "phase_metrics": self._phase_metrics(detail),
            "phase_change_cost_seconds": 5,
            "decision_interval_seconds": 5,
            "available_durations_seconds": self.duration_options,
        }
        self.last_state = payload
        # Compact JSON leaves room for the question and option descriptions under Laya's 1024-token limit.
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)

    @classmethod
    def _phase_descriptions(cls):
        return (
            "Eastbound and westbound through traffic",
            "Northbound and southbound through traffic",
            "Eastbound and westbound left turns",
            "Northbound and southbound left turns",
        )

    @classmethod
    def _phase_metrics(cls, detail):
        metrics = {}
        for i, (first, second) in enumerate(cls.PHASE_LANES, start=1):
            lanes = [detail.get(first, {}), detail.get(second, {})]
            queue = sum(float(x.get("queue_len", 0)) for x in lanes)
            waiting = sum(float(x.get("queue_len", 0)) * float(x.get("avg_wait_time", 0)) for x in lanes)
            near = sum(
                sum(float(v) * w for v, w in zip(x.get("cells", ()), (0.1, 0.3, 0.7, 1.0)))
                for x in lanes
            )
            metrics[f"phase_{i}"] = {
                "served_movements": [first, second],
                "queue_vehicles": round(queue, 2),
                "total_waiting_seconds": round(waiting, 2),
                "near_stopline_approaching": round(near, 2),
                "urgency_score": round(10 * queue + 0.05 * waiting + 2 * near, 2),
            }
        return metrics

    def _apply_guardrail(self, phase, duration, detail):
        if not self.use_guardrail:
            return phase, duration
        metrics = self._phase_metrics(detail)
        scores = [metrics[f"phase_{i + 1}"]["urgency_score"] for i in range(self.phase_num)]
        if scores:
            best = max(range(len(scores)), key=lambda i: (scores[i], -i))
            if scores[best] > 0 and scores[phase] < 0.60 * scores[best]:
                self.last_guardrail = {"model_phase": phase, "guardrail_phase": best, "phase_scores": scores}
                phase = best
            queue = metrics[f"phase_{phase + 1}"]["queue_vehicles"]
            if queue >= 40:
                duration = len(self.duration_options) - 1
            elif queue >= 20 and 30 in self.duration_options:
                duration = max(duration, self.duration_options.index(30))
        return phase, min(len(self.duration_options) - 1, max(0, duration))

    def _fallback(self, detail):
        loads = [sum(self._lane_load(detail.get(lane, {})) for lane in lanes) for lanes in self.PHASE_LANES[: self.phase_num]]
        loads.extend([0.0] * (self.phase_num - len(loads)))
        phase = max(range(self.phase_num), key=lambda i: (loads[i], -i))
        duration = min(len(self.duration_options) - 1, max(0, int(loads[phase] // 5)))
        return phase, duration

    @staticmethod
    def _lane_load(lane):
        return float(lane.get("queue_len", 0)) + sum(map(float, lane.get("cells", ())))

    @staticmethod
    def _match(value, options):
        value = str(value).lower().replace("-", "_").replace(" ", "")
        for i, option in enumerate(options):
            if value == option.lower().replace("-", "_").replace(" ", ""):
                return i
        digits = "".join(ch for ch in value if ch.isdigit())
        for i, option in enumerate(options):
            if digits and digits in option:
                return i
        return None

    @staticmethod
    def _current_phase(state):
        try:
            return int(state.get("cur_phase", [0])[0])
        except (AttributeError, TypeError, ValueError, IndexError):
            return None

    @staticmethod
    def _phase_elapsed(state):
        try:
            return float(state.get("time_this_phase", [0])[0])
        except (AttributeError, TypeError, ValueError, IndexError):
            return None
