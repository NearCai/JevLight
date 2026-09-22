"""TypeSafe Jev phase-and-duration agent."""

import copy
import json
import os
import re

import requests

from utils.my_utils import get_state_detail


class JevAgent:
    """Choose a zero-based signal phase and duration-option index with Jev."""

    API_URL = "https://api.typesafe.ai/v1/systemone"
    MODES = {"choice", "score", "noul"}
    PHASE_LANES = (("ET", "WT"), ("NT", "ST"), ("EL", "WL"), ("NL", "SL"))
    # This is deliberately kept in the agent as a fallback for deployments that
    # do not package the prompt files.  The same text is also available in
    # ``prompts/prompt_jev.json`` for inspection and prompt experiments.
    SYSTEM_PROMPT = (
        "You are controlling one isolated four-arm signalized intersection. "
        "The objective is to minimize queue length and waiting time while keeping "
        "all approaches moving. A phase serves the two movement groups listed in "
        "its legend; all other movements are red. ET/WT means eastbound and "
        "westbound through, NT/ST means northbound and southbound through, "
        "EL/WL means eastbound and westbound left-turn, and NL/SL means "
        "northbound and southbound left-turn. A movement key such as ET is the "
        "through movement from the east approach, not an arbitrary road id. "
        "queue_len counts vehicles currently stopped and waiting for a green; "
        "avg_wait_time is the mean age of those stopped vehicles. cells[0] is "
        "the farthest approaching bin and cells[3] is closest to the stop line. "
        "Approaching vehicles in close bins may join the queue during the next "
        "green, so they matter, but an existing queue and old waiting vehicles "
        "matter more. The reported mean speed is for moving vehicles only and a "
        "low value is a warning sign, not a reason to starve another approach. "
        "When changing phase, the simulator spends 5 seconds in yellow; that "
        "transition is additional to the requested green duration. The controller "
        "is checked every 5 seconds, so choose only the supplied duration values. "
        "Do not switch for a tiny difference: serve the largest urgent backlog, "
        "respect long waits, and avoid repeatedly selecting one direction when "
        "another direction has a material queue. Return only the requested "
        "structured answer; do not invent phase or duration names."
    )

    def __init__(
        self,
        dic_agent_conf,
        dic_traffic_env_conf,
        dic_path=None,
        intersection=None,
        inter_name="0",
        phase_num=4,
        **_,
    ):
        if intersection is None:
            raise ValueError("JevAgent requires intersection road metadata")

        self.roads = copy.deepcopy(intersection["roads"])
        self.inter_name = inter_name
        self.phase_num = int(phase_num)
        self.phase_options = [f"phase_{i + 1}" for i in range(self.phase_num)]
        durations = dic_traffic_env_conf.get(
            "ACTION_DURATION", {i: value for i, value in enumerate((15, 20, 25, 30, 35, 40))}
        )
        self.duration_options = [int(durations[key]) for key in sorted(durations, key=int)]
        self.yellow_time = int(dic_traffic_env_conf.get("YELLOW_TIME", 5))
        self.decision_interval = int(dic_traffic_env_conf.get("DECISION_INTERVAL", 5))
        self.duration_excludes_transition = bool(
            dic_traffic_env_conf.get("DURATION_EXCLUDES_YELLOW", True)
        )

        settings = {**dic_traffic_env_conf, **dic_agent_conf}
        self.system_prompt = settings.get("JEV_SYSTEM_PROMPT") or self._load_system_prompt()
        self.api_key = settings.get("JEV_API_KEY") or os.getenv("TYPESAFE_API_KEY", "")
        self.endpoint = settings.get("JEV_ENDPOINT", self.API_URL)
        self.model = settings.get("JEV_MODEL", "jev-latest")
        self.phase_mode = settings.get("JEV_PHASE_MODE", "choice").lower()
        self.duration_mode = settings.get("JEV_DURATION_MODE", "choice").lower()
        self.timeout = float(settings.get("JEV_TIMEOUT", 30))
        self.max_concurrency = int(settings.get("JEV_MAX_CONCURRENCY", 8))
        self.use_fallback = bool(settings.get("JEV_FALLBACK", True))
        self.use_guardrail = bool(settings.get("JEV_GUARDRAIL", True))
        self.last_error = None
        self.last_response = None
        self.last_exchange = []
        self.last_guardrail = None

        invalid = {self.phase_mode, self.duration_mode} - self.MODES
        if invalid:
            raise ValueError(f"Unsupported Jev decision mode: {', '.join(sorted(invalid))}")
        if not self.duration_options:
            raise ValueError("ACTION_DURATION must contain at least one duration")
        if self.decision_interval <= 0 or self.yellow_time <= 0:
            raise ValueError("Jev decision and transition times must be positive")
        if any(seconds <= 0 or seconds % self.decision_interval for seconds in self.duration_options):
            raise ValueError("Jev durations must be positive multiples of DECISION_INTERVAL")
        if self.phase_mode == "score" and not 2 <= len(self.phase_options) <= 10:
            raise ValueError("Jev score mode requires 2 to 10 phase levels")
        if self.duration_mode == "score" and not 2 <= len(self.duration_options) <= 10:
            raise ValueError("Jev score mode requires 2 to 10 duration levels")

    def choose_action(self, env, state=None, count=None):
        self.last_error = None
        self.last_response = None
        self.last_exchange = []
        self.last_guardrail = None
        if not self.api_key and not self.use_fallback:
            raise RuntimeError("TYPESAFE_API_KEY is required when fallback is disabled")

        detail, incoming, mean_speed = get_state_detail(self.roads, env)
        fallback = self._fallback(detail)
        if not self.api_key:
            return fallback

        try:
            phase_state = self._state_json(detail, incoming, mean_speed, state, count)
            phase_questions = self._questions("phase", self.phase_mode, self._phase_criteria(), "signal phase")
            phase_response = self._request(phase_state, phase_questions)
            self.last_exchange.append({
                "kind": "phase", "state": json.loads(phase_state),
                "questions": phase_questions, "response": phase_response,
            })
            phase = self._decision_index(
                phase_response["answers"], "phase", self.phase_mode, self.phase_options
            )
            duration_options = [f"duration_{seconds}" for seconds in self.duration_options]
            duration_state = self._state_json(
                detail, incoming, mean_speed, state, count, selected_phase=self.phase_options[phase]
            )
            duration_questions = self._questions(
                    "duration",
                    self.duration_mode,
                    [(name, f"Hold {self.phase_options[phase]} green for {seconds} seconds")
                     for name, seconds in zip(duration_options, self.duration_options)],
                    f"green duration for {self.phase_options[phase]}",
            )
            duration_response = self._request(duration_state, duration_questions)
            self.last_exchange.append({
                "kind": "duration", "state": json.loads(duration_state),
                "questions": duration_questions, "response": duration_response,
            })
            duration = self._decision_index(
                duration_response["answers"], "duration", self.duration_mode, duration_options
            )
            self.last_response = {"phase": phase_response, "duration": duration_response}
            phase, duration = self._apply_guardrail(phase, duration, detail, state)
            return phase, duration
        except (KeyError, TypeError, ValueError, requests.RequestException) as exc:
            self.last_error = str(exc)
            if self.use_fallback:
                return fallback
            raise RuntimeError(f"Jev request failed for {self.inter_name}: {exc}") from exc

    def _request(self, state, questions):
        response = requests.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"state": state, "model": self.model, "questions": questions},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def _state_json(self, detail, incoming, mean_speed, state, count, selected_phase=None):
        phase_index, phase_elapsed = self._current_phase_context(state)
        snapshot = {
            "traffic_control_task": (
                "Select the movement phase with the greatest urgent service need, "
                "then select a green duration that clears it without starving other approaches."
            ),
            "controller_instructions": self.system_prompt,
            "intersection": self.inter_name,
            "decision_step": count,
            "simulator_state": state,
            "current_phase": phase_index,
            "current_phase_elapsed_seconds": phase_elapsed,
            "outgoing_lanes": detail,
            "incoming_lanes": incoming,
            "mean_approaching_speed_mps": mean_speed,
            "phase_metrics": self._phase_metrics(detail),
            "available_phases": dict(self._phase_criteria()),
            "available_durations_seconds": self.duration_options,
            "decision_check_interval_seconds": self.decision_interval,
            "phase_change_transition_seconds": self.yellow_time,
            "duration_excludes_transition": self.duration_excludes_transition,
        }
        if selected_phase is not None:
            snapshot["selected_phase"] = selected_phase
        return json.dumps(snapshot, ensure_ascii=False, default=self._json_default, sort_keys=True)

    def _phase_criteria(self):
        descriptions = (
            "Eastbound and westbound through traffic",
            "Northbound and southbound through traffic",
            "Eastbound and westbound left turns",
            "Northbound and southbound left turns",
        )
        return [
            (name, descriptions[i] if i < len(descriptions) else f"Traffic signal phase {i + 1}")
            for i, name in enumerate(self.phase_options)
        ]

    @staticmethod
    def _current_phase_context(state):
        if not isinstance(state, dict):
            return None, None
        try:
            phase = int(state.get("cur_phase", [0])[0])
            phase = phase if phase > 0 else None
        except (TypeError, ValueError, IndexError):
            phase = None
        try:
            elapsed = float(state.get("time_this_phase", [0])[0])
        except (TypeError, ValueError, IndexError):
            elapsed = None
        return phase, elapsed

    @classmethod
    def _load_system_prompt(cls):
        prompt_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "prompts", "prompt_jev.json")
        )
        try:
            with open(prompt_path, encoding="utf-8") as prompt_file:
                prompt = json.load(prompt_file).get("system_prompt")
            if isinstance(prompt, str) and prompt.strip():
                return prompt.strip()
        except (OSError, TypeError, ValueError):
            pass
        return cls.SYSTEM_PROMPT

    @classmethod
    def _phase_metrics(cls, detail):
        """Return comparable, model-readable urgency summaries for each phase."""
        metrics = {}
        for index, (first, second) in enumerate(cls.PHASE_LANES, start=1):
            lanes = [detail.get(first, {}), detail.get(second, {})]
            queue = sum(float(lane.get("queue_len", 0.0)) for lane in lanes)
            waiting = sum(
                float(lane.get("queue_len", 0.0)) * float(lane.get("avg_wait_time", 0.0))
                for lane in lanes
            )
            approaching = sum(
                sum(float(value) for value in lane.get("cells", ())) for lane in lanes
            )
            # Cells nearest the stop line are more likely to affect the next
            # decision interval. Existing queues remain the dominant signal.
            near = sum(
                sum(float(value) * weight for value, weight in zip(
                    lane.get("cells", ()), (0.1, 0.3, 0.7, 1.0)
                ))
                for lane in lanes
            )
            score = 10.0 * queue + 0.05 * waiting + 2.0 * near
            metrics[f"phase_{index}"] = {
                "served_movements": [first, second],
                "queue_vehicles": round(queue, 3),
                "total_waiting_seconds": round(waiting, 3),
                "approaching_vehicles": round(approaching, 3),
                "near_stopline_weighted_approaching": round(near, 3),
                "urgency_score": round(score, 3),
            }
        return metrics

    def _apply_guardrail(self, phase, duration, detail, state):
        """Reject an implausible model action when it ignores a clear backlog.

        This is intentionally conservative: the model remains in control when
        phase scores are close, while a choice that serves less than 60% of the
        best urgent score is corrected to the best phase. This protects long
        runs from a malformed/overconfident response without hiding the reply.
        """
        if not self.use_guardrail:
            return phase, duration
        metrics = self._phase_metrics(detail)
        available = min(self.phase_num, len(self.PHASE_LANES))
        scores = [metrics[f"phase_{i + 1}"]["urgency_score"] for i in range(available)]
        if not scores:
            return phase, duration
        best = max(range(len(scores)), key=lambda i: (scores[i], -i))
        selected = min(len(scores) - 1, max(0, int(phase)))
        if scores[best] > 0 and scores[selected] < 0.60 * scores[best]:
            self.last_guardrail = {
                "reason": "selected phase served substantially less urgent backlog",
                "model_phase": selected,
                "guardrail_phase": best,
                "phase_scores": scores,
            }
            selected = best
        selected_queue = metrics[f"phase_{selected + 1}"]["queue_vehicles"]
        # Avoid very long green on an empty phase and ensure a visibly large
        # queue is not assigned the shortest option by an errant response.
        if max(scores) > 0 and selected_queue <= 1:
            duration = 0
        elif max(scores) > 0 and selected_queue >= 40:
            duration = len(self.duration_options) - 1
        elif max(scores) > 0 and selected_queue >= 20:
            duration = max(duration, self.duration_options.index(30)) if 30 in self.duration_options else duration
        return selected, min(len(self.duration_options) - 1, max(0, int(duration)))

    @staticmethod
    def _questions(name, mode, criteria, subject):
        if mode == "choice":
            return {name: {
                "type": "choice",
                "instructions": (
                    f"Choose the {subject} that best reduces current congestion. "
                    "Use phase_metrics first: queue vehicles and total waiting time "
                    "are more important than distant approaching vehicles. Break close "
                    "ties with near-stopline demand and fairness. Consider the current "
                    "phase and its elapsed time; a change costs 5 seconds of yellow. "
                    "Choose exactly one supplied option."
                ),
                "criteria": dict(criteria),
            }}
        if mode == "score":
            return {name: {
                "type": "score",
                "instructions": (
                    f"Choose the best {subject}; return the zero-based score whose legend entry is best. "
                    "Score options by urgent queue and accumulated waiting time, then by near-stopline "
                    "arrivals and fairness. Account for the 5-second yellow transition."
                ),
                "criteria": [f"{key}: {description}" for key, description in criteria],
            }}
        return {
            f"{name}_{key}": {
                "type": "noul",
                "instructions": (
                    f"{key} is the best {subject} for reducing current congestion. "
                    f"It means: {description}. Prefer high queue and old waiting vehicles, "
                    "then close arrivals; avoid starvation and unnecessary phase changes."
                ),
            }
            for key, description in criteria
        }

    @classmethod
    def _decision_index(cls, answers, name, mode, options):
        if mode != "noul":
            return cls._answer_index(answers[name], mode, options)
        values = [float(answers[f"{name}_{option}"]["noul"]) for option in options]
        return max(range(len(values)), key=values.__getitem__)

    @classmethod
    def _answer_index(cls, answer, mode, options):
        if not isinstance(answer, dict):
            answer = {mode: answer}

        if mode == "choice":
            value = answer.get("choice", answer.get("value", answer.get("answer")))
            matched = cls._match_option(value, options)
            if matched is not None:
                return matched
            probabilities = answer.get("probabilities", {})
            if probabilities:
                matched = cls._match_option(max(probabilities, key=probabilities.get), options)
                if matched is not None:
                    return matched
            raise ValueError(f"Invalid choice answer: {answer!r}")

        probabilities = answer.get("probabilities", {})
        if probabilities:
            key = max(probabilities, key=lambda item: float(probabilities[item]))
            try:
                index = int(key)
            except (TypeError, ValueError):
                index = cls._match_option(key, options)
            if index is not None and 0 <= index < len(options):
                return index

        value = answer.get(mode, answer.get("value", answer.get("answer")))
        number = float(value)
        return min(len(options) - 1, max(0, int(number + 0.5)))

    @staticmethod
    def _match_option(value, options):
        if value is None:
            return None
        normalized = re.sub(r"[^a-z0-9]", "", str(value).lower())
        for index, option in enumerate(options):
            if normalized == re.sub(r"[^a-z0-9]", "", option.lower()):
                return index

        numbers = re.findall(r"\d+", str(value))
        if numbers:
            number = int(numbers[-1])
            for index, option in enumerate(options):
                if str(number) in re.findall(r"\d+", option):
                    return index
        return None

    def _fallback(self, detail):
        loads = []
        for lanes in self.PHASE_LANES[: self.phase_num]:
            loads.append(sum(self._lane_load(detail.get(lane, {})) for lane in lanes))
        loads.extend([0.0] * (self.phase_num - len(loads)))
        phase = max(range(self.phase_num), key=lambda index: (loads[index], -index))

        # ponytail: coarse five-vehicle buckets are only an offline fallback;
        # calibrate them if fallback policy quality becomes operationally important.
        duration = min(len(self.duration_options) - 1, max(0, int(loads[phase] // 5)))
        return phase, duration

    @staticmethod
    def _lane_load(lane):
        return float(lane.get("queue_len", 0)) + sum(map(float, lane.get("cells", ())))

    @staticmethod
    def _json_default(value):
        if hasattr(value, "item"):
            return value.item()
        if hasattr(value, "tolist"):
            return value.tolist()
        return str(value)


def _self_check():
    options = ["phase_1", "phase_2", "phase_3", "phase_4"]
    assert JevAgent._answer_index({"choice": "Phase-3"}, "choice", options) == 2
    assert JevAgent._answer_index({"score": 1.8}, "score", options) == 2
    answers = {f"phase_{option}": {"noul": value} for option, value in zip(options, (0.1, 0.8, 0.2, 0.3))}
    assert JevAgent._decision_index(answers, "phase", "noul", options) == 1


if __name__ == "__main__":
    _self_check()
    print("JevAgent self-check passed")
