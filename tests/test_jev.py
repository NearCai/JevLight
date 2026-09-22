import json
import sys
import types
import unittest
from unittest.mock import Mock, patch

sys.modules.setdefault("cityflow", types.SimpleNamespace())

import models.jev as jev_module
import models.chatgpt as chatgpt_module
from utils.cityflow_env import CityFlowEnv


class JevTest(unittest.TestCase):
    def setUp(self):
        roads = {
            "road": {
                "location": "East",
                "type": "outgoing",
                "length": 100,
                "go_straight": "next",
                "turn_left": "next",
                "lanes": {"go_straight": [0], "turn_left": [1], "turn_right": []},
            }
        }
        self.traffic_conf = {
            "ACTION_DURATION": {i: value for i, value in enumerate((15, 20, 25, 30, 35, 40))},
            "YELLOW_TIME": 5,
            "DECISION_INTERVAL": 5,
        }
        self.intersection = {"roads": roads}

    def test_choice_requests_phase_then_duration(self):
        agent = jev_module.JevAgent(
            {"JEV_API_KEY": "test"}, self.traffic_conf, intersection=self.intersection
        )
        phase_response = Mock()
        phase_response.raise_for_status.return_value = None
        phase_response.json.return_value = {"answers": {"phase": {"choice": "phase_4"}}}
        duration_response = Mock()
        duration_response.raise_for_status.return_value = None
        duration_response.json.return_value = {
            "answers": {"duration": {"choice": "duration_35"}}
        }

        with patch.object(jev_module, "get_state_detail", return_value=({}, {}, 0.0)), patch.object(
            jev_module.requests, "post", side_effect=(phase_response, duration_response)
        ) as post:
            self.assertEqual(agent.choose_action(object()), (3, 4))
            self.assertEqual(post.call_count, 2)
            self.assertEqual([item["kind"] for item in agent.last_exchange], ["phase", "duration"])
            phase_payload = post.call_args_list[0].kwargs["json"]
            duration_payload = post.call_args_list[1].kwargs["json"]
            self.assertEqual(set(phase_payload["questions"]), {"phase"})
            self.assertEqual(set(duration_payload["questions"]), {"duration"})
            self.assertEqual(json.loads(duration_payload["state"])["selected_phase"], "phase_4")

    def test_missing_key_fails_when_fallback_is_disabled(self):
        with patch.dict(jev_module.os.environ, {"TYPESAFE_API_KEY": ""}):
            agent = jev_module.JevAgent(
                {"JEV_API_KEY": "", "JEV_FALLBACK": False},
                self.traffic_conf,
                intersection=self.intersection,
            )
            with self.assertRaisesRegex(RuntimeError, "TYPESAFE_API_KEY"):
                agent.choose_action(object())

    def test_score_projects_to_highest_probability_level(self):
        options = ["phase_1", "phase_2", "phase_3", "phase_4"]
        answer = {"score": 1.5, "probabilities": {"0": 0.5, "1": 0, "2": 0, "3": 0.5}}
        self.assertEqual(jev_module.JevAgent._answer_index(answer, "score", options), 0)

    def test_phase_change_adds_transition_outside_green_duration(self):
        env = object.__new__(CityFlowEnv)
        env.dic_traffic_env_conf = {
            "ACTION_DURATION": {0: 15},
            "DURATION_EXCLUDES_YELLOW": True,
            "YELLOW_TIME": 5,
            "ACTION_PATTERN": "set",
            "MODEL_NAME": "Jev",
        }
        env.actions, env.duration, env.decision_interval = [0], [0], 5
        env.list_memory = [[None, None, None, []]]
        env.list_intersection = [types.SimpleNamespace(current_phase_index=1)]
        env.get_state = lambda **_: ([{}], False)
        env.get_current_time = lambda: 0
        env.get_feature = lambda: [{}]
        env._inner_step = lambda _: None
        env.get_reward = lambda state=None: [0]
        env.get_reward_dy = lambda ids: [0 for _ in ids]

        env.step([1], [0])
        self.assertEqual(env.duration, [15])  # 15 green seconds remain after 5 seconds of yellow.

        env.step([0], [0])
        self.assertEqual(env.duration, [10])  # No transition: five seconds of green elapsed.

        _, reward, _, _ = env.step([0])
        self.assertEqual(env.duration, [0])
        self.assertEqual(reward, [0])

    def test_state_payload_explains_control_context_and_phase_metrics(self):
        agent = jev_module.JevAgent(
            {"JEV_API_KEY": "test"}, self.traffic_conf, intersection=self.intersection
        )
        state = {"cur_phase": [2], "time_this_phase": [25]}
        detail = {
            "ET": {"queue_len": 10, "avg_wait_time": 20, "cells": [1, 2, 3, 4]},
            "WT": {"queue_len": 5, "avg_wait_time": 10, "cells": [0, 0, 1, 2]},
            "NT": {"queue_len": 1, "avg_wait_time": 2, "cells": [0, 0, 0, 1]},
            "ST": {"queue_len": 1, "avg_wait_time": 2, "cells": [0, 0, 0, 1]},
            "EL": {"queue_len": 0, "avg_wait_time": 0, "cells": [0, 0, 0, 0]},
            "WL": {"queue_len": 0, "avg_wait_time": 0, "cells": [0, 0, 0, 0]},
            "NL": {"queue_len": 0, "avg_wait_time": 0, "cells": [0, 0, 0, 0]},
            "SL": {"queue_len": 0, "avg_wait_time": 0, "cells": [0, 0, 0, 0]},
        }
        payload = json.loads(agent._state_json(detail, {}, 8.0, state, 3))
        self.assertEqual(payload["current_phase"], 2)
        self.assertEqual(payload["current_phase_elapsed_seconds"], 25.0)
        self.assertIn("yellow", payload["controller_instructions"])
        self.assertGreater(
            payload["phase_metrics"]["phase_1"]["urgency_score"],
            payload["phase_metrics"]["phase_2"]["urgency_score"],
        )

    def test_guardrail_corrects_phase_that_ignores_large_queue(self):
        agent = jev_module.JevAgent(
            {"JEV_API_KEY": "test"}, self.traffic_conf, intersection=self.intersection
        )
        detail = {
            "ET": {"queue_len": 50, "avg_wait_time": 100, "cells": [0, 0, 0, 0]},
            "WT": {"queue_len": 50, "avg_wait_time": 100, "cells": [0, 0, 0, 0]},
        }
        phase, duration = agent._apply_guardrail(1, 0, detail, {})
        self.assertEqual(phase, 0)
        self.assertEqual(duration, len(agent.duration_options) - 1)
        self.assertIsNotNone(agent.last_guardrail)

    def test_chatgpt_compatible_fallback_uses_real_phase_lanes(self):
        roads = {"road": {"location": "East", "length": 100}}
        state = {
            "ET": {"queue_len": 20, "avg_wait_time": 10, "cells": [0, 0, 0, 0]},
            "WT": {"queue_len": 20, "avg_wait_time": 10, "cells": [0, 0, 0, 0]},
            "NT": {"queue_len": 1, "avg_wait_time": 1, "cells": [0, 0, 0, 0]},
            "ST": {"queue_len": 1, "avg_wait_time": 1, "cells": [0, 0, 0, 0]},
            "EL": {"queue_len": 0, "avg_wait_time": 0, "cells": [0, 0, 0, 0]},
            "WL": {"queue_len": 0, "avg_wait_time": 0, "cells": [0, 0, 0, 0]},
            "NL": {"queue_len": 0, "avg_wait_time": 0, "cells": [0, 0, 0, 0]},
            "SL": {"queue_len": 0, "avg_wait_time": 0, "cells": [0, 0, 0, 0]},
        }
        agent = chatgpt_module.ChatGPTTLCS_Commonsense(
            "Qwen/test", {"roads": roads}, "0", 4, "/tmp", "test"
        )
        with patch.dict(chatgpt_module.os.environ, {"CHATGPT_API_KEY": "", "CHATGPT_FALLBACK": "1"}), \
             patch.object(chatgpt_module, "get_state_detail", return_value=(state, {}, 5.0)):
            agent.choose_action(object())
        self.assertEqual(agent.temp_action_logger, 0)
        self.assertEqual(agent.fallback_count, 1)


if __name__ == "__main__":
    unittest.main()
