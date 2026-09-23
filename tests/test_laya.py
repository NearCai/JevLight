import sys
import types
import unittest
from unittest.mock import Mock, patch

sys.modules.setdefault("cityflow", types.SimpleNamespace())

import models.laya_agent as laya_module


class LayaAgentTest(unittest.TestCase):
    def setUp(self):
        self.agent_conf = {
            "LAYA_MODEL": "test-model",
            "LAYA_DEVICE": "cpu",
            "LAYA_GUARDRAIL": True,
        }
        self.traffic_conf = {
            "ACTION_DURATION": {i: value for i, value in enumerate((15, 20, 25, 30, 35, 40))},
        }
        roads = {"road": {"location": "East", "type": "outgoing"}}
        self.intersection = {"roads": roads}
        laya_module.LayaAgent._model = None
        laya_module.LayaAgent._model_key = None

    def make_agent(self):
        with patch.object(laya_module.LayaAgent, "_ensure_model"):
            return laya_module.LayaAgent(
                self.agent_conf,
                self.traffic_conf,
                intersection=self.intersection,
                inter_name="0",
            )

    def test_choice_and_duration_are_mapped_to_cityflow_indices(self):
        agent = self.make_agent()
        model = Mock()
        model.system_one.return_value = {
            "answers": {
                "phase": {"choice": "phase_3"},
                "duration": {"choice": "duration_35"},
            }
        }
        detail = {
            "ET": {"queue_len": 0, "cells": [0, 0, 0, 0]},
            "WT": {"queue_len": 0, "cells": [0, 0, 0, 0]},
            "NT": {"queue_len": 0, "cells": [0, 0, 0, 0]},
            "ST": {"queue_len": 0, "cells": [0, 0, 0, 0]},
            "EL": {"queue_len": 0, "cells": [0, 0, 0, 0]},
            "WL": {"queue_len": 0, "cells": [0, 0, 0, 0]},
            "NL": {"queue_len": 0, "cells": [0, 0, 0, 0]},
            "SL": {"queue_len": 0, "cells": [0, 0, 0, 0]},
        }
        with patch.object(agent, "_ensure_model", return_value=model), patch.object(
            laya_module, "get_state_detail", return_value=(detail, {}, 4.0)
        ):
            self.assertEqual(agent.choose_action(object(), {"cur_phase": [1]}, 7), (2, 4))

        model.system_one.assert_called_once()
        state, questions = model.system_one.call_args.args
        self.assertIn("phase_metrics", state)
        self.assertEqual(list(questions["phase"]["criteria"]), ["phase_1", "phase_2", "phase_3", "phase_4"])
        self.assertEqual(len(questions["duration"]["criteria"]), 6)

    def test_guardrail_corrects_phase_against_large_queue(self):
        agent = self.make_agent()
        detail = {
            "ET": {"queue_len": 40, "avg_wait_time": 30, "cells": [0, 0, 0, 0]},
            "WT": {"queue_len": 40, "avg_wait_time": 30, "cells": [0, 0, 0, 0]},
            "NT": {"queue_len": 0, "cells": [0, 0, 0, 0]},
            "ST": {"queue_len": 0, "cells": [0, 0, 0, 0]},
            "EL": {"queue_len": 0, "cells": [0, 0, 0, 0]},
            "WL": {"queue_len": 0, "cells": [0, 0, 0, 0]},
            "NL": {"queue_len": 0, "cells": [0, 0, 0, 0]},
            "SL": {"queue_len": 0, "cells": [0, 0, 0, 0]},
        }
        self.assertEqual(agent._apply_guardrail(1, 0, detail), (0, 5))
        self.assertEqual(agent.last_guardrail["guardrail_phase"], 0)

    def test_model_failure_returns_local_backlog_fallback(self):
        agent = self.make_agent()
        detail = {
            lane: {"queue_len": 0, "cells": [0, 0, 0, 0]}
            for lane in ("ET", "WT", "NT", "ST", "EL", "WL", "NL", "SL")
        }
        detail["NT"]["queue_len"] = 3
        model = Mock()
        model.system_one.side_effect = RuntimeError("inference failed")
        with patch.object(agent, "_ensure_model", return_value=model), patch.object(
            laya_module, "get_state_detail", return_value=(detail, {}, 0.0)
        ):
            self.assertEqual(agent.choose_action(object()), (1, 0))
        self.assertIn("inference failed", agent.last_error)

    def test_low_confidence_can_be_routed_to_fallback(self):
        conf = dict(self.agent_conf, LAYA_MIN_CONFIDENCE=0.6)
        with patch.object(laya_module.LayaAgent, "_ensure_model"):
            agent = laya_module.LayaAgent(
                conf, self.traffic_conf, intersection=self.intersection, inter_name="0"
            )
        detail = {
            lane: {"queue_len": 0, "cells": [0, 0, 0, 0]}
            for lane in ("ET", "WT", "NT", "ST", "EL", "WL", "NL", "SL")
        }
        model = Mock()
        model.system_one.return_value = {
            "answers": {
                "phase": {"choice": "phase_1", "confidence": 0.9, "probabilities": {}},
                "duration": {"choice": "duration_20", "confidence": 0.2, "probabilities": {}},
            }
        }
        with patch.object(agent, "_ensure_model", return_value=model), patch.object(
            laya_module, "get_state_detail", return_value=(detail, {}, 0.0)
        ):
            self.assertEqual(agent.choose_action(object()), (0, 0))
        self.assertIn("below threshold", agent.last_error)


if __name__ == "__main__":
    unittest.main()
