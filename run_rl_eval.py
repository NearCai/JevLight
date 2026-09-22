"""Run one full-horizon evaluation for a shared-network RL controller.

This intentionally evaluates a freshly initialized network (or a checkpoint
placed in ``--model_dir``) without the legacy multi-round training wrapper.
It gives every RL baseline the same CityFlow horizon and metrics as the rule
based and Jev runs.
"""

import argparse
import json
import os
import time

from utils import config
from utils.my_utils import merge
from utils.oneline import OneLine


DATA_PATH = "data/Jinan/3_4"
TRAFFIC_FILE = "anon_3_4_jinan_real.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("PressLight", "MPLight", "Colight", "DynamicLight"), required=True)
    parser.add_argument("--run_counts", type=int, default=3600)
    parser.add_argument("--traffic_file", default=TRAFFIC_FILE)
    parser.add_argument("--model_dir", default="", help="optional directory containing baseline checkpoints")
    return parser.parse_args()


def _phase():
    return {
        1: [0, 1, 0, 1, 0, 0, 0, 0],
        2: [0, 0, 0, 0, 0, 1, 0, 1],
        3: [1, 0, 1, 0, 0, 0, 0, 0],
        4: [0, 0, 0, 0, 1, 0, 1, 0],
    }


def build_config(model, traffic_file, run_counts):
    base = {
        "MODEL_NAME": model,
        "MODEL": model,
        "PROJECT_NAME": f"{model}-full-eval",
        "RUN_COUNTS": run_counts,
        "NUM_ROW": 3,
        "NUM_COL": 4,
        "NUM_AGENTS": 1,
        "NUM_INTERSECTIONS": 12,
        "TRAFFIC_FILE": traffic_file,
        "ROADNET_FILE": "roadnet_3_4.json",
        "NUM_ROUNDS": 1,
        "NUM_GENERATORS": 1,
        "LIST_MODEL_NEED_TO_UPDATE": [],
        "DIC_REWARD_INFO": {"queue_length": -0.25, "pressure": -0.25},
        "PHASE": _phase(),
        "PHASE_LIST": ["WT_ET", "NT_ST", "WL_EL", "NL_SL"],
        "ACTION_PATTERN": "set",
        "YELLOW_TIME": 5,
        "BINARY_PHASE_EXPANSION": True,
        "NUM_PHASES": 4,
        "NUM_LANE": 12,
        "TOP_K_ADJACENCY": 5,
    }
    if model == "PressLight":
        base.update({
            "LIST_STATE_FEATURE": ["cur_phase", "traffic_movement_pressure_queue"],
            "PHASE_MAP": [[1, 4, 12, 13, 14, 15, 16, 17], [7, 10, 18, 19, 20, 21, 22, 23],
                          [0, 3, 18, 19, 20, 21, 22, 23], [6, 9, 12, 13, 14, 15, 16, 17]],
        })
    elif model == "MPLight":
        base.update({
            "LIST_STATE_FEATURE": ["cur_phase", "traffic_movement_pressure_num"],
            "PHASE_MAP": [[1, 4, 12, 13, 14, 15, 16, 17], [7, 10, 18, 19, 20, 21, 22, 23],
                          [0, 3, 18, 19, 20, 21, 22, 23], [6, 9, 12, 13, 14, 15, 16, 17]],
            "list_lane_order": ["WL", "WT", "EL", "ET", "NL", "NT", "SL", "ST"],
        })
    elif model == "Colight":
        base.update({
            "LIST_STATE_FEATURE": ["cur_phase", "lane_num_vehicle", "adjacency_matrix"],
            "PHASE_MAP": [[1, 4, 12, 13, 14, 15, 16, 17], [7, 10, 18, 19, 20, 21, 22, 23],
                          [0, 3, 18, 19, 20, 21, 22, 23], [6, 9, 12, 13, 14, 15, 16, 17]],
            "list_lane_order": ["WL", "WT", "EL", "ET", "NL", "NT", "SL", "ST"],
        })
    else:
        base.update({
            "LIST_STATE_FEATURE": ["phase_total", "lane_queue_vehicle_in", "lane_run_in_part", "num_in_deg"],
            "LIST_STATE_FEATURE_1": ["phase_total", "lane_queue_vehicle_in", "lane_run_in_part", "num_in_deg"],
            "LIST_STATE_FEATURE_2": ["num_in_deg"],
            "PHASE_MAP": [[1, 4], [7, 10], [0, 3], [6, 9]],
            "PHASETOTAL": {1: [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],
                            2: [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],
                            3: [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],
                            4: [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]},
            "MAX_LANE": 16,
            "ACTION_DURATION": {0: 10, 1: 15, 2: 20, 3: 25, 4: 30, 5: 35, 6: 40},
            "NUM_LANES": [3, 3, 3, 3],
            "RUN_COUNTS2": run_counts,
            "MEASURE_TIME": 10,
            "MIN_ACTION_TIME": 15,
            "MIN_ACTION_TIME1": 5,
        })
    return merge(config.dic_traffic_env_conf, base)


def main(args):
    traffic_path = os.path.join(DATA_PATH, args.traffic_file)
    if not os.path.exists(traffic_path):
        raise FileNotFoundError(traffic_path)
    timestamp = time.strftime("%m_%d_%H_%M_%S")
    work_dir = os.path.join("records", "rl_full_eval", f"{args.model}_{timestamp}")
    model_dir = args.model_dir or os.path.join("model", "rl_full_eval", f"{args.model}_{timestamp}")
    traffic_conf = build_config(args.model, args.traffic_file, args.run_counts)
    agent_conf = merge(config.DIC_BASE_AGENT_CONF, {"CNN_layers": [[32, 32]]})
    if args.model == "DynamicLight":
        agent_conf.update({"SROUND": 0, "SROUND2": 0, "SROUND3": 0,
                           "LEARNING_RATE2": 0.0002, "LEARNING_RATE3": 0.0005,
                           "BATCH_SIZE1": agent_conf["BATCH_SIZE"],
                           "SAMPLE_SIZE1": agent_conf["SAMPLE_SIZE"]})
    paths = {
        "PATH_TO_MODEL": model_dir,
        "PATH_TO_WORK_DIRECTORY": work_dir,
        "PATH_TO_DATA": DATA_PATH,
        "PATH_TO_ERROR": os.path.join("errors", "rl_full_eval"),
    }
    result = OneLine(
        dic_agent_conf=agent_conf,
        dic_traffic_env_conf=traffic_conf,
        dic_path=paths,
        roadnet="Jinan-3_4",
        trafficflow=args.traffic_file.rsplit(".", 1)[0],
    ).train(round=0)
    print(json.dumps({"model": args.model, **result}, default=float, sort_keys=True))
    return result


if __name__ == "__main__":
    main(parse_args())
