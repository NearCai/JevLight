"""Run Jev traffic-signal control with CityFlow.

Jev makes two structured decisions at each due intersection: a phase and a
phase duration.  The simulator advances in five-second ticks; phase changes
use the existing five-second yellow transition in ``CityFlowEnv``.
"""

import argparse
import os
import time


DATASETS = {
    "jinan": ("3_4", "Jinan"),
    "hangzhou": ("4_4", "Hangzhou"),
    "newyork_28x7": ("28_7", "NewYork"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run Jev phase/duration control")
    parser.add_argument("--dataset", choices=DATASETS, default="jinan")
    parser.add_argument("--traffic_file", default="anon_3_4_jinan_real.json")
    parser.add_argument("--proj_name", default="jev-TSCS")
    parser.add_argument("--memo", default="Jev")
    parser.add_argument("--run_counts", type=int, default=3600)
    parser.add_argument("--jev_model", default="jev-latest")
    parser.add_argument("--api_key", default=os.getenv("TYPESAFE_API_KEY", ""))
    parser.add_argument("--endpoint", default="https://api.typesafe.ai/v1/systemone")
    parser.add_argument("--phase_mode", choices=("choice", "score", "noul"), default="choice")
    parser.add_argument("--duration_mode", choices=("choice", "score", "noul"), default="choice")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max_concurrency", type=int, default=8,
                        help="maximum simultaneous Jev intersection requests")
    parser.add_argument("--no_fallback", action="store_true",
                        help="fail on API errors instead of using the local fallback")
    parser.add_argument("--no_guardrail", action="store_true",
                        help="trust valid Jev actions without the backlog guardrail")
    return parser.parse_args()


def main(args):
    road_net, template = DATASETS[args.dataset]
    rows, cols = map(int, road_net.split("_"))
    num_intersections = rows * cols
    data_path = os.path.join("data", template, road_net)

    traffic_path = os.path.join(data_path, args.traffic_file)
    roadnet_file = f"roadnet_{road_net}.json"
    roadnet_path = os.path.join(data_path, roadnet_file)
    for path in (traffic_path, roadnet_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"CityFlow input not found: {path}")

    from utils import config
    from utils.utils import merge, oneline_wrapper

    timestamp = time.strftime("%m_%d_%H_%M_%S")
    agent_conf = {
        "JEV_API_KEY": args.api_key,  # intentionally empty until the user fills it
        "JEV_ENDPOINT": args.endpoint,
        "JEV_MODEL": args.jev_model,
        "JEV_PHASE_MODE": args.phase_mode,
        "JEV_DURATION_MODE": args.duration_mode,
        "JEV_TIMEOUT": args.timeout,
        "JEV_MAX_CONCURRENCY": args.max_concurrency,
        "JEV_FALLBACK": not args.no_fallback,
        "JEV_GUARDRAIL": not args.no_guardrail,
    }
    traffic_conf = {
        "MODEL_NAME": "Jev",
        "MODEL": "Jev",
        "PROJECT_NAME": args.proj_name,
        "RUN_COUNTS": args.run_counts,
        "NUM_ROW": rows,
        "NUM_COL": cols,
        "NUM_AGENTS": num_intersections,
        "NUM_INTERSECTIONS": num_intersections,
        "TRAFFIC_FILE": args.traffic_file,
        "ROADNET_FILE": roadnet_file,
        "ACTION_PATTERN": "set",
        # Retained for legacy config readers; Jev uses DECISION_INTERVAL.
        "MIN_ACTION_TIME": 10,
        "MIN_ACTION_TIME1": 5,
        "DECISION_INTERVAL": 5,
        "YELLOW_TIME": 5,
        "DURATION_EXCLUDES_YELLOW": True,
        "ACTION_DURATION": {i: seconds for i, seconds in enumerate((15, 20, 25, 30, 35, 40))},
        "LIST_STATE_FEATURE": ["cur_phase", "time_this_phase", "traffic_movement_pressure_queue"],
        "DIC_REWARD_INFO": {"queue_length": -0.25},
        "LIST_MODEL_NEED_TO_UPDATE": [],
        "NUM_PHASES": 4,
    }
    paths = {
        "PATH_TO_MODEL": os.path.join("model", args.memo, f"{args.traffic_file}_{timestamp}"),
        "PATH_TO_WORK_DIRECTORY": os.path.join("records", args.memo, f"{args.traffic_file}_{timestamp}"),
        "PATH_TO_DATA": data_path,
    }

    oneline_wrapper(
        dic_agent_conf=agent_conf,
        dic_traffic_env_conf=merge(config.dic_traffic_env_conf, traffic_conf),
        dic_path=merge(config.DIC_PATH, paths),
        roadnet=f"{template}-{road_net}",
        trafficflow=args.traffic_file.rsplit(".", 1)[0],
    )


if __name__ == "__main__":
    main(parse_args())
