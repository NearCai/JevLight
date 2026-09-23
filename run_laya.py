"""Run the local Laya typed-decision model with CityFlow."""

import argparse
import os
import time


DATASETS = {
    "jinan": ("3_4", "Jinan"),
    "hangzhou": ("4_4", "Hangzhou"),
    "newyork_28x7": ("28_7", "NewYork"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run local Laya phase/duration control")
    parser.add_argument("--dataset", choices=DATASETS, default="jinan")
    parser.add_argument("--traffic_file", default="anon_3_4_jinan_real.json")
    parser.add_argument("--proj_name", default="jev-TSCS")
    parser.add_argument("--memo", default="Laya")
    parser.add_argument("--run_counts", type=int, default=3600)
    parser.add_argument("--model", default="convaiinnovations/laya-typed-decisions")
    parser.add_argument("--device", default=os.getenv("LAYA_DEVICE") or None,
                        help="torch device (auto by default; e.g. cuda or cpu)")
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--min_confidence", type=float, default=0.0,
                        help="fallback when the least confident answer is below this value")
    parser.add_argument("--max_concurrency", type=int, default=8,
                        help="maximum simultaneous intersection decision tasks")
    parser.add_argument("--no_guardrail", action="store_true",
                        help="disable local backlog guardrail")
    return parser.parse_args()


def main(args):
    # Import Transformers before CityFlow's native extension.  On this CUDA
    # build, loading Triton lazily after CityFlow can segfault during module
    # initialization; importing Laya first keeps the native libraries isolated.
    import transformers  # noqa: F401
    import laya  # noqa: F401
    from models.laya_agent import LayaAgent
    # Materialize Transformers/Triton before the CityFlow native extension is
    # imported (see the import-order note above).
    LayaAgent._ensure_model(args.model, args.device)
    warmup_error = LayaAgent.warmup(args.model, args.device)
    if isinstance(warmup_error, str):
        print(f"Laya warmup warning: {warmup_error}")

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
        "LAYA_MODEL": args.model,
        "LAYA_DEVICE": args.device,
        "LAYA_MAX_LEN": args.max_len,
        "LAYA_GUARDRAIL": not args.no_guardrail,
        "LAYA_MIN_CONFIDENCE": args.min_confidence,
    }
    traffic_conf = {
        "MODEL_NAME": "Laya",
        "MODEL": "Laya",
        "PROJECT_NAME": args.proj_name,
        "RUN_COUNTS": args.run_counts,
        "NUM_ROW": rows,
        "NUM_COL": cols,
        "NUM_AGENTS": num_intersections,
        "NUM_INTERSECTIONS": num_intersections,
        "TRAFFIC_FILE": args.traffic_file,
        "ROADNET_FILE": roadnet_file,
        "ACTION_PATTERN": "set",
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
        "JEV_MAX_CONCURRENCY": args.max_concurrency,
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
