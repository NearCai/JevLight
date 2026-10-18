"""Run JevLight signal control on the SUMO stack.

Control flow mirrors ChatLight's ``run_LLMLight.py``: collect observations,
decide one action per intersection, step SUMO, log traces and metrics.  The
decision model is Jev — the community MCP endpoint (default, ``$JEV_API_KEY``),
the official TypeSafe API (``$TYPESAFE_API_KEY``), or a local Jev-compatible
server serving open Jev-style models (``--jev_transport local``, started via
``scripts/serve_jev.sh``).  ``--agent`` selects the prompt and request
pattern: ``jevlight`` (per-intersection agent, LLMLight prompt) or
``cojevlight`` (per-intersection agent with direct neighbours, default).
See ``jevlight/controller.py`` for the question design.

Examples:
    # Mock transport, no API key needed (pipeline smoke test):
    python run_jevlight.py --mock_jev --count 60

    # Community MCP endpoint (default transport):
    export JEV_API_KEY=jev_...
    python run_jevlight.py --dataset jinan --count 900

    # Official TypeSafe API:
    export TYPESAFE_API_KEY=...
    python run_jevlight.py --jev_transport official --dataset jinan --count 900

    # Open Jev-style model served locally (see scripts/serve_jev.sh):
    scripts/serve_jev.sh --model tev1          # in another shell / GPU node
    python run_jevlight.py --jev_transport local --dataset jinan --count 900

    # ChatLight baseline request patterns on the same Jev interface:
    python run_jevlight.py --agent jevlight   # per-intersection agent
    python run_jevlight.py --agent cojevlight # target and direct neighbours (default)

    # MaxPressure baseline, no Jev calls at all:
    python run_jevlight.py --dry_run --count 3600

    # Watch the experiment in 3D (serves viz/, opens the browser):
    python run_jevlight.py --mock_jev --count 60 --visualize
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
import hashlib
import json
import os
import platform
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from tqdm import tqdm

from jevlight.config import (
    DATASETS,
    DIC_PATH,
    EIGHT_PHASE_LIST,
    EIGHT_PHASE_PHASES,
    dic_traffic_env_conf,
    setup_dataset_config,
)
from jevlight.controller import (
    AGENT_MODES,
    AGENT_MODE_ALIASES,
    AGENT_MODE_DEFAULTS,
    JEV_FALLBACKS,
    JEV_PACKAGINGS,
    JevLightController,
    MaxPressureController,
)
from jevlight.env_bridge import JevLightEnv
from jevlight.evaluation_metrics import calculate_evaluation_metrics_from_env
from jevlight.images import ImageRenderer
from jevlight.jev_client import (
    DEFAULT_JEV_MODEL,
    DEFAULT_LOCAL_BASE_URL,
    DEFAULT_MCP_BASE_URL,
    DEFAULT_OFFICIAL_BASE_URL,
    JEV_TRANSPORTS,
    JevClient,
    resolve_jev_api_key,
)
from jevlight.observation import collect_observations
from jevlight.rolling_evaluation import RollingMetricRecorder
from jevlight.viz_server import (
    DEFAULT_VIZ_HOST,
    DEFAULT_VIZ_PORT,
    NetworkModel,
    PassiveRuntime,
    open_viz_page,
    start_viz_server,
)
from jevlight.wandb_utils import safe_wandb_finish, safe_wandb_log, wandb_init_if_enabled


def parse_bool(value):
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise argparse.ArgumentTypeError("expected true or false")


def parse_args(argv=None):
    packaging_defaults = ", ".join(
        f"{mode}={defaults['packaging']}"
        for mode, defaults in AGENT_MODE_DEFAULTS.items()
    )
    parser = argparse.ArgumentParser(
        description="Run JevLight signal control with a Jev decision model"
    )
    parser.add_argument("--config", help="JSON experiment parameters in config/<model>/<experiment>.json")
    parser.add_argument("--print_config", action="store_true", help="Print resolved parameters without running an experiment")
    parser.add_argument(
        "--jev_transport",
        choices=JEV_TRANSPORTS,
        default="mcp",
        help=(
            "mcp: community server at www.jevai.org/api/mcp (default, uses "
            "$JEV_API_KEY); official: TypeSafe System One API (uses "
            "$TYPESAFE_API_KEY); local: self-hosted Jev-compatible server "
            f"({DEFAULT_LOCAL_BASE_URL} or $JEV_LOCAL_BASE_URL, no key; "
            "start one with scripts/serve_jev.sh)"
        ),
    )
    parser.add_argument(
        "--jev_model",
        default=None,
        help=(
            "Model ID passed through to the endpoint. Default: omit the "
            "field so the server chooses (required for the community MCP "
            "endpoint, which rejects official aliases like jev-latest and "
            "expects identifiers such as typesafe-ai/jev); for the official "
            f"transport the default becomes {DEFAULT_JEV_MODEL}"
        ),
    )
    parser.add_argument(
        "--jev_api_key",
        default=None,
        help="Jev API key; defaults to $JEV_API_KEY / $TYPESAFE_API_KEY",
    )
    parser.add_argument(
        "--jev_base_url",
        default=None,
        help=(
            "Override the endpoint URL (defaults: "
            f"{DEFAULT_MCP_BASE_URL} for mcp, "
            f"{DEFAULT_OFFICIAL_BASE_URL} for official, "
            f"{DEFAULT_LOCAL_BASE_URL} or $JEV_LOCAL_BASE_URL for local)"
        ),
    )
    parser.add_argument(
        "--jev_timeout",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--jev_retries",
        type=int,
        default=3,
        help="Retry attempts on rate-limit/network errors (default: 3)",
    )
    parser.add_argument(
        "--jev_min_interval",
        type=float,
        default=None,
        help=(
            "Minimum seconds between Jev requests (pacing for the burst "
            "quota on the community endpoint). Default: 3s for mcp, 0 for "
            "official"
        ),
    )
    parser.add_argument(
        "--agent",
        choices=AGENT_MODES + tuple(AGENT_MODE_ALIASES),
        default="cojevlight",
        help=(
            "Prompt and request pattern: jevlight = JevLight "
            "per-intersection agent prompt, per_intersection packaging, with a "
            "congestion question; cojevlight = CoJevLight target and direct "
            "neighbours, per_intersection packaging (default). llmlight and "
            "collmlight are accepted as aliases. Sets the defaults for "
            "--packaging; cojevlight requires per_intersection. Both modes include a congestion question by default"
        ),
    )
    parser.add_argument(
        "--packaging",
        choices=JEV_PACKAGINGS,
        default=None,
        help=(
            "network: one request per step with every active intersection as a "
            "parallel question; per_intersection: one request per active "
            "intersection per step, like LLMLight. Default: the --agent "
            f"mode's packaging ({packaging_defaults})"
        ),
    )
    parser.add_argument(
        "--fallback",
        choices=JEV_FALLBACKS,
        default="previous",
        help=(
            "Signal used when an answer is missing, invalid, rate-limited, or "
            "below --min_confidence: previous = keep the previous signal "
            "(source convention, default); ranking = local waiting-time ranking"
        ),
    )
    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.0,
        help=(
            "Reject Jev answers whose confidence is below this value and use "
            "the fallback signal (default: 0.0 = disabled)"
        ),
    )
    parser.add_argument(
        "--speculative", type=parse_bool, nargs="?", const=True, default=False,
        metavar="{true,false}",
        help="Include the optional congestion question (default: false in both modes)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Concurrent requests for per_intersection packaging",
    )
    parser.add_argument('--jev_batch_size', type=int, default=0,
                        help='Local native batch cap; 0 preserves individual requests')
    parser.add_argument(
        "--cojev_history_steps", type=int, default=0,
        help="Compatibility field; must be 0 (CoJevLight uses current state only)",
    )
    parser.add_argument(
        "--mock_jev",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deterministic local answers for pipeline smoke tests (no API)",
    )
    parser.add_argument(
        "--image_decisions",
        action="store_true",
        help=(
            "Enable experimental direct image-to-phase decisions on official "
            "TypeSafe. Select Three.js or VLALight cameras with --image_renderer."
        ),
    )
    parser.add_argument(
        "--image_prompt",
        default=None,
        help="Append task instructions to the direct image-to-phase prompt",
    )
    parser.add_argument("--image_renderer", choices=("threejs", "vlalight"), default="threejs")
    parser.add_argument("--vlalight_root", default="logs/vlalight-source/VLALight",
                        help="VLALight source directory containing TransSimHub")
    parser.add_argument("--render_batch_size", type=int, default=4)
    parser.add_argument("--fail_on_image_fallback", action="store_true",
                        help="Stop an image benchmark after any fallback; preserve the trace")
    parser.add_argument(
        "--dry_run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "MaxPressure baseline: signals chosen by local argmax-pressure "
            "(queued + approaching vehicles per phase), no Jev requests at "
            "all — no API key or transport needed"
        ),
    )
    parser.add_argument("--memo", default="JevLight")
    parser.add_argument("--proj_name", default="JevLight")
    parser.add_argument(
        "--dataset",
        default="jinan",
        help=f"Dataset key; registered: {', '.join(sorted(DATASETS))}",
    )
    parser.add_argument("--traffic_file", default="anon_3_4_jinan_real.rou.xml")
    parser.add_argument("--road_net", default=None)
    parser.add_argument("--roadnet_file", default=None)
    parser.add_argument("--num_intersections", type=int, default=None)
    parser.add_argument("--duration", type=int, default=15)
    parser.add_argument("--count", type=int, default=3600)
    parser.add_argument("--start_time", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eightphase", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--use_wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument(
        "--rolling_evaluation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Emit canonical hourly_results/* curves and one complete final/* result",
    )
    parser.add_argument("--hourly_interval", type=int, default=3600)
    parser.add_argument("--evaluation_output_dir", default=None)
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Serve the 3D visualizer (viz/) on this run's own simulation and "
            "open it in the browser; the page's play/pause/step buttons gate "
            "the experiment loop"
        ),
    )
    parser.add_argument(
        "--viz_host", default=DEFAULT_VIZ_HOST, help="Visualizer bind host"
    )
    parser.add_argument(
        "--viz_port",
        type=int,
        default=DEFAULT_VIZ_PORT,
        help=f"Visualizer port (default {DEFAULT_VIZ_PORT}; next free port used if taken)",
    )
    parser.add_argument(
        "--viz_hold",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep serving the final 3D state after the run until Ctrl-C",
    )
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config")
    config_path = config_parser.parse_known_args(argv)[0].config
    if config_path:
        try:
            defaults = load_experiment_defaults(parser, config_path)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if args.jev_batch_size < 0:
        parser.error('--jev_batch_size must be nonnegative')
    if args.jev_batch_size and args.jev_transport != 'local':
        parser.error('--jev_batch_size requires local transport')
    if args.jev_batch_size and args.packaging == 'network':
        parser.error('--jev_batch_size requires per_intersection packaging')
    if args.cojev_history_steps != 0:
        parser.error("--cojev_history_steps must be 0: current state only")
    if AGENT_MODE_ALIASES.get(args.agent, args.agent) == "cojevlight":
        if args.packaging not in (None, "per_intersection"):
            parser.error("cojevlight requires --packaging per_intersection (target and direct neighbours only)")
        args.packaging = "per_intersection"
    return args


def load_experiment_defaults(parser, filename):
    """Validate file defaults; explicit CLI flags retain argparse precedence."""
    path = Path(filename).resolve()
    config_root = Path(__file__).resolve().parent / "config"
    if path.parent.parent != config_root or path.suffix != ".json":
        raise ValueError("Experiment configs must be config/<model>/<experiment>.json")
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {filename}: {exc}") from exc
    if not isinstance(values, dict):
        raise ValueError("Experiment config must be a JSON object")
    actions = {a.dest: a for a in parser._actions}
    forbidden = {"help", "config", "print_config", "jev_api_key"}
    for key, value in values.items():
        if key not in actions or key in forbidden:
            raise ValueError(f"Unknown or prohibited config parameter: {key}")
        action = actions[key]
        if value is None and action.default is None:
            continue
        if action.type is parse_bool or isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse.BooleanOptionalAction)):
            valid = type(value) is bool
        elif action.type is int:
            valid = type(value) is int
        elif action.type is float:
            valid = type(value) in (int, float)
        else:
            valid = isinstance(value, str)
        if not valid:
            raise ValueError(f"Invalid type for config parameter {key}: {value!r}")
        if action.choices is not None and value not in action.choices:
            raise ValueError(f"Invalid value for config parameter {key}: {value!r}")
    return values


def experiment_parameters(args):
    """Reproducible resolved parameters without credentials or preflight flags."""
    return {key: value for key, value in vars(args).items()
            if key not in {"jev_api_key", "print_config"}}


def build_configs(args, dataset_config, run_id):
    num_intersections = args.num_intersections or dataset_config["num_intersections"]
    phase = dic_traffic_env_conf["PHASE"]
    phase_list = dic_traffic_env_conf["PHASE_LIST"]
    if args.eightphase:
        phase = EIGHT_PHASE_PHASES
        phase_list = EIGHT_PHASE_LIST

    agent_config = {
        "POLICY": "jevlight",
        "LLM_MODEL": args.jev_model,
        "CONTROL_INTERVAL": args.duration,
    }
    env_config = copy.deepcopy(dic_traffic_env_conf)
    env_config.update({
        "NUM_AGENTS": num_intersections,
        "NUM_INTERSECTIONS": num_intersections,
        "RUN_COUNTS": args.count,
        "MIN_ACTION_TIME": args.duration,
        "MEASURE_TIME": args.duration,
        "GREEN_TIME": args.duration - 5,
        "YELLOW_TIME": 5,
        "MODEL_NAME": "jevlight",
        "MODEL": "jevlight",
        "PROJECT_NAME": args.proj_name,
        "USE_WANDB": args.use_wandb,
        "TRAFFIC_FILE": args.traffic_file,
        "ROADNET_FILE": dataset_config["roadnet_file"],
        "LIST_STATE_FEATURE": [
            "cur_phase_four",
            "lane_num_waiting_vehicle_in",
            "adjacency_matrix",
        ],
        "DIC_REWARD_INFO": {"queue_length": -0.25},
        "SEED": getattr(args, "seed", 1),
        "PHASE": phase,
        "PHASE_LIST": phase_list,
    })
    paths = copy.deepcopy(DIC_PATH)
    model_name = (Path(args.config).parent.name if getattr(args, "config", None)
                  else ("maxpressure" if args.dry_run else args.jev_model or "jev"))
    model_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in model_name)
    if model_name in {"", ".", ".."}:
        raise ValueError("model name must identify a results subdirectory")
    memo_path = Path(args.memo)
    if (
        memo_path.is_absolute()
        or not memo_path.parts
        or any(part in {"", ".", ".."} for part in memo_path.parts)
    ):
        raise ValueError("memo must be a single directory name or safe relative path")
    memo = str(memo_path)
    # Keep the VLALight benchmark's historical records/ layout so its resume
    # scanner can reuse completed seeds; portable model experiments use the
    # upstream results/<model>/<memo>/<run> layout.
    root_dir = "records" if (args.image_decisions or args.dry_run) else os.path.join("results", model_name)
    result_dir = Path(root_dir, memo, run_id).as_posix()
    if args.evaluation_output_dir and not args.image_decisions:
        output = Path(args.evaluation_output_dir).resolve()
        if not output.is_relative_to(Path("results").resolve()):
            raise ValueError("evaluation_output_dir must be inside results/")
    paths.update({
        "PATH_TO_MODEL": Path(result_dir, "model").as_posix(),
        "PATH_TO_WORK_DIRECTORY": result_dir,
        "PATH_TO_DATA": Path(
            "data", dataset_config["template"], str(dataset_config["road_net"])
        ).as_posix(),
        "PATH_TO_ERROR": Path(result_dir, "errors").as_posix(),
    })
    return agent_config, env_config, paths


def calculate_metrics(env, total_reward, namespace="final"):
    """Same metric names and aggregate definitions as ChatLight's runner."""
    metrics = calculate_evaluation_metrics_from_env(env, namespace)
    metrics[f"{namespace}/reward"] = float(total_reward)
    return metrics


def build_wandb_group(dataset_config, env_config, traffic_file):
    traffic_file_str = traffic_file.split(".")[0]
    phase_count = len(env_config.get("PHASE", {}))
    roadnet_group = f"{dataset_config['template']}-{dataset_config['road_net']}"
    return f"JevLight-{roadnet_group}-{traffic_file_str}-{phase_count}_Phases"


def run(args):
    with ExitStack() as resources:
        return _run(args, resources)


def _run(args, resources):
    wall_started = time.perf_counter()
    if args.duration <= 5:
        raise ValueError("duration must be greater than the 5-second yellow time")
    if args.count <= 0:
        raise ValueError("count must be positive")
    if args.image_decisions and args.dry_run:
        raise ValueError("--image_decisions cannot be combined with --dry_run")
    if args.image_decisions and args.jev_transport != "official":
        raise ValueError("--image_decisions requires --jev_transport official")
    if args.image_decisions:
        if args.packaging not in (None, "per_intersection") or args.fallback != "previous":
            raise ValueError("image mode requires per_intersection packaging and previous fallback")
        args.agent = "jevlight"
        args.packaging = "per_intersection"
        args.speculative = False
        args.jev_model = args.jev_model or "jev-1.14.0"

    dataset_config = setup_dataset_config(args.dataset)
    if args.road_net:
        dataset_config["road_net"] = args.road_net
    if args.roadnet_file:
        dataset_config["roadnet_file"] = args.roadnet_file
    traffic_path = os.path.join(
        "data", dataset_config["template"], str(dataset_config["road_net"]), args.traffic_file
    )
    roadnet_path = os.path.join(
        "data", dataset_config["template"], str(dataset_config["road_net"]), dataset_config["roadnet_file"]
    )
    if not os.path.isfile(traffic_path):
        raise FileNotFoundError(f"Traffic file does not exist: {traffic_path}")
    if not os.path.isfile(roadnet_path):
        raise FileNotFoundError(f"Roadnet file does not exist: {roadnet_path}")

    # Private ordered credentials are the default for official runs without an
    # explicitly supplied single key. Only the vault path enters the environment.
    if args.jev_transport == "official" and not args.mock_jev and not args.dry_run:
        from jevlight.api_key_pool import DEFAULT_VAULT, KeyPool
        pool_path = os.environ.get("JEV_API_KEY_POOL")
        if not pool_path and not args.jev_api_key and not resolve_jev_api_key() and DEFAULT_VAULT.exists():
            pool_path = str(DEFAULT_VAULT)
            os.environ["JEV_API_KEY_POOL"] = pool_path
        pool_key = KeyPool(pool_path).select()["secret"] if pool_path else None
    else:
        pool_key = None

    # The local transport must not pick a hosted key up from the
    # environment; only an explicitly passed --jev_api_key is forwarded.
    api_key = (
        args.jev_api_key
        if args.jev_transport == "local"
        else pool_key or resolve_jev_api_key(args.jev_api_key)
    )
    if (
        not args.mock_jev
        and not args.dry_run
        and args.jev_transport != "local"
        and api_key is None
    ):
        raise ValueError(
            "Jev API key missing: pass --jev_api_key or set "
            "$JEV_API_KEY / $TYPESAFE_API_KEY (or use --mock_jev, "
            "--jev_transport local for a self-hosted server, or --dry_run "
            "for the MaxPressure baseline)"
        )

    timestamp = time.strftime("%m_%d_%H_%M_%S")
    run_id = f"jevlight_{Path(args.traffic_file).stem}_{timestamp}_pid{os.getpid()}"
    agent_config, env_config, paths = build_configs(args, dataset_config, run_id)

    min_interval = args.jev_min_interval
    if min_interval is None:
        min_interval = 3.0 if args.jev_transport == "mcp" else 0.0
    args.agent = AGENT_MODE_ALIASES.get(args.agent, args.agent)
    agent_defaults = AGENT_MODE_DEFAULTS[args.agent]
    packaging = args.packaging or agent_defaults["packaging"]
    speculative = args.speculative
    if args.dry_run:
        # MaxPressure baseline: purely local decisions, no Jev client at all.
        packaging = "none"
        controller = MaxPressureController(args.duration)
    else:
        client = JevClient(
            transport=args.jev_transport,
            model=(
                (args.jev_model or DEFAULT_JEV_MODEL)
                if args.jev_transport == "official"
                else args.jev_model
            ),
            api_key=api_key,
            base_url=args.jev_base_url,
            timeout=args.jev_timeout,
            max_retries=args.jev_retries,
            min_interval=min_interval,
            mock=args.mock_jev,
            batch_size=args.jev_batch_size,
        )
        controller = JevLightController(
            client,
            args.duration,
            packaging=packaging,
            fallback=args.fallback,
            min_confidence=args.min_confidence,
            speculative=speculative,
            max_workers=args.max_workers,
            agent_mode=args.agent,
            cojev_history_steps=args.cojev_history_steps,
        )
    agent_mode = "maxpressure" if args.dry_run else args.agent

    simulator = JevLightEnv(
        agent_config,
        env_config,
        paths,
        args.traffic_file,
    )
    env = simulator.env
    resources.callback(env.end_engine)
    (Path(paths["PATH_TO_WORK_DIRECTORY"]) / "experiment_config.json").write_text(
        json.dumps(experiment_parameters(args), indent=2), encoding="utf-8"
    )
    if args.start_time:
        simulator.reset_env_with_start_time(args.start_time)

    # 3D visualization attaches to THIS simulation (same traci connection):
    # the control loop publishes one frame per step, and the page's
    # play/pause/step buttons gate the loop between steps.
    viz_runtime: Optional[PassiveRuntime] = None
    viz_server = None
    image_renderer = None
    if args.image_decisions:
        frame_dir = Path(paths["PATH_TO_WORK_DIRECTORY"]) / "vision_frames"
        if args.image_renderer == "vlalight":
            from jevlight.vlalight_images import VLALightRenderer
            image_renderer = resources.enter_context(VLALightRenderer(
                args.vlalight_root, dataset_config, roadnet_path, frame_dir,
                batch_size=args.render_batch_size,
            ))
        else:
            image_renderer = resources.enter_context(ImageRenderer(roadnet_path, frame_dir))

    class TeleportCounter:
        count = 0

        def on_simulation_second(self, env):
            self.count += env.traci_conn.simulation.getStartingTeleportNumber()

    teleport_counter = TeleportCounter()
    env.add_step_listener(teleport_counter)
    if args.visualize:
        viz_network = NetworkModel.load(roadnet_path)
        viz_conn = getattr(env, "traci_conn", None)
        viz_runtime = PassiveRuntime(
            step_length=(
                float(viz_conn.simulation.getDeltaT()) if viz_conn else 1.0
            ),
            offset=tuple(viz_network.payload["offset"]),
        )
        viz_server, viz_url = start_viz_server(
            viz_runtime,
            viz_network.payload,
            host=args.viz_host,
            port=args.viz_port,
        )
        resources.callback(viz_server.server_close)
        resources.callback(viz_server.shutdown)
        resources.callback(viz_runtime.close)
        if viz_conn:
            viz_runtime.publish(viz_conn)
        open_viz_page(viz_url)

    logger = wandb_init_if_enabled(
        env_config,
        project=args.proj_name,
        entity=args.wandb_entity,
        group=build_wandb_group(dataset_config, env_config, args.traffic_file),
        name=run_id,
        config={**experiment_parameters(args), "run_id": run_id},
    )
    if logger is not None:
        logger.define_metric("final/checkpoint")
        for metric_name in (
            "final/reward",
            "final/avg_travel_time",
            "final/avg_waiting_time",
            "final/avg_queue_len",
            "final/queuing_vehicle_num",
            "final/vehicle_count",
            "final/arrived_vehicle_count",
            "final/observed_vehicle_count",
            "final/intersection_observation_count",
            "final/queue_observation_step_count",
        ):
            logger.define_metric(metric_name, step_metric="final/checkpoint")
    trace_path = Path(paths["PATH_TO_WORK_DIRECTORY"]) / "jev_decisions.jsonl"
    rolling_recorder = None
    if args.rolling_evaluation:
        rolling_recorder = RollingMetricRecorder(
            args.evaluation_output_dir or paths["PATH_TO_WORK_DIRECTORY"],
            total_duration=args.count,
            hourly_interval=args.hourly_interval,
            logger=logger,
            metadata={
                "method": "jevlight",
                "agent_mode": agent_mode,
                "jev_transport": None if args.dry_run else args.jev_transport,
                "jev_model": None if args.dry_run else args.jev_model,
                "jev_retry_limit": None if args.dry_run else args.jev_retries,
                "packaging": packaging,
                "fallback": args.fallback,
                "min_confidence": args.min_confidence,
                "traffic_file": args.traffic_file,
                "input_mode": "image" if args.image_decisions else "text",
                "mock_jev": bool(args.mock_jev),
            },
        )
        rolling_recorder.attach(env, evaluation_start_time=args.start_time)
    total_reward = 0.0
    current_time = float(env.get_current_time())
    evaluation_start_time = current_time
    simulation_end_time = args.start_time + args.count
    if args.rolling_evaluation:
        remaining_duration = max(0, simulation_end_time - int(current_time))
        max_steps = (remaining_duration + args.duration - 1) // args.duration
    else:
        max_steps = args.count // args.duration
    step = 0
    done = False
    with trace_path.open("a", encoding="utf-8") as trace_file:
        with tqdm(
            total=max_steps * args.duration,
            desc="jevlight simulation",
            unit="s",
        ) as progress:
            while not done and step < max_steps:
                if viz_runtime is not None:
                    viz_runtime.wait_until_running()
                observations = collect_observations(env)
                if args.image_decisions:
                    image_paths = image_renderer.capture(
                        env.traci_conn,
                        [observation.intersection_id for observation in observations],
                        step,
                    )
                    actions, traces = controller.decide_images(
                        observations, image_paths, step, prompt=args.image_prompt
                    )
                else:
                    actions, traces = controller.decide(observations, step)
                step_duration = args.duration
                if args.rolling_evaluation:
                    step_duration = min(
                        args.duration,
                        max(1, int(simulation_end_time - current_time)),
                    )
                _, rewards, done, _ = env.step(actions, min_action_time=step_duration)
                total_reward += float(sum(rewards))
                if viz_runtime is not None:
                    viz_runtime.publish(env.traci_conn)
                for trace in traces:
                    trace_file.write(json.dumps(trace, ensure_ascii=False) + "\n")
                trace_file.flush()
                if args.image_decisions and args.fail_on_image_fallback and any(
                    trace.get("fallback") or trace.get("error") for trace in traces
                ):
                    raise RuntimeError("Image benchmark stopped after a failed/fallback decision; see decision trace")
                current_time = float(env.get_current_time())
                step += 1
                progress.update(step_duration)
                simulated_elapsed = current_time - evaluation_start_time
                usage = controller.usage_summary()
                print(
                    f"[SUMO-TIME] controller={agent_mode} "
                    f"step={step}/{max_steps} current={current_time:.0f}s "
                    f"elapsed={simulated_elapsed:.0f}/{max_steps * args.duration}s "
                    f"requests={usage['requests']} "
                    f"failures={usage['failures']} "
                    f"rate_limited={usage['rate_limited']} "
                    f"latency_mean_s={usage['latency_mean_s'] or 0:.3f}",
                    flush=True,
                )

    if rolling_recorder is not None:
        metrics = rolling_recorder.finalize(env)
    else:
        checkpoint_start = int(round(evaluation_start_time))
        checkpoint_end = int(round(current_time))
        metrics = calculate_metrics(env, total_reward, namespace="final")
        metrics.update({
            "final/checkpoint": 0,
            "final/start_time": checkpoint_start,
            "final/end_time": checkpoint_end,
            "final/duration": checkpoint_end - checkpoint_start,
        })
        safe_wandb_log(logger, metrics)
    safe_wandb_finish(logger)
    summary_path = Path(paths["PATH_TO_WORK_DIRECTORY"]) / "summary.json"
    fallback_count = 0
    with trace_path.open(encoding="utf-8") as traces:
        for line in traces:
            if line.strip():
                fallback_count += bool(json.loads(line).get("fallback"))
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "diff", "HEAD", "--stat"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    summary_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "dataset": args.dataset,
                "traffic_file": args.traffic_file,
                "seed": args.seed,
                "requested_horizon_s": args.count,
                "control_interval_s": args.duration,
                "roadnet_sha256": hashlib.sha256(Path(roadnet_path).read_bytes()).hexdigest(),
                "traffic_sha256": hashlib.sha256(Path(traffic_path).read_bytes()).hexdigest(),
                "git_revision": revision,
                "git_dirty": dirty,
                "python_version": platform.python_version(),
                "sumo_version": env.traci_conn.getVersion(),
                "wall_time_s": time.perf_counter() - wall_started,
                "teleport_count": teleport_counter.count,
                "fallback_count": fallback_count,
                "image_renderer": args.image_renderer if args.image_decisions else None,
                "renderer_metadata": getattr(image_renderer, "metadata", None),
                "agent_mode": agent_mode,
                "jev_transport": None if args.dry_run else args.jev_transport,
                "jev_model": None if args.dry_run else args.jev_model,
                "jev_retry_limit": None if args.dry_run else args.jev_retries,
                "packaging": packaging,
                "fallback": args.fallback,
                "min_confidence": args.min_confidence,
                "dry_run": bool(args.dry_run),
                "visualize": bool(args.visualize),
                "image_decisions": bool(args.image_decisions),
                "input_mode": "image" if args.image_decisions else "text",
                "mock_jev": bool(args.mock_jev),
                **metrics,
                "jev": controller.usage_summary(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if viz_runtime is not None and args.viz_hold:
        print("[viz] holding the final 3D state — Ctrl-C to exit", flush=True)
        try:
            while not viz_runtime.stop_event.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
    env.batch_log()
    print(json.dumps(metrics, indent=2))
    print(f"Decision log: {trace_path}")
    print(f"Summary: {summary_path}")
    return metrics


if __name__ == "__main__":
    args = parse_args()
    if args.print_config:
        print(json.dumps(experiment_parameters(args), indent=2))
    else:
        run(args)
