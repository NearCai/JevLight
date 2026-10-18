"""Run the available SUMO scenarios with matched seeds and fixed horizons.

Planning is free. --execute runs the selected method; Jev images require an API
key. Mock image runs are only smoke tests and never enter the score table.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jevlight.config import DATASETS
from scripts.fetch_vlalight_renderer import REVISION as RENDERER_REVISION


def save_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def check_image_api(output, retries=8):
    from PIL import Image
    from jevlight.jev_client import JevClient

    probe_dir = ROOT / "logs" / "benchmark_vlalight"
    probe_dir.mkdir(parents=True, exist_ok=True)
    image = probe_dir / "api_preflight.jpg"
    Image.new("RGB", (768, 1440), "gray").save(image)
    result = JevClient(transport="official", model="jev-1.14.0", max_retries=retries).evaluate(
        {"camera_image": "api-preflight"},
        {"probe": {"type": "choice", "instructions": "Read api-preflight; choose wait for an empty scene.",
                   "criteria": {"go": "Vehicles can proceed", "wait": "No visible traffic"}}},
        assets={"api-preflight": image},
    )
    error = result.error
    answer = result.answers.get("probe")
    if not error and (result.model != "jev-1.14.0" or
                      not isinstance(answer, dict) or answer.get("choice") not in {"go", "wait"}):
        error = "Image preflight returned an unexpected model or invalid choice"
    record = {"checked_at": datetime.now().astimezone().isoformat(),
              "status": "blocked_api" if error else "ready", "error": error,
              "model": result.model, "input_tokens": result.input_tokens,
              "output_tokens": result.output_tokens, "latency_s": result.latency}
    save_json(output / "api_preflight.json", record)
    return error


def build_cases():
    cases = []
    for dataset, config in DATASETS.items():
        data = ROOT / "data" / config["template"] / config["road_net"]
        if not (data / config["roadnet_file"]).is_file():
            continue
        for filename in config["traffic_files"]:
            path = data / filename
            if not path.is_file():
                continue
            # Cover the entire departure schedule, rounded up to 5 minutes.
            last_depart = 0.0
            for _, elem in ET.iterparse(path, events=("end",)):
                if elem.tag in {"vehicle", "trip", "flow"}:
                    value = elem.get("end", elem.get("depart", elem.get("begin", "0")))
                    last_depart = max(last_depart, float(value))
                elem.clear()
            horizon = max(3600, math.ceil((last_depart + 1) / 300) * 300)
            cases.append({
                "id": f"{dataset}/{path.name.removesuffix('.rou.xml')}",
                "dataset": dataset, "traffic_file": filename,
                "intersections": config["num_intersections"], "horizon_s": horizon,
                "traffic_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "roadnet_sha256": hashlib.sha256((data / config["roadnet_file"]).read_bytes()).hexdigest(),
                "image_requests_per_seed": config["num_intersections"] * math.ceil((horizon - 1) / 15),
            })
    return cases


def validate_summary(summary, case, seed, method):
    for field in ("dataset", "traffic_file", "traffic_sha256", "roadnet_sha256"):
        if summary.get(field) != case[field]:
            return "invalid_metadata"
    if summary.get("seed") != seed or summary.get("control_interval_s") != 15:
        return "invalid_metadata"
    if summary.get("final/end_time") != case["horizon_s"]:
        return "incomplete_horizon"
    if method == "maxpressure":
        return "complete" if (summary.get("dry_run") and
                              summary.get("agent_mode") == "maxpressure" and
                              not summary.get("image_decisions")) else "invalid_method"
    if not summary.get("image_decisions") or summary.get("image_renderer") != "vlalight":
        return "invalid_method"
    if (summary.get("jev_transport") != "official" or
            summary.get("jev_model") != "jev-1.14.0" or
            summary.get("renderer_metadata", {}).get("revision") != RENDERER_REVISION):
        return "invalid_method"
    expected_mock = method == "smoke"
    if bool(summary.get("mock_jev")) != expected_mock:
        return "invalid_method"
    if summary.get("jev", {}).get("requests") != case["image_requests_per_seed"]:
        return "incomplete_requests"
    if summary.get("jev", {}).get("failures") != 0 or summary.get("fallback_count") != 0:
        return "degraded_fallbacks"
    return "smoke_only" if expected_mock else "complete"


def aggregate(runs):
    groups = {}
    for run in runs:
        if run["status"] == "complete":
            groups.setdefault(run["case"], []).append(run["summary"])
    result = {}
    for name, summaries in groups.items():
        fields = ("avg_travel_time", "avg_waiting_time", "avg_queue_len",
                  "arrived_vehicle_count", "running_vehicle_count", "pending_vehicle_count")
        result[name] = {"seeds": [s["seed"] for s in summaries]}
        for field in fields:
            values = [s[f"final/{field}"] for s in summaries]
            result[name][field] = {"mean": statistics.mean(values),
                                   "sample_std": statistics.stdev(values) if len(values) > 1 else None}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("maxpressure", "jev-image", "smoke"), default="jev-image")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--output", type=Path, default=Path("results/vlalight_jev"))
    parser.add_argument("--workers", type=int, default=1, choices=(1, 2))
    parser.add_argument("--vlalight_root", default="logs/vlalight-source/VLALight")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--jev_retries", type=int, default=8)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    if args.jev_retries < 0:
        parser.error("jev_retries must be nonnegative")
    if args.method != "maxpressure" and args.workers != 1:
        parser.error("image experiments run serially to bound GPU memory and API load")
    cases = build_cases()
    if args.datasets:
        available = {c["dataset"] for c in cases}
        if set(args.datasets) - available:
            parser.error(f"unavailable datasets: {set(args.datasets) - available}")
        cases = [c for c in cases if c["dataset"] in args.datasets]
    if args.method == "smoke":
        # One short route per network is enough to exercise all camera batches.
        first = {}
        for case in cases:
            first.setdefault(case["dataset"], case)
        cases = [dict(c, horizon_s=30, image_requests_per_seed=c["intersections"] * 2)
                 for c in first.values()]
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    plan = {"method": args.method, "seeds": args.seeds, "control_interval_s": 15,
            "yellow_time_s": 5, "cases": cases,
            "jev_retries": args.jev_retries if args.method == "jev-image" else 0,
            "estimated_image_requests": sum(c["image_requests_per_seed"] for c in cases) * len(args.seeds),
            "protocol": "fixed departure horizon; no clearance; unfinished trips are censored",
            "execute": args.execute}
    save_json(args.output / f"plan_{args.method}.json", plan)
    print(f"{len(cases)} scenarios x {len(args.seeds)} seeds; image requests: {plan['estimated_image_requests']:,}", flush=True)
    if not args.execute:
        return
    if args.method == "jev-image" and not (os.getenv("TYPESAFE_API_KEY") or os.getenv("JEV_API_KEY")):
        parser.error("TypeSafe API credentials missing; plan saved, no paid calls made")
    if args.method == "jev-image":
        error = check_image_api(args.output, retries=args.jev_retries)
        if error:
            parser.error(f"Image API preflight failed; existing results preserved: {error}")

    def execute(item):
        case, seed = item
        memo = f"VLALightBenchmark/{args.method}/{case['id']}/seed_{seed}"
        record_dir = ROOT / "records" / memo
        previous = sorted(record_dir.glob("*/summary.json"), key=lambda p: p.stat().st_mtime)
        if args.resume:
            for path in reversed(previous):
                try:
                    summary = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    print(f"Ignoring unreadable summary: {path}", flush=True)
                    continue
                status = validate_summary(summary, case, seed, args.method)
                if status in {"complete", "smoke_only"}:
                    print(f"Reusing {case['id']} seed {seed}: {status}", flush=True)
                    return {"case": case["id"], "seed": seed, "status": status,
                            "summary_path": str(path.relative_to(ROOT)), "summary": summary}
        command = [sys.executable, "run_jevlight.py", "--dataset", case["dataset"],
                   "--traffic_file", case["traffic_file"], "--count", str(case["horizon_s"]),
                   "--seed", str(seed), "--duration", "15", "--rolling_evaluation", "--memo", memo]
        if args.method == "maxpressure":
            command += ["--dry_run"]
        else:
            command += ["--jev_transport", "official", "--image_decisions", "--image_renderer", "vlalight",
                        "--vlalight_root", args.vlalight_root, "--jev_model", "jev-1.14.0",
                        "--jev_retries", str(args.jev_retries),
                        "--fail_on_image_fallback"]
            if args.method == "smoke":
                command += ["--mock_jev"]
        log_dir = ROOT / "logs" / "benchmark_vlalight" / args.method / case["dataset"]
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
        log = log_dir / f"{case['traffic_file']}.seed_{seed}.{stamp}.log"
        print(f"Running {args.method}: {case['id']} seed {seed}", flush=True)
        with log.open("w", encoding="utf-8") as output:
            completed = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
        paths = sorted(record_dir.glob("*/summary.json"), key=lambda p: p.stat().st_mtime)
        result = {"case": case["id"], "seed": seed, "status": "failed", "log": str(log.relative_to(ROOT))}
        new_paths = [p for p in paths if p not in previous]
        if completed.returncode == 0 and new_paths:
            path = new_paths[-1]
            summary = json.loads(path.read_text(encoding="utf-8"))
            result.update(status=validate_summary(summary, case, seed, args.method),
                          summary_path=str(path.relative_to(ROOT)), summary=summary)
        print(f"{case['id']} seed {seed}: {result['status']}", flush=True)
        return result

    runs = []
    output_path = args.output / f"{args.method}.json"
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(execute, [(c, seed) for c in cases for seed in args.seeds]):
            runs.append(result)
            save_json(output_path, {"plan": plan, "runs": runs, "aggregate": aggregate(runs)})
    print(output_path)
    if any(r["status"] not in {"complete", "smoke_only"} for r in runs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
