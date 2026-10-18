"""Write the local benchmark report from validated, completed seed summaries."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_vlalight_jev import aggregate, validate_summary


def read_result(directory, method):
    path = directory / f"{method}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def checked_runs(result):
    if not result:
        return []
    cases = {case["id"]: case for case in result["plan"]["cases"]}
    runs = []
    for run in result["runs"]:
        status = "failed"
        if "summary" in run:
            status = validate_summary(run["summary"], cases[run["case"]],
                                      run["seed"], result["plan"]["method"])
        runs.append({**run, "status": status})
    return runs


def score_table(result, plan):
    runs = checked_runs(result)
    scores = aggregate(runs)
    lines = ["| Scenario | Horizon (s) | Complete seeds | Travel (s) | Waiting (s) | Network queue |",
             "| --- | ---: | --- | ---: | ---: | ---: |"]
    for case in plan["cases"]:
        score = scores.get(case["id"])
        if not score:
            values, seeds = ["pending"] * 3, "0/" + str(len(plan["seeds"]))
        else:
            values = []
            for name in ("avg_travel_time", "avg_waiting_time", "avg_queue_len"):
                metric = score[name]
                values.append(f"{metric['mean']:.2f}" +
                              (f" ± {metric['sample_std']:.2f}" if metric["sample_std"] is not None else ""))
            seeds = f"{len(score['seeds'])}/{len(plan['seeds'])} ({','.join(map(str, score['seeds']))})"
        lines.append(f"| `{case['id']}` | {case['horizon_s']} | {seeds} | {' | '.join(values)} |")
    return "\n".join(lines)


def write_report(directory):
    plan = json.loads((directory / "plan_jev-image.json").read_text(encoding="utf-8"))
    baseline = read_result(directory, "maxpressure")
    jev = read_result(directory, "jev-image")
    smoke = read_result(directory, "smoke")
    expected = len(plan["cases"]) * len(plan["seeds"])
    baseline_runs, jev_runs = checked_runs(baseline), checked_runs(jev)
    counts = [sum(r["status"] == "complete" for r in runs) for runs in (baseline_runs, jev_runs)]
    complete = all(count == expected for count in counts)
    api = read_result(directory, "api_preflight")
    status = "complete" if complete else "in progress / incomplete"
    if not complete and api and api.get("status") == "blocked_api":
        status = "blocked: image API unavailable"
    lines = ["# VLALight cameras + Jev: local evaluation",
             "", f"Updated: {datetime.now().astimezone().isoformat(timespec='seconds')}", "",
             f"Status: **{status}**. "
             f"MaxPressure: **{counts[0]}/{expected}** completed seeds; "
             f"real Jev: **{counts[1]}/{expected}** completed seeds.", "",
             "A row with fewer than three seeds is preliminary. Pending cells are not model scores. "
             "Failed, fallback and mock runs are excluded. No conclusion about Jev improving over "
             "the baseline is supported until matched scenarios and seeds complete.", "",
             "## Protocol", "",
             "- Four supplied SUMO networks, ten route files, seeds 1/2/3; 15-second control "
             "interval and five seconds of yellow on a phase change.",
             "- Horizons cover the departure schedule, rounded up to five minutes, with a "
             "one-hour minimum. Jinan synthetic spans 24 hours; New York 28x7 spans 3,900 seconds. "
             "No clearance period is added; vehicles still travelling at the horizon are censored.",
             "- This repository's `MaxPressureController` maximizes incoming queued + approaching "
             "vehicle count. It does not subtract downstream pressure. It uses the same SUMO "
             "data, control timing and metric recorder as the Jev path, without rendering or API calls.",
             "- Real method: official `jev-1.14.0`, VLALight/TransSimHub/Panda3D cameras, "
             "one labelled 768x1440 JPEG + one prompt per intersection per boundary. No "
             "numeric traffic observations, perception agent or fast/slow reasoning are sent to Jev.",
             "- Means and sample standard deviations are over completed seeds. Travel time "
             "includes completed trips and elapsed time of active trips; it is not a completed-trip-only "
             "mean. Waiting is mean accumulated stopped time of observed vehicles. Network queue "
             "is the per-second total stopped queue averaged across time, not per intersection.",
             "- SUMO's existing teleport settings are retained. Teleports and unfinished vehicles "
             "are reported below because a low travel-time value alone can hide congestion. "
             "`min_expected` includes running and not-yet-inserted vehicles; do not add it to running.", "",
             "## Baseline results", "", score_table(baseline, plan), "",
             "## Real Jev results", "", score_table(jev, plan), "",
             "## Traffic completion diagnostics", "",
             "Values are means over completed seeds; counts may therefore be fractional.", "",
             "| Method / scenario | Seeds | Arrived | Running | Min expected | Teleports |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for method, runs in (("MaxPressure", baseline_runs), ("Jev", jev_runs)):
        for case in plan["cases"]:
            summaries = [r["summary"] for r in runs if r["case"] == case["id"] and r["status"] == "complete"]
            if summaries:
                values = [statistics.mean(s[field] for s in summaries) for field in
                          ("final/arrived_vehicle_count", "final/running_vehicle_count",
                           "final/pending_vehicle_count", "teleport_count")]
                lines.append(f"| {method} / `{case['id']}` | {len(summaries)} | " +
                             " | ".join(f"{value:.1f}" for value in values) + " |")
    if smoke:
        smoke_runs = checked_runs(smoke)
        requests = sum(r["summary"]["jev"]["requests"] for r in smoke_runs if r["status"] == "smoke_only")
        lines.extend(["", "## Pipeline verification", "",
                      f"{len(smoke_runs)} network checks, 30 seconds each, {requests} mock image "
                      "requests; these are plumbing checks and have no model score."])
    preflight = directory / "preflight.json"
    if preflight.exists():
        summary = json.loads(preflight.read_text(encoding="utf-8"))
        lines.append(f"A separate real 30-second Jinan preflight made {summary['jev']['requests']} "
                     f"requests, with {summary['jev']['failures']} failures and "
                     f"{summary['fallback_count']} fallback decisions. This short run is also excluded "
                     "from the full-horizon score tables.")
    requests = sum(r["summary"]["jev"]["requests"] for r in jev_runs if r["status"] == "complete")
    tokens = sum(r["summary"]["jev"]["input_tokens"] for r in jev_runs if r["status"] == "complete")
    lines.extend(["", "## Usage and reproducibility", "",
                  f"The planned real matrix has **{plan['estimated_image_requests']:,} requests before retries**. "
                  f"Completed real seeds currently report {requests:,} requests and {tokens:,} input tokens. "
                  "This excludes probes, failed/in-flight runs and any provider-side billing differences.", "",
                  "Official [model pricing](https://docs.typesafe.ai/models) at evaluation setup was "
                  "$0.042 per million input tokens, with free outputs; "
                  "[images](https://docs.typesafe.ai/images) count as input tokens. "
                  f"Completed-seed usage at that rate is approximately **${tokens * .042 / 1_000_000:.4f}**.", "",
                  "The first runs used Python 3.13.5, SUMO 1.24.0, Panda3D 1.10.16 and Pillow 12.2.0 "
                  "on Windows. Per-seed summaries embed simulator version, seed, wall time, "
                  "network/route SHA-256 hashes, Jev usage, renderer configuration and source revision. "
                  "Renderer assets are pinned to VLALight `9844f1b6bbd6b4fe80b481d3dd16fadce13c3b5d`. "
                  "Initial runs correctly record the pre-commit HEAD with `git_dirty: true`; "
                  "the local implementation commits preserve that working-tree implementation.", "",
                  "- [Frozen Jev plan](plan_jev-image.json)",
                  "- [Baseline seed summaries and aggregates](maxpressure.json)",
                  "- [Mock camera checks](smoke.json)"])
    if jev:
        lines.append("- [Real Jev seed summaries and aggregates](jev-image.json)")
    if api:
        lines.extend(["- [Latest image API preflight](api_preflight.json)", "",
                      f"Latest API check ({api['checked_at']}): {api['status']}. "
                      f"{api.get('error') or 'Preflight succeeded; this probe is excluded from model scores.'}"])
    for recovery in sorted(directory.glob("recovery_*.json")):
        lines.extend(["", f"- [Interruption and recovery record]({recovery.name})"])
    lines.extend(["", "Detailed frames, decision traces, generated SUMO configurations and runtime "
                  "logs remain in ignored `records/` and `logs/` directories. Reproduce using the "
                  "commands in the root README. These measurements are separate from the old "
                  "CityFlow benchmark; no remote homepage or PR is updated.", ""])
    (directory / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/vlalight_jev"))
    write_report(parser.parse_args().output)
