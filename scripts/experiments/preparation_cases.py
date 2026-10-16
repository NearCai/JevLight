"""Fixed synthetic traffic requests built with the unchanged benchmark functions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from jevlight.controller import congestion_question, jev_intersection_state, jevlight_phase_question
from jevlight.observation import CONTROLLED_LANES, IntersectionObservation, LaneObservation

ROOT = Path(__file__).resolve().parents[2]
FLOWS = {
    "jinan-1": ("jinan", "data/Jinan/3_4", "anon_3_4_jinan_real.rou.xml"),
    "jinan-2": ("jinan", "data/Jinan/3_4", "anon_3_4_jinan_real_2000.rou.xml"),
    "jinan-3": ("jinan", "data/Jinan/3_4", "anon_3_4_jinan_real_2500.rou.xml"),
    "hangzhou-1": ("hangzhou", "data/Hangzhou/4_4", "anon_4_4_hangzhou_real.rou.xml"),
    "hangzhou-2": ("hangzhou", "data/Hangzhou/4_4", "anon_4_4_hangzhou_real_5816.rou.xml"),
    "newyork-1": ("newyork", "data/NewYork/28_7", "anon_28_7_newyork_real_double.rou.xml"),
    "newyork-2": ("newyork", "data/NewYork/28_7", "anon_28_7_newyork_real_triple.rou.xml"),
}
SCENARIOS = ("empty", "directional_backlog", "symmetric", "turn_conflict", "long_wait", "high_load", "phase_change")
PHASES = ["ETWT", "NTST", "ELWL", "NLSL"]


def cases(speculative=False):
    for traffic_id, (_, directory, route) in FLOWS.items():
        network_path = next((ROOT / directory).glob("*.net.xml"))
        network = ET.parse(network_path).getroot()
        junction_id = network.find("tlLogic").attrib["id"]
        for index, scenario in enumerate(SCENARIOS):
            lanes = {}
            for name in CONTROLLED_LANES:
                queue, waiting, cells = 0, 0, [0, 0, 0, 0]
                if scenario == "directional_backlog" and name in ("NT", "ST"):
                    queue, waiting, cells = 32, 90, [5, 3, 2, 1]
                elif scenario == "symmetric":
                    queue, waiting, cells = 12, 45, [2, 3, 4, 1]
                elif scenario == "turn_conflict" and name in ("NL", "EL", "WT", "ST"):
                    queue, waiting, cells = 22, 100, [4, 2, 1, 0]
                elif scenario == "long_wait" and name in ("EL", "WL"):
                    queue, waiting, cells = 8, 900, [1, 1, 1, 1]
                elif scenario == "high_load":
                    queue, waiting, cells = 150 + index, 9999, [100, 150, 200, 250]
                elif scenario == "phase_change" and name in ("NL", "SL"):
                    queue, waiting, cells = 45, 180, [8, 5, 3, 2]
                lanes[name] = LaneObservation(queue=queue, avg_wait=waiting, llmlight_cells=cells,
                    approaching=sum(cells), road_length=300.0, occupancy=min(1.0, queue / 50))
            observation = IntersectionObservation(junction_id, lanes, PHASES,
                empty=scenario == "empty", current_phase=PHASES[index % 4])
            questions = {f"phase_{junction_id}": jevlight_phase_question(observation, 15)}
            if speculative:
                questions[f"congested_{junction_id}"] = congestion_question("$")
            yield {"case_id": f"{traffic_id}:{scenario}", "traffic_id": traffic_id,
                   "source": "synthetic; not captured SUMO observations", "traffic_file": f"{directory}/{route}",
                   "scenario": scenario, "request": {"state": jev_intersection_state(observation, 15),
                   "questions": questions}}


def source_hashes():
    files = ["run_jevlight.py", "jevlight/controller.py", "jevlight/observation.py",
             "jevlight/jev_client.py", "jevlight/cooperation.py", "utils/sumo_env.py",
             "jevlight/evaluation_metrics.py", "jevlight/rolling_evaluation.py",
             "config/jev/jinan.json", "config/jev/jinan_smoke.json"]
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/decision_models/preparation/fixed_cases")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "results"):
        parser.error("output must be inside results/")
    output.mkdir(parents=True, exist_ok=True)
    (output / "prompt_cases.jsonl").write_text("".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases()))
    (output / "benchmark_source_hashes.json").write_text(json.dumps(source_hashes(), indent=2) + "\n")


if __name__ == "__main__":
    main()
