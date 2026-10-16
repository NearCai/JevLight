"""Fetch pinned runtime/config files before downloading the large weight files."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json

from download import CATALOG, ROOT, metadata, now, save, selected, transfer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+")
    args = parser.parse_args()
    for key, identity in json.loads(CATALOG.read_text()).items():
        if args.models and key not in args.models:
            continue
        report = {"model_key": key, "started_at": now()}
        try:
            meta, endpoint = metadata(identity["model_id"], identity["revision"])
            destination = ROOT / ".local/models" / key / "weights"
            files = [entry for entry in meta["siblings"]
                     if selected(entry["rfilename"], key)
                     and entry["rfilename"].endswith((".py", ".json", ".md", ".txt", ".jinja"))]
            with ThreadPoolExecutor(max_workers=4) as pool:
                report["files"] = list(pool.map(lambda entry: transfer(identity["model_id"], identity["revision"], entry, destination), files))
            report.update(status="inspected", revision=meta["sha"], metadata_source=endpoint)
        except Exception as error:
            report.update(status="blocked", reason=f"{type(error).__name__}: {error}")
        report["finished_at"] = now()
        save(ROOT / "results" / key / "preparation/source_inspection.json", report)
        print(key, report["status"], flush=True)


if __name__ == "__main__":
    main()
