"""Fetch the pinned VLALight renderer/assets, without its training checkpoints."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request

REPOSITORY = "usail-hkust/VLALight"
REVISION = "9844f1b6bbd6b4fe80b481d3dd16fadce13c3b5d"
SCENES = {"Jinan", "Hangzhou", "NewYork", "NewYork16x3"}


def selected(path):
    prefix = "VLALight/TransSimHub/"
    if path.startswith(prefix + "tshub/"):
        return path.endswith(".py") or (
            "/_assets_3d/" in path and "/event/" not in path
            and "/vehicles_high_poly/" not in path
        )
    return any(path.startswith(prefix + f"examples/{scene}/3d_assets/")
               for scene in SCENES) or path in {"LICENSE", prefix + "LICENSE"}


def blob_hash(data):
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("logs/vlalight-source"))
    args = parser.parse_args()
    tree = json.loads(subprocess.check_output([
        "gh", "api", f"repos/{REPOSITORY}/git/trees/{REVISION}?recursive=1"
    ]))
    if tree.get("truncated"):
        raise RuntimeError("GitHub returned an incomplete tree")
    entries = [e for e in tree["tree"] if e["type"] == "blob" and selected(e["path"])]

    def fetch(entry):
        target = args.destination / entry["path"]
        if target.is_file() and blob_hash(target.read_bytes()) == entry["sha"]:
            return
        url = f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/{entry['path']}"
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=90) as response:
                    data = response.read()
                if blob_hash(data) != entry["sha"]:
                    raise ValueError(f"Git blob checksum mismatch: {entry['path']}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                return
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(fetch, entries))
    manifest = {"repository": REPOSITORY, "revision": REVISION, "files": entries}
    args.destination.mkdir(parents=True, exist_ok=True)
    (args.destination / "renderer-manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Verified {len(entries)} files at {args.destination.resolve()}")


if __name__ == "__main__":
    main()
