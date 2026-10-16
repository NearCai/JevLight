"""Fetch frozen author runtimes without replacing an existing checkout."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    sources = json.loads(Path(__file__).with_name("runtime_sources.json").read_text())
    for name, identity in sources.items():
        target = ROOT / identity["path"]
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--no-checkout", identity["url"], str(target)], check=True)
            subprocess.run(["git", "-C", str(target), "checkout", "--detach", identity["revision"]], check=True)
        revision = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(target), "status", "--porcelain", "--untracked-files=no"], text=True)
        if revision != identity["revision"] or dirty:
            raise RuntimeError(f"{name}: existing runtime differs from frozen source; refusing to overwrite")
        print(name, revision)


if __name__ == "__main__":
    main()
