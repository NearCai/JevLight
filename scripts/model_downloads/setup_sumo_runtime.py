"""Restore uv SUMO and, on Ubuntu, its missing user-local Xrender library."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    subprocess.run(["uv", "sync", "--locked", "--extra", "sumo", "--extra", "preparation"], cwd=ROOT, check=True)
    sumo = ROOT / ".venv/bin/sumo"
    env = os.environ.copy()
    runtime = ROOT / ".local/sumo_runtime"
    libraries = runtime / "usr/lib/x86_64-linux-gnu"
    env["LD_LIBRARY_PATH"] = str(libraries) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    check = subprocess.run([str(sumo), "--version"], env=env, capture_output=True, text=True)
    if check.returncode and "libXrender.so.1" in check.stderr:
        downloads = runtime / "debs"
        downloads.mkdir(parents=True, exist_ok=True)
        subprocess.run(["apt-get", "download", "libxrender1"], cwd=downloads, check=True)
        for package in downloads.glob("libxrender1*.deb"):
            subprocess.run(["dpkg-deb", "-x", str(package), str(runtime)], check=True)
        check = subprocess.run([str(sumo), "--version"], env=env, capture_output=True, text=True)
    if check.returncode:
        raise RuntimeError(check.stderr)
    output = ROOT / "results/decision_models/preparation/sumo_runtime.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    packages = [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in (runtime / "debs").glob("*.deb")]
    output.write_text(json.dumps({"sumo_version": check.stdout, "local_libraries": str(libraries), "packages": packages}, indent=2) + "\n")
    print(check.stdout)


if __name__ == "__main__":
    main()
