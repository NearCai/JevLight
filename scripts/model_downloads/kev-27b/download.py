"""Download the frozen kev-27b checkpoint, native components and base."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from download import main

if __name__ == "__main__":
    sys.argv.extend(["--catalog", str(Path(__file__).resolve().parents[1] / "models.json")])
    main(default_models=["kev-27b"])
