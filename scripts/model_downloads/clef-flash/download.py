"""Download or reverify the frozen clef-flash checkpoint and its base."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from download import main

if __name__ == "__main__":
    main(default_models=["clef-flash"])
