"""Install batch diagnostics, then execute the unchanged single-run entrypoint."""
from pathlib import Path
import runpy


def main():
    from jevlight import jev_client
    from jevlight.step_batch_support import StepBatchClient
    jev_client.JevClient = StepBatchClient
    runpy.run_path(str(Path(__file__).resolve().parents[2] / 'run_jevlight.py'), run_name='__main__')


if __name__ == '__main__':
    main()
