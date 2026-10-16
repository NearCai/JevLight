"""Cached complete CPU encoding before any GPU model load."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from scripts.experiments.preparation_cases import source_hashes


def ensure_cases(root):
    root = Path(root)
    output = root / 'results/decision_models/preparation'
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'cases.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = source_hashes()
        for directory, module in [('fixed_cases', 'preparation_cases'), ('cooperative_cases', 'cooperative_cases')]:
            path = output / directory
            identity = path / 'benchmark_source_hashes.json'
            if identity.exists() and json.loads(identity.read_text()) == current and (path / 'prompt_cases.jsonl').exists():
                continue
            if path.exists():
                digest = hashlib.sha256((identity.read_bytes() if identity.exists() else b'') +
                    ((path / 'prompt_cases.jsonl').read_bytes() if (path / 'prompt_cases.jsonl').exists() else b'')).hexdigest()
                history = output / 'case_history' / directory / digest
                if not history.exists():
                    history.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(path, history)
            subprocess.run(['uv', 'run', '--locked', 'python', '-m', 'scripts.experiments.' + module], cwd=root, check=True)


def cpu_preflight(root, key, project):
    from scripts.experiments.acceptance_identity import acceptance_identity
    root = Path(root)
    ensure_cases(root)
    identity = acceptance_identity(root, key)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output = root / 'results' / key / 'preparation/cpu' / digest
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        gate = output / 'gate.json'
        if gate.exists():
            cached = json.loads(gate.read_text())
            raw = output / 'encoders' / (key + '.json')
            if cached.get('identity') == identity and cached.get('status') == 'passed' and raw.exists():
                prior = json.loads(raw.read_text())
                cases = prior.get('cases', [])
                if (prior.get('status') == 'passed' and len(cases) == 98
                        and len({r.get('case_id') for r in cases}) == 98
                        and all(r.get('status') == 'passed' for r in cases)):
                    return output
        for mode, source in [('local', 'fixed_cases'), ('co', 'cooperative_cases')]:
            target = output / mode
            target.mkdir(exist_ok=True)
            shutil.copy(root / 'results/decision_models/preparation' / source / 'prompt_cases.jsonl', target / 'prompt_cases.jsonl')
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
        env.pop('VIRTUAL_ENV', None)
        env.pop('UV_PROJECT_ENVIRONMENT', None)
        with (output / 'encoding.log').open('w') as log:
            subprocess.run(['uv', 'run', '--locked', '--project', str(project), 'python',
                'scripts/model_downloads/audit_prompt_encoding.py', '--model', key,
                '--cases-root', str(output)], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads((output / 'encoders' / (key + '.json')).read_text())
        cases = report.get('cases', [])
        passed = (report.get('status') == 'passed' and len(cases) == 98
                  and len({r['case_id'] for r in cases}) == 98
                  and all(r['status'] == 'passed' for r in cases))
        gate.write_text(json.dumps(dict(status='passed' if passed else 'blocked', identity=identity,
            report=str(output / 'encoders' / (key + '.json')), scope='Complete native CPU encoding; not GPU inference'), indent=2) + '\n')
        if not passed:
            raise RuntimeError(f'{key}: complete CPU encoding blocked; see {output}')
        return output


def gpu_margin_mib(total_bytes):
    return max(4096.0, total_bytes / 1024**2 * 0.10)
