"""Run unchanged benchmark configurations and retain every attempt."""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

from scripts.experiments.preparation_cases import ROOT, FLOWS, source_hashes


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.tmp')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    pending.replace(path)


def matrix(key):
    catalog = json.loads((ROOT / 'scripts/model_downloads/models.json').read_text())
    if key not in catalog:
        raise ValueError('Select a model from the frozen catalog')
    return [dict(model_key=key, mode=mode, traffic_id=traffic, seed=seed,
                 config=f'config/{key}/{mode}_{traffic}_s{seed}.json')
            for traffic in FLOWS for seed in (1, 2, 3)
            for mode in ('jevlight', 'cojevlight')]


def row_id(row):
    return f"{row['model_key']}_{row['mode']}_{row['traffic_id']}_s{row['seed']}"


def run_child(command, log, environment):
    with Path(log).open('w') as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            return process.wait()
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def execute(row, backend, overrides=None):
    """Overrides are transport/batch controls or a bounded smoke duration."""
    if row not in matrix(row['model_key']):
        raise ValueError('Run is outside the frozen 42-condition matrix')
    path = ROOT / row['config']
    config = json.loads(path.read_text())
    if config.get('mock_jev') or config.get('dry_run') or config['agent'] != row['mode']:
        raise ValueError('Model experiments require real inference in the declared mode')
    changes = dict(overrides or {})
    if set(changes) - {'count', 'jev_batch_size', 'jev_base_url', 'memo', 'rolling_evaluation'}:
        raise ValueError('Experiment protocol changes require an explicit configuration')
    config.update(changes)
    identifier = row_id(row)
    attempt = ROOT / 'results/experiments/attempts' / (identifier + '_' + uuid.uuid4().hex)
    attempt.mkdir(parents=True)
    config['memo'] = attempt.name
    command = ['uv', 'run', '--locked', '--project', str(ROOT), '--extra', 'sumo',
               'python', '-m', 'scripts.experiments.step_batch_runner', '--config', str(path)]
    for name in ('memo', 'count', 'jev_batch_size', 'jev_base_url'):
        if name in config:
            command.extend(['--' + name, str(config[name])])
    if 'rolling_evaluation' in changes:
        command.append('--rolling_evaluation' if config['rolling_evaluation'] else '--no-rolling_evaluation')
    environment = dict(os.environ)
    environment.pop('VIRTUAL_ENV', None)
    environment.pop('UV_PROJECT_ENVIRONMENT', None)
    libraries = ROOT / '.local/sumo_runtime/usr/lib/x86_64-linux-gnu'
    if libraries.is_dir():
        environment['LD_LIBRARY_PATH'] = str(libraries) + ':' + environment.get('LD_LIBRARY_PATH', '')
    original_config = path.read_bytes()
    record = dict(row, id=identifier, attempt_path=str(attempt), status='running',
                  stage='smoke' if config['count'] != 3600 else 'formal',
                  started_at=datetime.now(timezone.utc).isoformat())
    record['hashes'] = dict(benchmark=source_hashes(),
        service_code=hashlib.sha256((ROOT/'jevlight/backends/decision_model_service.py').read_bytes()).hexdigest(),
        weights_manifest=hashlib.sha256((ROOT/'.local/models'/row['model_key']/'manifest.json').read_bytes()).hexdigest())
    save(attempt / 'record.json', record)
    backend.log_path = attempt / 'native_requests.jsonl'
    started = time.monotonic()
    try:
        if run_child(command + ['--print_config'], attempt / 'effective_config.json', environment):
            raise RuntimeError('Final experiment configuration was rejected')
        code = run_child(command, attempt / 'runner.log', environment)
        summaries = list((ROOT / 'results' / row['model_key'] / config['memo']).glob('*/summary.json'))
        if code or len(summaries) != 1:
            raise RuntimeError(f'Runner exit={code}; expected one summary, found {len(summaries)}')
        summary_path = summaries[0]
        summary = json.loads(summary_path.read_text())
        trace = summary_path.with_name('jev_decisions.jsonl')
        decisions = [json.loads(line) for line in trace.read_text().splitlines()]
        values = metrics(summary, decisions)['values']
        valid = (all(values[k] is not None for k in ('aql', 'awt', 'att'))
                 and values['simulation_duration_s'] == config['count']
                 and values['model_decisions'] > 0 and values['model_failure_fallbacks'] == 0)
        record.update(values, summary_path=str(summary_path),
                      status='completed' if valid else 'needs_review')
        record['artifacts'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (summary_path, trace)}
        save(summary_path.with_name('metrics_record.json'), metrics(summary, decisions))
    except Exception as error:
        record.update(status='failed', reason=f'{type(error).__name__}: {error}')
    except BaseException:
        record['status'] = 'interrupted'
        raise
    finally:
        record['wall_duration_s'] = time.monotonic() - started
        if path.read_bytes() != original_config:
            record.update(status='needs_review', reason='Experiment configuration changed during execution')
        backend.log_path = None
        save(attempt / 'record.json', record)
    return record


def metrics(summary, decisions, baseline=False):
    aliases = {"aql": "final/avg_queue_len", "awt": "final/avg_waiting_time", "att": "final/avg_travel_time"}
    values = {}
    for alias, field in aliases.items():
        value = summary.get(field)
        values[alias] = value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None
    empty = sum(row.get("fallback") == "empty_junction_local_ranking" for row in decisions)
    failures = sum(bool(row.get("error")) or row.get("fallback") not in (None, "empty_junction_local_ranking") for row in decisions)
    denominator = len(decisions) - empty
    values.update(empty_junction_actions=empty, model_decisions=0 if baseline else denominator,
                  model_failure_fallbacks=0 if baseline else failures,
                  fallback_rate=None if baseline or not denominator else failures / denominator,
                  simulation_duration_s=summary.get("final/duration"))
    return {"aliases": aliases, "values": values, "original_metrics": {k: v for k, v in summary.items() if k.startswith("final/")},
            "semantics": "Original benchmark metrics; ATT includes elapsed time of some unfinished vehicles. AQL is network queue per sample. Empty-junction local actions excluded from model fallback denominator."}
