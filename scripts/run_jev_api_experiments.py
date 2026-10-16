"""Run the frozen official-API matrix through run_jevlight.py, with audits.

Credentials are read into the child environment only. API requests and answers
are retained without headers. Failed preflights never count as valid experiments.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import dataclasses
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import subprocess
import statistics
import sys
import threading
import time
import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VALIDATION_VERSION = 'official-choice-authoritative-v3'
REUSABLE_VALIDATION_VERSIONS = {VALIDATION_VERSION, 'official-strict-argmax-retry-v2'}


def validation_protocol():
    return dict(version=VALIDATION_VERSION, decision_source='api_choice',
                probability_validation=False, probability_retries=0,
                unchanged_request=True, other_validation_errors_fatal=True,
                reuse_completed_versions=sorted(REUSABLE_VALIDATION_VERSIONS),
                decision_policy='Use the actual valid choice; probabilities are audit data')


def source_hashes():
    from scripts.experiments.preparation_cases import source_hashes as benchmark_hashes
    hashes = benchmark_hashes()
    # The choice-only official path does not import the local GPU model adapter.
    for name in ('scripts/run_jev_api_experiments.py', 'jevlight/api_key_pool.py', 'uv.lock', 'pyproject.toml'):
        hashes[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    return hashes


def incompatible_sources(previous, current):
    return [name for name, digest in previous.items()
            if name not in ('scripts/run_jev_api_experiments.py', 'jevlight/backends/decision_model_service.py')
            and current.get(name) != digest]


def network_environment(environment):
    """Use the caller's networking environment without machine-local configuration."""
    return dict(environment)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def export_records(batch, records):
    write_json(batch / 'experiment_records.json', records)
    fields = ['config', 'status', 'started_at', 'wall_time_s', 'fallback_rate', 'summary', 'reason', 'aql', 'awt', 'att']
    with (batch / 'experiment_records.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields, extrasaction='ignore')
        writer.writeheader()
        for record in records:
            metrics = record.get('metrics', {})
            writer.writerow(dict(record, aql=metrics.get('final/avg_queue_len'), awt=metrics.get('final/avg_waiting_time'), att=metrics.get('final/avg_travel_time')))
    groups = {}
    for record in records:
        if record['status'] != 'completed' or record.get('stage') != 'formal':
            continue
        identity = Path(record['config']).stem
        group = identity.rsplit('_s', 1)[0]
        groups.setdefault(group, []).append(record)
    aggregates = []
    for group, members in groups.items():
        row = dict(condition=group, successful_runs=len(members))
        for metric, key in [('aql', 'avg_queue_len'), ('awt', 'avg_waiting_time'), ('att', 'avg_travel_time')]:
            values = [r['metrics']['final/'+key] for r in members]
            row[metric+'_raw'] = values
            row[metric+'_mean'] = statistics.mean(values)
            row[metric+'_std'] = statistics.stdev(values) if len(values) > 1 else None
        aggregates.append(row)
    write_json(batch / 'aggregate.json', aggregates)
    aggregate_fields = ['condition', 'successful_runs'] + [metric+'_'+suffix for metric in ('aql', 'awt', 'att') for suffix in ('raw', 'mean', 'std')]
    with (batch / 'aggregate.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregates)
    write_json(batch / 'progress.json', dict(updated_at=datetime.now(timezone.utc).isoformat(), completed=sum(r['status']=='completed' and r.get('stage')=='formal' for r in records), blocked=sum(r['status']=='blocked' for r in records), records=len(records)))


def validate_official_choices(request, result):
    """Use the official choice directly; probabilities are audit data only."""
    answers = result.get('answers', {})
    if set(answers) != set(request['questions']):
        raise ValueError('answer IDs differ from question IDs')
    for question_id, question in request['questions'].items():
        answer = answers[question_id]
        if answer.get('type') != question['type']:
            raise ValueError('answer type differs from question type')
        if question['type'] == 'choice' and answer.get('choice') not in question['criteria']:
            raise ValueError('invalid phase choice')
    return result


def reusable_completed(record):
    if record.get('status') != 'completed' or record.get('validation_version') not in REUSABLE_VALIDATION_VERSIONS:
        return False
    summary = record.get('summary')
    return bool(summary and (ROOT / summary).is_file())


def audited_evaluate(client, state, questions, original, counts, emit, lock=None):
    request = dict(state=state, questions=dict(questions), model=client.model)
    lock = lock or threading.RLock()
    with lock:
        logical_id = counts['logical_requests'] + 1
        counts['logical_requests'] = logical_id
    started = time.monotonic()
    result = original(client, state, questions)
    raw_response = dataclasses.asdict(result)
    validation_error = None
    if result.error is None:
        try:
            validate_official_choices(request, raw_response)
            if result.model != client.model:
                raise ValueError(f'model changed: {result.model}')
        except (ValueError, TypeError, KeyError) as exc:
            validation_error = str(exc)
    with lock:
        counts['requests'] += 1
        counts['failures'] += result.error is not None or validation_error is not None
        counts['validation_failures'] += validation_error is not None
        counts['max_input_tokens'] = max(counts['max_input_tokens'], result.input_tokens)
        counts['max_request_bytes'] = max(counts['max_request_bytes'], len(json.dumps(request).encode()))
        counts['max_state_bytes'] = max(counts['max_state_bytes'], len(json.dumps(state).encode()))
        counts['max_questions_per_request'] = max(counts['max_questions_per_request'], len(questions))
        if result.model and result.model not in counts['models']:
            counts['models'].append(result.model)
        error = ('invalid API answer: ' + validation_error) if validation_error else result.error
        if error and error not in counts['errors']:
            counts['errors'].append(error)
        if validation_error:
            counts['unresolved_validation_failures'] += 1
        emit(dict(request=request, response=raw_response, validation_error=validation_error,
                  logical_request=logical_id, attempt=1, retry=False,
                  validation_version=VALIDATION_VERSION))
    # Match HTTP status at the error prefix, never digits in request_id.
    fatal_http = bool(error and re.search(r'^(?:TypeSafeAPIError:\s+POST https?://\S+:\s+|HTTPError:\s+)(?:402|422)\b', error))
    if error and (fatal_http or 'max_tokens_exceeded' in error or validation_error):
        raise RuntimeError(error)
    result.latency = time.monotonic() - started
    return result


def child(config, audit):
    from jevlight.jev_client import JevClient
    audit.mkdir(parents=True, exist_ok=True)
    original = JevClient.evaluate
    lock = threading.Lock()
    counts = dict(requests=0, logical_requests=0, failures=0, validation_failures=0,
                  unresolved_validation_failures=0, recovered_requests=0,
                  validation_version=VALIDATION_VERSION,
                  max_input_tokens=0, max_request_bytes=0, max_state_bytes=0,
                  max_questions_per_request=0, models=[], errors=[])
    with gzip.open(audit / 'requests.jsonl.gz', 'wt', encoding='utf-8') as stream:
        def emit(row):
            stream.write(json.dumps(row) + '\n')
            stream.flush()
            write_json(audit / 'api_audit.json', counts)

        def evaluate(client, state, questions):
            if client.transport != 'official':
                raise RuntimeError('JEV API queue requires official transport')
            return audited_evaluate(client, state, questions, original, counts, emit, lock=lock)

        JevClient.evaluate = evaluate
        sys.argv = ['run_jevlight.py', '--config', str(config)]
        runpy.run_path(str(ROOT / 'run_jevlight.py'), run_name='__main__')


def next_attempt(batch, identity):
    condition = batch / identity
    condition.mkdir(parents=True, exist_ok=True)
    for index in range(1, 10000):
        attempt = condition / f'attempt_{index:04d}'
        try:
            attempt.mkdir()
            return attempt
        except FileExistsError:
            continue
    raise RuntimeError('Attempt limit reached')


def run_one(config, batch, env):
    settings = json.loads(config.read_text())
    if config.stem.startswith('maxpressure_') or settings.get('dry_run') or settings.get('agent') == 'maxpressure':
        raise ValueError('API model campaign cannot execute or depend on MaxPressure; run it separately later')
    identity = config.stem
    attempt = next_attempt(batch, identity)
    resolved = subprocess.run(['uv', 'run', 'python', 'run_jevlight.py', '--config', str(config), '--print_config'], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    (attempt / 'resolved_config.json').write_text(resolved.stdout)
    settings = json.loads(resolved.stdout)
    output = ROOT / 'results/jev-api' / settings['memo']
    before = set(output.glob('*/summary.json'))
    started = time.time()
    record = dict(validation_version=VALIDATION_VERSION, stage='smoke' if identity.endswith('_smoke') else 'formal', attempt=attempt.name, config=str(config.relative_to(ROOT)), config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(), source_hashes=source_hashes(), started_at=datetime.now(timezone.utc).isoformat(), status='running', audit=str(attempt.relative_to(ROOT)))
    write_json(attempt / 'record.json', record)
    write_json(batch / 'current.json', record)
    print(f'Starting {identity}', flush=True)
    with (attempt / 'console.log').open('w') as log:
        process = subprocess.Popen(['uv', 'run', 'python', str(Path(__file__).resolve()), '--child', '--config', str(config), '--audit', str(attempt)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record.update(child_pid=process.pid, queue_pid=os.getpid())
        write_json(attempt / 'record.json', record)
        write_json(batch / 'current.json', record)
        try:
            process.wait()
        except BaseException as exc:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            record.update(status='interrupted', reason=str(exc), exit_code=process.returncode)
            write_json(attempt / 'record.json', record)
            write_json(batch / 'current.json', record)
            raise
    summaries = set(output.glob('*/summary.json')) - before
    record.update(exit_code=process.returncode, wall_time_s=time.time()-started, status='failed')
    if len(summaries) == 1:
        summary_path = summaries.pop()
        summary = json.loads(summary_path.read_text())
        traces = [json.loads(line) for line in (summary_path.parent / 'jev_decisions.jsonl').open()]
        active = [t for t in traces if t.get('fallback') != 'empty_junction_local_ranking']
        fallback = sum(t.get('fallback') is not None for t in active)
        rate = fallback / len(active) if active else None
        valid_metrics = all(type(summary.get('final/'+key)) in (int,float) and math.isfinite(summary['final/'+key]) for key in ('avg_queue_len','avg_waiting_time','avg_travel_time'))
        record.update(summary=str(summary_path.relative_to(ROOT)), fallback_rate=rate, active_decisions=len(active), empty_decisions=len(traces)-len(active), metrics=summary)
        api_audit = json.loads((attempt / 'api_audit.json').read_text()) if (attempt / 'api_audit.json').exists() else {}
        valid = process.returncode == 0 and valid_metrics and summary.get('final/duration') == settings['count'] and (rate is not None and rate <= 0.01 and api_audit.get('unresolved_validation_failures') == 0 and api_audit.get('validation_version') == VALIDATION_VERSION)
        record['status'] = 'completed' if valid else 'needs_review'
        write_json(summary_path.parent / 'metrics_record.json', record)
    write_json(attempt / 'record.json', record)
    write_json(batch / 'current.json', record)
    print(f"Finished {identity}: {record['status']} ({record['wall_time_s']:.1f}s)", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', type=Path, help='Resume a compatible batch, preserving completed conditions')
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--stage', choices=['smoke', 'formal', 'all'], default='smoke')
    args = parser.parse_args()
    if args.child:
        child(args.config, args.audit)
        return
    # One official-API queue at a time, including after a chat interruption.
    lock_path = ROOT / 'results/jev-api/queue.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    queue_lock = lock_path.open('w')
    fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / 'cmdline').read_bytes().decode().split('\0')
        except (OSError, UnicodeError):
            continue
        if '--child' in command and any(Path(arg).name == 'run_jev_api_experiments.py' for arg in command):
            raise RuntimeError(f'Existing JEV child PID {entry.name}; reconcile orphan before starting queue')
    env = network_environment(os.environ)
    from jevlight.api_key_pool import DEFAULT_VAULT, KeyPool
    if env.get('JEV_API_KEY_POOL') or DEFAULT_VAULT.exists() or not env.get('TYPESAFE_API_KEY'):
        vault = env.get('JEV_API_KEY_POOL') or str(DEFAULT_VAULT)
        KeyPool(vault).select()  # Fail before starting a batch if all credits are exhausted.
        env['JEV_API_KEY_POOL'] = vault
        env.pop('TYPESAFE_API_KEY', None)
    env.pop('JEV_API_KEY', None)
    libraries = ROOT / '.local/sumo_runtime/usr/lib/x86_64-linux-gnu'
    env['LD_LIBRARY_PATH'] = str(libraries) + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    batch = args.resume.resolve() if args.resume else ROOT / 'results/jev-api' / args.stage / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    if not batch.is_relative_to(ROOT / 'results/jev-api'):
        raise ValueError('Batch must be inside results/jev-api')
    if args.resume:
        old_hashes = json.loads((batch / 'source_hashes.json').read_text())
        current_hashes = source_hashes()
        if incompatible_sources(old_hashes, current_hashes):
            raise RuntimeError('Cannot resume: benchmark changed')
        old_protocol = json.loads((batch / 'validation_protocol.json').read_text())
        if old_protocol.get('version') not in REUSABLE_VALIDATION_VERSIONS:
            raise RuntimeError('Cannot resume: incompatible validation protocol')
        for name, expected in json.loads((batch / 'config_hashes.json').read_text()).items():
            if hashlib.sha256((ROOT / 'config/jev-api' / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError('Cannot resume: config changed')
        if old_protocol.get('version') != VALIDATION_VERSION:
            write_json(batch / 'protocol_migration.json', dict(previous_protocol=old_protocol, new_protocol=validation_protocol(), reuse_formal=True, revalidate_all_smoke=False, reason='User authorized choice-only validation; completed stricter runs retain original identities'))
        write_json(batch / 'scheduler_migration.json', dict(previous_hashes=old_hashes, new_hashes=current_hashes, reason='Resume with compatible validation; original records and failed attempts retained'))
    else:
        batch.mkdir(parents=True, exist_ok=False)
    from scripts.experiments.preparation_cases import FLOWS
    for name in source_hashes():
        target = batch / 'sources' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    write_json(batch / 'source_hashes.json', source_hashes())
    write_json(batch / 'context_limits.json', dict(source='https://docs.typesafe.ai/models', model='jev-1.13.0', request_tokens=64000, state_plus_longest_question_tokens=32000, tokenizer='Hosted tokenizer/encoding not published; API usage is actual server count on accepted requests. Internal limits cannot be independently verified.', truncation=False))
    write_json(batch / 'validation_protocol.json', validation_protocol())
    configs = ROOT / 'config/jev-api'
    if args.stage == 'smoke':
        names = [f'{mode}_{flow}_smoke.json' for mode in ('cojevlight','jevlight') for flow in ('jinan-2','newyork-2')]
    elif args.stage == 'formal':
        names = [f'{mode}_{flow}_s{seed}.json' for flow in FLOWS for seed in (1,2,3) for mode in ('jevlight','cojevlight')]
    else:
        smoke_flows = ['newyork-2', 'jinan-2'] + [flow for flow in FLOWS if flow not in ('newyork-2', 'jinan-2')]
        names = [f'{mode}_{flow}_smoke.json' for flow in smoke_flows for mode in ('cojevlight','jevlight')]
        names += [f'{mode}_{flow}_s{seed}.json' for flow in FLOWS for seed in (1,2,3) for mode in ('jevlight','cojevlight')]
    records = json.loads((batch / 'experiment_records.json').read_text()) if args.resume else []
    if args.resume:
        write_json(batch / 'baseline_records_before_sharing.json', [r for r in records if r.get('stage') == 'baseline'])
        records = [r for r in records if r.get('stage') != 'baseline']
    preflight = {Path(r['config']).stem.removesuffix('_smoke'): r for r in records if r.get('stage') == 'smoke' and reusable_completed(r)}
    completed_configs = {r['config'] for r in records if reusable_completed(r)}
    frozen_hashes = source_hashes()
    frozen_configs = {name: hashlib.sha256((configs / name).read_bytes()).hexdigest() for name in names}
    write_json(batch / 'config_hashes.json', frozen_configs)
    for name in names:
        (batch / 'configs').mkdir(exist_ok=True)
        (batch / 'configs' / name).write_bytes((configs / name).read_bytes())
    global ACTIVE_BATCH
    ACTIVE_BATCH = batch
    write_json(ROOT / 'results/jev-api/active_queue.json', dict(pid=os.getpid(), batch=str(batch), stage=args.stage, status='running', validation_version=VALIDATION_VERSION))
    write_json(batch / 'process.json', dict(pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat(), stage=args.stage, status='running'))
    export_records(batch, records)
    for name in names:
        if f'config/jev-api/{name}' in completed_configs:
            continue
        is_smoke = name.endswith('_smoke.json')
        condition = name.removesuffix('_smoke.json') if is_smoke else name.rsplit('_s', 1)[0]
        stage = 'smoke' if is_smoke else 'formal'
        if source_hashes() != frozen_hashes or hashlib.sha256((configs / name).read_bytes()).hexdigest() != frozen_configs[name]:
            raise RuntimeError('Source/config changed while queue was running; stopping to preserve protocol identity.')
        if args.stage == 'all' and not is_smoke  and preflight.get(condition, {}).get('status') != 'completed':
            records.append(dict(config=f'config/jev-api/{name}', stage=stage, status='blocked', reason='Current-protocol smoke did not pass', preflight=preflight.get(condition)))
            export_records(batch, records)
            continue
        record = run_one(configs / name, batch, env)
        record['stage'] = stage
        records.append(record)
        if is_smoke:
            preflight[condition] = record
        export_records(batch, records)
        if not is_smoke and record['status'] != 'completed':
            raise RuntimeError('Formal matrix stopped at failed/needs_review attempt; preserve artifacts before resuming.')
    print(f'Batch records: {batch}', flush=True)
    write_json(batch / 'process.json', dict(pid=os.getpid(), status='finished', finished_at=datetime.now(timezone.utc).isoformat()))
    write_json(ROOT / 'results/jev-api/active_queue.json', dict(pid=os.getpid(), batch=str(batch), stage=args.stage, status='finished'))


ACTIVE_BATCH = None


def terminate_queue(signum, frame):
    raise RuntimeError(f'Queue interrupted by signal {signum}')


if __name__ == '__main__':
    if '--child' not in sys.argv:
        signal.signal(signal.SIGTERM, terminate_queue)
    try:
        main()
    except BaseException as exc:
        if ACTIVE_BATCH is not None:
            state = dict(pid=os.getpid(), batch=str(ACTIVE_BATCH), status='failed',
                         reason=f'{type(exc).__name__}: {exc}', finished_at=datetime.now(timezone.utc).isoformat())
            write_json(ACTIVE_BATCH / 'process.json', state)
            write_json(ROOT / 'results/jev-api/active_queue.json', state)
        raise
