"""Qualify native batch inference and run the common model experiment matrix."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import statistics
from pathlib import Path
import signal
import subprocess
import sys
import threading
import tomllib

from scripts.experiments.acceptance_identity import acceptance_identity
from scripts.experiments.batch_acceptance import compare_phases, validate_sumo_pair
from scripts.experiments.execution import ROOT, execute, matrix, row_id, save
from scripts.experiments.serve_decision_model import build_backend, load_selection

MODES = ('jevlight', 'cojevlight')


def artifact_matches(artifacts):
    return bool(artifacts) and all(Path(p).is_file() and
        hashlib.sha256(Path(p).read_bytes()).hexdigest() == checksum
        for p, checksum in artifacts.items())


def report_ready(report, identity, device):
    """Reject partial preparation, missing evidence and stale measurements."""
    if (report.get('status') != 'ready' or report.get('identity', {}).get('inputs') != identity
            or report['identity'].get('gpu_uuid') != device
            or report.get('encoded_case_count') != 98
            or report.get('native_input_mapping_valid') is not True
            or not artifact_matches(report.get('artifacts', {}))):
        return False
    expected_artifacts = set()
    for mode in MODES:
        entry = report.get('modes', {}).get(mode, {})
        profile = entry.get('short_profile', {})
        if (entry.get('status') != 'ready' or entry.get('capacity_probe_mapping_valid') is not True
                or entry.get('largest_encoded_tokens', 0) <= 0
                or profile.get('warmup') != 3 or profile.get('measure') != 10
                or max(entry.get('short_profile_gpu_rows', []), default=0)
                   <= max(1, entry.get('single_request_gpu_rows', 0))
                or not math.isfinite(entry.get('smoke_speedup', 0))
                or entry.get('smoke_speedup', 0) <= 1
                or report.get('mode_batch_sizes', {}).get(mode, 0) < entry.get('full_step_target', 196)
                or entry.get('sumo_behavior_valid') is not True):
            return False
        try:
            pair = entry.get('smoke', [])
            validate_sumo_pair(pair)
            if any(row.get('simulation_duration_s') != 60 or row.get('model_decisions', 0) <= 0
                   for row in pair):
                return False
            for row in pair:
                summary = Path(row['summary_path'])
                expected_artifacts.update((str(summary), str(summary.with_name('jev_decisions.jsonl'))))
        except (ValueError, OSError, KeyError, TypeError):
            return False
    mapping = report.get('mapping_artifact')
    if not mapping:
        return False
    expected_artifacts.add(mapping)
    return expected_artifacts <= report['artifacts'].keys()


def require_scope_ready(root, keys, device=None):
    reports = {}
    for key in keys:
        path = Path(root) / 'results' / key / 'preparation/qualification.json'
        if not path.is_file():
            raise RuntimeError(f'{key}: batch qualification is missing')
        report = json.loads(path.read_text())
        expected_device = device if device is not None else report.get('identity', {}).get('gpu_uuid')
        if not expected_device or not report_ready(report, acceptance_identity(root, key, report.get('execution_profile')), expected_device):
            raise RuntimeError(f'{key}: current local/cooperative GPU batch qualification is incomplete')
        reports[key] = report
    return reports


def environment_identity():
    versions = {}
    for package in ('torch', 'transformers', 'vllm', 'vllm-jev', 'peft', 'laya', 'gliner2', 'triton'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def qualify(key, backend, device):
    from jevlight.step_batch_support import encoded_request_length
    from scripts.experiments.step_batch_acceptance import select_step_batch

    initial = acceptance_identity(ROOT, key, backend.execution_profile)
    rows = matrix(key)
    complete = []
    for mode, directory in zip(MODES, ('fixed_cases', 'cooperative_cases')):
        path = ROOT / 'results/decision_models/preparation' / directory / 'prompt_cases.jsonl'
        cases = [json.loads(line) for line in path.read_text().splitlines()]
        if len(cases) != 49 or len({case['case_id'] for case in cases}) != 49:
            raise ValueError('Expected 49 distinct complete benchmark cases per mode')
        requests = [case['request'] for case in cases]
        lengths = [encoded_request_length(backend.runtime.encode(request)) for request in requests]
        if any(length <= 0 for length in lengths):
            raise ValueError('An encoded input has unknown or invalid length')
        original = [backend.runtime.native(request) for request in requests]
        batched = backend.runtime.native_batch(requests, max(len(requests), 196))
        compare_phases(requests, original, batched, require_equal=False)
        complete.extend(dict(mode=mode, case_id=case['case_id'], encoded_tokens=length,
                             request=case['request'], native=a, batch=b)
                        for case, length, a, b in zip(cases, lengths, original, batched, strict=True))
    select_step_batch(ROOT, key, backend, rows, execute, device)
    path = backend.qualification_report_path
    report = json.loads(path.read_text())
    evidence = path.with_name('native_mapping.json')
    save(evidence, complete)
    report.update(encoded_case_count=len(complete), native_input_mapping_valid=True,
                  execution_profile=backend.execution_profile, environment=environment_identity(),
                  mapping_artifact=str(evidence))
    report.setdefault('artifacts', {})[str(evidence)] = hashlib.sha256(evidence.read_bytes()).hexdigest()
    if acceptance_identity(ROOT, key, backend.execution_profile) != initial:
        report.update(status='blocked', reason='Qualification inputs changed during execution')
    save(path, report)
    save(ROOT / 'results' / key / 'preparation/qualification.json', report)
    if not report_ready(report, acceptance_identity(ROOT, key, backend.execution_profile), device):
        raise RuntimeError(f'{key}: qualification incomplete; see {path}')
    return report


def summarize(records):
    """Aggregate exactly 42 completed conditions without changing raw metrics."""
    if len(records) != 42 or len({row_id(r) for r in records}) != 42:
        raise ValueError('A complete unique 42-condition matrix is required')
    expected = {row_id(r) for r in matrix(records[0]['model_key'])}
    if {row_id(r) for r in records} != expected or any(r['status'] != 'completed' for r in records):
        raise ValueError('Incomplete or mixed model matrix')
    rows = {(r['traffic_id'], r['mode'], r['seed']): r for r in records}
    groups = []
    for traffic in dict.fromkeys(r['traffic_id'] for r in records):
        values = {}
        for metric in ('aql', 'awt', 'att'):
            modes = {mode: [rows[traffic, mode, seed][metric] for seed in (1, 2, 3)] for mode in MODES}
            modes['co_minus_local'] = [b-a for a,b in zip(modes['jevlight'], modes['cojevlight'], strict=True)]
            if not all(math.isfinite(v) for group in modes.values() for v in group):
                raise ValueError('Nonfinite matrix metrics')
            values[metric] = {mode: dict(values=v, mean=statistics.mean(v), sample_std=statistics.stdev(v))
                              for mode,v in modes.items()}
        groups.append(dict(traffic_id=traffic, metrics=values))
    return dict(model_key=records[0]['model_key'], condition_count=42, groups=groups,
                records=[r['summary_path'] for r in records])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--stage', choices=('matrix', 'qualify', 'formal'), default='matrix')
    parser.add_argument('--scope', type=Path, help='JSON list of models assigned to this deployment; defaults to all models')
    parser.add_argument('--batch-runtime', choices=('step', 'native'), default='step')
    parser.add_argument('--attention', choices=('native', 'laya_sdpa', 'deberta_tiled'), default='native')
    parser.add_argument('--engine-selection', type=Path)
    parser.add_argument('--author-environment', type=Path)
    args = parser.parse_args()
    catalog = json.loads((ROOT / 'scripts/model_downloads/models.json').read_text())
    keys = json.loads(args.scope.read_text()) if args.scope else list(catalog)
    if not isinstance(keys, list) or not keys or len(keys) != len(set(keys)) or any(k not in catalog for k in keys):
        parser.error('scope must be a nonempty, unique list of frozen model keys')
    if args.model not in keys:
        parser.error('model must belong to the selected deployment scope')
    rows = matrix(args.model)
    if args.stage == 'matrix':
        print(json.dumps(rows, indent=2))
        return
    directory = ROOT / 'results' / args.model / 'preparation'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'experiment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stop = ROOT / 'results/experiments/stop'
        if stop.exists():
            raise RuntimeError('Experiment stop marker is present')
        config = tomllib.loads((ROOT / 'config' / args.model / 'serving.toml').read_text())
        if args.stage == 'qualify':
            # Integrity and complete CPU encoding precede GPU model loading.
            subprocess.run([sys.executable, 'scripts/model_downloads/verify_snapshots.py',
                            '--models', args.model], cwd=ROOT, check=True)
            from scripts.experiments.preflight import cpu_preflight
            family = config['family']
            environment = ('encoder' if family in ('laya', 'gliner2', 'agent-jev') else
                           'valen' if family == 'valen' else 'open-jev' if family == 'open-jev' else 'decoder')
            cpu_preflight(ROOT, args.model, args.author_environment or
                          ROOT / 'scripts/model_downloads/environments' / environment)
        selection = load_selection(args.engine_selection)
        profile = dict(batch_runtime=args.batch_runtime, attention=args.attention, selection=selection)
        if args.stage == 'formal':
            report = require_scope_ready(ROOT, keys)[args.model]
            if report.get('execution_profile') != profile or report.get('environment') != environment_identity():
                raise RuntimeError('Serving profile or installed packages differ from qualification')
        initial = acceptance_identity(ROOT, args.model, profile)
        backend = build_backend(config, **profile)
        backend.execution_profile = profile
        backend.step_progress_path = directory / 'batch_progress.json'
        from jevlight.step_batch_support import make_step_server
        server = make_step_server(backend, config.get('host', '127.0.0.1'), config.get('port', 8123), config['model_id'])
        threading.Thread(target=server.serve_forever, daemon=True).start()
        import torch
        device = str(torch.cuda.get_device_properties(0).uuid)
        try:
            if args.stage == 'qualify':
                qualify(args.model, backend, device)
                return
            report = require_scope_ready(ROOT, keys, device)[args.model]
            if report['execution_profile'] != profile or report['environment'] != environment_identity():
                raise RuntimeError('Serving profile or installed packages differ from qualification')
            for item in report.get('capacity_cache', []):
                getattr(backend.runtime, '_step_capacity', {})[(item['co'], item['char_bucket'])] = item['capacity']
            completed = []
            for row in rows:
                if stop.exists():
                    raise RuntimeError('Experiment stop marker is present')
                if acceptance_identity(ROOT, args.model, profile) != initial:
                    raise RuntimeError('Loaded model inputs changed; requalification is required')
                record_path = ROOT / 'results/experiments/records' / (row_id(row) + '.json')
                previous = json.loads(record_path.read_text()) if record_path.exists() else {}
                if (previous.get('status') == 'completed' and previous.get('identity') == initial
                        and previous.get('execution_profile') == profile
                        and artifact_matches(previous.get('artifacts', {}))):
                    completed.append(previous)
                    continue
                result = execute(row, backend, {'jev_batch_size': report['mode_batch_sizes'][row['mode']],
                    'jev_base_url': f"http://127.0.0.1:{config.get('port', 8123)}"})
                if acceptance_identity(ROOT, args.model, profile) != initial:
                    result.update(status='needs_review', reason='Model inputs changed during execution')
                result.update(identity=initial, execution_profile=profile)
                save(record_path, result)
                if result['status'] != 'completed':
                    raise RuntimeError(f'{row_id(row)}: incomplete run; preserved for inspection')
                completed.append(result)
            save(ROOT / 'results' / args.model / 'matrix_summary.json', summarize(completed))
        finally:
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    main()
