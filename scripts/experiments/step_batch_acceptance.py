"""Brief whole-step capacity qualification; adapt memory during real runs."""
import hashlib
import json
from pathlib import Path
import statistics
import time
from scripts.experiments.acceptance_identity import acceptance_identity
from scripts.experiments.batch_acceptance import compare_phases, stats, validate_sumo_pair
from jevlight.step_batch_support import encoded_request_length, prepare_step_backend


def junction_count(traffic):
    return 196 if traffic.startswith('newyork') else 16 if traffic.startswith('hangzhou') else 12


def step_identity(root, key, device, profile=None):
    import torch
    return dict(inputs=acceptance_identity(root, key, profile), device=device, execution_profile=profile or {},
        gpu_uuid=str(torch.cuda.get_device_properties(0).uuid), torch=torch.__version__,
        cuda=torch.version.cuda,
        policy='whole_step_then_graded_memory_v1')


def profile_short_step(runtime, representatives, entry, budget_s=120):
    """Keep a full 3/10 profile within budget without capping whole-step rows."""
    import torch
    if budget_s <= 0 or len(representatives) < 2:
        raise ValueError('Short profile needs a positive budget and multiple requests')
    started = time.monotonic()
    group = list(representatives)
    warm = []
    entry['short_profile_budget_s'] = budget_s
    entry['short_profile_screening'] = []

    def pair():
        torch.cuda.synchronize(); start = time.monotonic()
        baseline = [runtime.native(request) for request in group]
        torch.cuda.synchronize(); serial = (time.monotonic()-start)*1000
        start = time.monotonic()
        result = runtime.native_batch(group, 196)
        torch.cuda.synchronize(); batch = (time.monotonic()-start)*1000
        compare_phases(group, baseline, result, require_equal=False)
        entry['short_profile_last_pair'] = dict(requests=len(group), serial_ms=serial,
            batch_ms=batch, elapsed_s=time.monotonic()-started)
        if time.monotonic()-started > budget_s:
            raise ValueError('Required 3/10 profile exceeded time budget; incomplete profile rejected')
        return serial, batch

    while len(warm) < 3:
        sample = pair()
        entry['short_profile_screening'].append(dict(requests=len(group), serial_ms=sample[0], batch_ms=sample[1]))
        warm.append(sample)
        pending = (13-len(warm))*statistics.mean(serial+batch for serial,batch in warm)/1000
        remaining = budget_s-(time.monotonic()-started)
        if pending*1.1 > remaining:
            if len(group) == 2:
                raise ValueError('Required 3/10 profile exceeds time budget; no partial profile used for adoption')
            group = group[:-1]
            warm = []
    # A small representative group may lose to scalar calls while a complete
    # control step still accelerates. Record this; actual whole-step SUMO
    # speedup decides adoption. Keep this profile bounded and never extend it.
    entry['short_profile_initial_speedup'] = (statistics.mean(serial for serial,_ in warm)
                                            / statistics.mean(batch for _,batch in warm))
    serial_times, batch_times = [], []
    for _ in range(10):
        serial, batch = pair()
        serial_times.append(serial); batch_times.append(batch)
    entry.update(short_profile=dict(warmup=3, measure=10, requests=len(group),
        serial=stats(serial_times,len(group)), batch=stats(batch_times,len(group)),
        speedup=statistics.mean(serial_times)/statistics.mean(batch_times)),
        short_profile_elapsed_s=time.monotonic()-started,
        short_profile_gpu_rows=runtime.last_step_batch['gpu_forward_rows'])


def recorded_profile_cases(root, key, mode, benchmark):
    """Read complete actual short-step inputs; never reuse their old answers."""
    root=Path(root)
    reports=sorted((root/'results'/key/'preparation').glob('step_batch/*/report.json'),reverse=True)
    selected_logs=set();by_traffic={};seen=set();cases=[]
    for report_path in reports:
        report=json.loads(report_path.read_text())
        if (report.get('native_input_mapping_valid') is not True
                or report.get('encoded_case_count')!=98
                or report.get('identity',{}).get('inputs',{}).get('benchmark')!=benchmark):
            continue
        for smoke in report.get('modes',{}).get(mode,{}).get('smoke',[]):
            traffic=smoke.get('traffic_id')
            if (smoke.get('status')!='completed' or smoke.get('simulation_duration_s')!=60
                    or smoke.get('model_decisions',0)<=0
                    or smoke.get('hashes',{}).get('benchmark')!=benchmark
                    or by_traffic.get(traffic,0)>=2):
                continue
            log=Path(smoke['attempt_path'])/'native_requests.jsonl'
            if log in selected_logs or not log.is_file():continue
            selected_logs.add(log);by_traffic[traffic]=by_traffic.get(traffic,0)+1
            for line in log.read_text().splitlines():
                request=json.loads(line)['request']
                digest=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()
                if digest in seen:continue
                seen.add(digest)
                cases.append(dict(request=request,source=str(log),traffic_id=traffic,request_sha256=digest))
    return cases


def longest_recorded(root, key, mode, expected_hashes=None):
    """Use an actual grown request too, rather than only synthetic lengths."""
    root = Path(root)
    directories = [root/'results/experiments/records']
    directories.extend((root/'results/experiments').glob('gpu*/records'))
    candidates = (path for directory in directories for path in directory.glob(f'{key}_{mode}_*.json'))
    longest = None
    seen = set()
    for path in candidates:
        record = json.loads(path.read_text())
        if expected_hashes and any(record.get('hashes', {}).get(k) != v for k, v in expected_hashes.items()):
            continue
        trace = Path(record.get('attempt_path', ''))/'native_requests.jsonl'
        if not trace.is_file() or trace.resolve() in seen:
            continue
        seen.add(trace.resolve())
        with trace.open() as stream:
            for line in stream:
                # A running worker may still be appending its last record.
                if not line.endswith('\n'):
                    continue
                item = json.loads(line)
                if not item.get('encoding') or 'request' not in item:
                    continue
                tokens = encoded_request_length(item['encoding'])
                if longest is None or tokens > longest[0]:
                    longest = tokens, item['request'], str(trace)
    return longest


def select_step_batch(root, key, backend, rows, execute, device):
    import torch
    root = Path(root)
    prepare_step_backend(backend)
    identity = step_identity(root, key, device, getattr(backend, 'execution_profile', {}))
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output = root/'results'/key/'preparation/step_batch'/digest
    output.mkdir(parents=True, exist_ok=True)
    backend.step_progress_path = output/'batch_progress.json'
    backend.qualification_report_path = output/'report.json'
    path = output/'report.json'
    if path.exists():
        prior = json.loads(path.read_text())
        if prior.get('identity') == identity and prior.get('status') == 'ready':
            from scripts.experiments.model_experiments import report_ready
            if report_ready(prior, identity['inputs'], identity['gpu_uuid']):
                for item in prior.get('capacity_cache',[]):
                    getattr(backend.runtime, '_step_capacity', {})[(item['co'],item['char_bucket'])]=item['capacity']
                return prior['mode_batch_sizes']
    sizes = dict(jevlight=0, cojevlight=0, per_flow={})
    report = dict(identity=identity, status='testing', mode_batch_sizes=sizes, modes={},
                  scope='one HTTP request per complete time step; native GPU batch first, graded memory capacity')
    def save():
        path.write_text(json.dumps(report, indent=2)+'\n')
    save()
    if not hasattr(backend.runtime, 'probe_step_capacity'):
        report.update(status='blocked', reason='Author independent-input GPU batch requires implementation/inspection')
        save(); return sizes
    for mode, directory in [('jevlight', 'fixed_cases'), ('cojevlight', 'cooperative_cases')]:
        entry = report['modes'][mode] = dict(status='testing')
        save()
        cases = [json.loads(s)['request'] for s in (root/'results/decision_models/preparation'/directory/'prompt_cases.jsonl').read_text().splitlines()]
        try:
            lengths = [encoded_request_length(backend.runtime.encode(r)) for r in cases]
            worst = cases[max(range(len(cases)), key=lengths.__getitem__)]
            inputs = identity['inputs']
            expected = dict(benchmark=inputs['benchmark'],
                # Historical execution records store the complete file hash,
                # while model acceptance now scopes unrelated adapter branches.
                service_code=hashlib.sha256((root/'jevlight/backends/decision_model_service.py').read_bytes()).hexdigest(),
                weights_manifest=inputs['files'][f'.local/models/{key}/manifest.json'])
            actual = longest_recorded(root, key, mode, expected)
            actual_tokens = encoded_request_length(backend.runtime.encode(actual[1])) if actual else 0
            if actual and actual_tokens > max(lengths):
                worst = actual[1]
            entry['largest_encoded_tokens'] = max(max(lengths), actual_tokens)
            entry['actual_input_source'] = actual[2] if actual and actual_tokens > max(lengths) else None
            candidates=[dict(request=request,source=str(root/'results/decision_models/preparation'/directory/'prompt_cases.jsonl'),encoded_tokens=length)
                        for request,length in zip(cases,lengths,strict=True)]
            for actual_case in recorded_profile_cases(root,key,mode,inputs['benchmark']):
                candidate=dict(actual_case,encoded_tokens=encoded_request_length(backend.runtime.encode(actual_case['request'])))
                candidates.append(candidate)
                if candidate['encoded_tokens']>entry['largest_encoded_tokens']:
                    worst=candidate['request']
                    entry.update(largest_encoded_tokens=candidate['encoded_tokens'],actual_input_source=candidate['source'])
            reference = backend.runtime.native(worst)
            sample, cap, attempts = backend.runtime.probe_step_capacity([worst]*196, 196)
            compare_phases([worst]*len(sample), [reference]*len(sample), sample, require_equal=False)
            entry.update(full_step_target=196, largest_safe_probe_capacity=cap,
                         capacity_attempts=attempts, capacity_probe_mapping_valid=True, phase_equality_required=False)
            save()
            # Profile a short representative group. Capacity probes and real
            # SUMO below verify step coverage; do not run 13 expensive full grids.
            # Actual boundary junctions can be shorter than the synthetic
            # high-degree cases. Re-encode them completely with this backend
            # and recompute native phases on every timed pair. The longest
            # input and 196-request capacity probe remain independent.
            short=sorted(candidates,key=lambda row:row['encoded_tokens'])[:4]
            group=[row['request'] for row in short]
            entry['short_profile_input_sources']=[{k:v for k,v in row.items() if k!='request'} for row in short]
            backend.runtime.native_batch(group[:1],196)
            single_gpu_rows=max(backend.runtime.last_step_batch['gpu_forward_rows'],default=0)
            profile_short_step(backend.runtime,group,entry,
                               float(backend.config.get('batch_profile_budget_s',120)))
            entry['single_request_gpu_rows']=single_gpu_rows
            if max(entry['short_profile_gpu_rows'],default=0)<=max(1,single_gpu_rows):
                raise ValueError('No GPU junction batch growth over the native single-request forward')
            save()
            traffic = 'newyork-2' if mode == 'jevlight' else 'jinan-2'
            row = dict(next(r for r in rows if r['mode']==mode and r['traffic_id']==traffic and r['seed']==1))
            pair = [execute(row, backend, dict(count=60,jev_batch_size=size,
                jev_base_url=f"http://127.0.0.1:{backend.config['port']}",
                memo=f'step_batch_gate_{mode}_{device}_{size}_{digest[:12]}',rolling_evaluation=True)) for size in (0,196)]
            entry['smoke'] = pair
            if any(r['status']!='completed' for r in pair):
                raise ValueError('Whole-step SUMO smoke failed')
            validate_sumo_pair(pair)
            entry['smoke_speedup'] = pair[0]['wall_duration_s']/pair[1]['wall_duration_s']
            if entry['smoke_speedup'] <= 1:
                raise ValueError('Whole-step batch slower in actual SUMO smoke')
            entry.update(status='ready', sumo_behavior_valid=True, phase_equality_required=False)
            sizes[mode] = 196
            entry['transport_progress']='GPU chunk completion; original client read timeout unchanged'
        except (RuntimeError, ValueError) as error:
            entry.update(status='blocked',reason=str(error))
            torch.cuda.empty_cache()
        save()
    report['status'] = 'ready' if all(sizes[m] for m in ('jevlight','cojevlight')) else 'blocked'
    report['capacity_cache']=[dict(co=k[0],char_bucket=k[1],capacity=v) for k,v in getattr(backend.runtime, '_step_capacity', {}).items()]
    artifacts = {}
    for mode in report['modes'].values():
        for smoke in mode.get('smoke', []):
            if smoke.get('summary_path'):
                for artifact in (Path(smoke['summary_path']), Path(smoke['summary_path']).with_name('jev_decisions.jsonl')):
                    if artifact.is_file():
                        artifacts[str(artifact)] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    report['artifacts'] = artifacts
    save(); return sizes
