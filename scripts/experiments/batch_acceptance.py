"""Validation helpers for independent model requests and SUMO results."""
import json
import math
import statistics
from pathlib import Path

def compare_phases(requests, before, after, *, require_equal=True):
    """Validate independent IDs, types and candidates; equality may be disabled explicitly by the responsible queue."""
    if len(requests) != len(before) or len(before) != len(after):
        raise ValueError('batch result count mismatch')
    for request, a, b in zip(requests, before, after, strict=True):
        for result in (a, b):
            if set(result['answers']) != set(request['questions']):
                raise ValueError('batch answer IDs changed')
            for qid, question in request['questions'].items():
                answer = result['answers'][qid]
                if answer.get('type') != question['type']:
                    raise ValueError('batch answer type changed')
                if question['type'] == 'choice' and answer.get('choice') not in question['criteria']:
                    raise ValueError('batch phase is outside the original candidates')
        for qid, question in request['questions'].items():
            if require_equal and question['type'] == 'choice' and a['answers'][qid]['choice'] != b['answers'][qid]['choice']:
                raise ValueError('batch phase disagreement')

def validate_sumo_pair(pair):
    """Require successful real simulation and finite metrics, allowing new actions."""
    if len(pair) != 2 or any(row.get('status') != 'completed' for row in pair):
        raise ValueError('Real SUMO batch smoke failed')
    for row in pair:
        if any(not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key])
               for key in ('aql', 'awt', 'att')):
            raise ValueError('Real SUMO metrics missing or nonfinite')
        trace = Path(row['summary_path']).with_name('jev_decisions.jsonl')
        decisions = [json.loads(line) for line in trace.read_text().splitlines()]
        if not decisions or any(item.get('error') or
                item.get('fallback') not in (None, False, '', 'empty_junction_local_ranking')
                for item in decisions):
            raise ValueError('Real SUMO decision errors or fallback')
        summary=json.loads(Path(row['summary_path']).read_text())
        if summary.get('jev',{}).get('failures',0):
            raise ValueError('Real SUMO recorded inference failures')

def compare_batch_answers(before, after):
    """Legacy batch probes retain structural/phase checks without numeric differences."""
    import math
    if isinstance(before, dict):
        if not isinstance(after, dict) or before.keys() != after.keys():
            raise ValueError('batch answer/candidate mapping changed')
        for key in before: compare_batch_answers(before[key], after[key])
    elif isinstance(before, (int, float)):
        if not isinstance(after, (int, float)) or not math.isfinite(after):
            raise ValueError('nonfinite batch output')
    elif before != after:
        raise ValueError('batch phase/type/legend changed')
    return 0

def stats(values, size):
    ordered = sorted(values)
    return dict(p50_ms=statistics.median(values), p95_ms=ordered[max(0, math.ceil(len(ordered)*0.95)-1)], p99_ms=ordered[max(0, math.ceil(len(ordered)*0.99)-1)], samples=len(values),
                throughput_per_s=1000 * size / statistics.mean(values))
