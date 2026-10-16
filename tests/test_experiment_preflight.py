import json

from contextlib import nullcontext

from types import ModuleType, SimpleNamespace

import sys

import pytest

from scripts.experiments.batch_acceptance import compare_phases

def request(i):
    return {'state': {'id': i}, 'questions': {
        'phase_'+str(i): {'type': 'choice', 'instructions': 'Choose', 'criteria': {'A': 'first', 'B': 'second'}},
        'congested_'+str(i): {'type': 'noul', 'instructions': 'Congested?'}}}

def result(i, p=0.6, noul=0.2):
    return {'answers': {'phase_'+str(i): {'type': 'choice', 'choice': 'A', 'confidence': p,
        'probabilities': {'A': p, 'B': 1-p}}, 'congested_'+str(i): {'type': 'noul', 'noul': noul}}}

def test_batch_probability_changes_are_not_a_gate_but_phase_and_validity_are():
    compare_phases([request(1)], [result(1)], [result(1, 0.9, 0.8)])
    changed = result(1); changed['answers']['phase_1']['choice'] = 'B'; changed['answers']['phase_1']['probabilities'] = {'A': 0.4, 'B': 0.6}
    with pytest.raises(ValueError, match='phase disagreement'): compare_phases([request(1)], [result(1)], [changed])
    invalid = result(1); invalid['answers'].pop('congested_1')
    with pytest.raises(ValueError): compare_phases([request(1)], [result(1)], [invalid])

def test_valen_uses_author_batch_with_independent_question_ids(monkeypatch):
    from jevlight.backends.decision_model_service import ValenNative
    runtime=ValenNative.__new__(ValenNative)
    def compile(payload):
        request=payload['request']; i=json.loads(request['state'])['id']
        return SimpleNamespace(questions=[SimpleNamespace(qid=qid, index=i) for qid in request['questions']], logical_tokens=123, compute_tokens=124)
    runtime.compiler=SimpleNamespace(compile=compile)
    calls=[]
    def forward(states, **kwargs):
        calls.append(len(states)); return [[0]*len(s.questions) for s in states]
    runtime.model=SimpleNamespace(model_name='valen',forward_state_batch=forward)
    package=ModuleType('types.responses')
    package.answer=lambda q,z,t: result(q.index)['answers'][q.qid]
    monkeypatch.setitem(sys.modules,'types.responses',package)
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(inference_mode=nullcontext))
    inputs=[request(i) for i in range(3)]
    outputs=runtime.native_batch(inputs,2)
    assert calls==[2,1]
    assert [list(r['answers']) for r in outputs]==[list(r['questions']) for r in inputs]

def test_short_profile_statistics_use_actual_sample_count():
    from scripts.experiments.batch_acceptance import stats
    report = stats(list(range(1, 11)), 4)
    assert report['samples'] == 10
    assert report['p50_ms'] == 5.5
    assert report['p95_ms'] == report['p99_ms'] == 10
