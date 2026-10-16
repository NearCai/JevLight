from types import SimpleNamespace

import pytest

from jevlight.backends.decision_model_service import LayaNative

def request(index, instruction='Choose'):
    return {'state': {'id': index}, 'questions': {
        f'phase_{index}': {'type': 'choice', 'instructions': instruction,
                         'criteria': {'A': 'first', 'B': 'second'}},
        f'congested_{index}': {'type': 'noul', 'instructions': 'Congested?'}}}

def test_native_batch_groups_content_and_restores_ids_without_mutating_requests():
    calls = []

    def predict(states, questions, **kwargs):
        calls.append((states, questions, kwargs))
        return [{'answers': {'q0': {'type': 'choice', 'choice': 'A',
                                   'probabilities': {'A': 0.6, 'B': 0.4}, 'confidence': 0.6},
                             'q1': {'type': 'noul', 'noul': state['id']/10}},
                 'usage': {'truncated': False}} for state in states]

    runtime = LayaNative.__new__(LayaNative)
    runtime.agent = SimpleNamespace(predict_batch=predict)
    runtime.max_len = 8192
    runtime.head_max_len = 1024
    runtime.sort_batch_by_length = True
    inputs = [request(1), request(2, 'Different instructions'), request(3)]
    import copy
    before = copy.deepcopy(inputs)
    outputs = runtime.native_batch(inputs, 12)
    assert inputs == before
    assert [[state['id'] for state in call[0]] for call in calls] == [[1, 3], [2]]
    assert all(call[2]['batch_size'] == 12 for call in calls)
    assert all(call[2]['sort_by_length'] for call in calls)
    for index, result in enumerate(outputs, 1):
        assert list(result['answers']) == [f'phase_{index}', f'congested_{index}']
        assert result['answers'][f'congested_{index}']['noul'] == index/10

@pytest.mark.parametrize('outcomes,reason', [([], 'count mismatch'),
    ([{'usage': {'truncated': True}}], 'truncation')])
def test_native_batch_rejects_missing_or_truncated_results(outcomes, reason):
    runtime = LayaNative.__new__(LayaNative)
    runtime.agent = SimpleNamespace(predict_batch=lambda *a, **k: outcomes)
    runtime.max_len = 8192
    runtime.head_max_len = 1024
    with pytest.raises(ValueError, match=reason):
        runtime.native_batch([request(1)], 12)

def test_batch_transport_and_controller_keep_states_and_question_ids_separate():
    import threading
    from jevlight.controller import JevLightController
    from jevlight.jev_client import JevClient
    from jevlight.local_server import MockBackend, make_server
    from tests.test_jevlight_controller import make_observation

    class BatchMock(MockBackend):
        calls = []

        def evaluate_batch(self, requests, size):
            self.calls.append((requests, size))
            return [self.evaluate(item['state'], item['questions']) for item in requests]

    backend = BatchMock()
    server = make_server(backend, '127.0.0.1', 0, 'test')
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = JevClient(transport='local', batch_size=12, max_retries=0,
            base_url=f'http://127.0.0.1:{server.server_port}')
        observations = [make_observation(intersection_id=f'i{i}', queue_by_phase={'WTET': i+1})
                        for i in range(3)]
        observations.append(make_observation(intersection_id='empty', empty=True))
        controller = JevLightController(client, 15, agent_mode='jevlight', speculative=True)
        actions, traces = controller.decide(observations, 0)
        assert len(actions) == len(traces) == 4
        assert len(backend.calls) == 1
        requests, size = backend.calls[0]
        assert size == 12 and len(requests) == 3
        assert [item['state']['intersection_id'] for item in requests] == ['i0', 'i1', 'i2']
        for i, item in enumerate(requests):
            assert set(item['questions']) == {f'phase_i{i}', f'congested_i{i}'}
            assert 'intersections' not in item['state']
        assert all(trace.get('error') is None for trace in traces)
    finally:
        server.shutdown()
        server.server_close()

def test_batch_transport_rejects_missing_results_instead_of_misassigning(monkeypatch):
    from jevlight.jev_client import JevClient
    import requests
    response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'results': []})
    monkeypatch.setattr(requests, 'post', lambda *a, **k: response)
    client = JevClient(transport='local', batch_size=12, max_retries=0)
    outcomes = client.evaluate_batch([request(1), request(2)])
    assert len(outcomes) == 2
    assert all('count mismatch' in outcome.error for outcome in outcomes)

@pytest.mark.parametrize('args', [
    ['--jev_batch_size', '-1'],
    ['--jev_batch_size', '12', '--jev_transport', 'mcp'],
    ['--jev_batch_size', '12', '--jev_transport', 'local', '--agent', 'jevlight', '--packaging', 'network'],
])
def test_invalid_batch_configuration_is_rejected_before_running(args):
    from run_jevlight import parse_args
    with pytest.raises(SystemExit):
        parse_args(args)
