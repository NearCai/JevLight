from concurrent.futures import ThreadPoolExecutor
import json
import threading

import pytest

from jevlight.api_key_pool import KeyPool, PoolExhausted, migrate
from jevlight.jev_client import JevClient, JevResult


def make_pool(tmp_path, n=5):
    source = tmp_path / 'old.txt'
    source.write_text('\n'.join('apikey_test_' + str(i) for i in range(n)))
    vault = tmp_path / 'private' / 'keys.json'
    ledger = tmp_path / 'results' / 'usage.sqlite3'
    migrate(source, vault, ledger)
    assert 'apikey_' not in source.read_text()
    assert vault.stat().st_mode & 0o777 == 0o600
    return KeyPool(vault)


def test_order_restart_and_no_secrets(tmp_path):
    pool = make_pool(tmp_path)
    seen = []
    def call(key):
        seen.append(key['id'])
        if key == pool.keys[0]:
            return JevResult(error='402 ' + key['secret'], http_status=402)
        return JevResult(input_tokens=80, output_tokens=5)
    pool.evaluate(call)
    assert seen == [k['id'] for k in pool.keys[:2]]
    restarted = KeyPool(pool.vault)
    assert restarted.select()['id'] == pool.keys[1]['id']
    status = restarted.status()
    assert status['keys'][0]['remaining_usd'] is None
    assert status['keys'][0]['state'] == 'exhausted'
    assert status['keys'][1]['remaining_usd'] is None
    assert status['keys'][1]['input_tokens'] == 80
    assert b'apikey_' not in pool.ledger.read_bytes()
    assert 'secret' not in json.dumps(status)


@pytest.mark.parametrize('status', [401,403,422,429,500,None])
def test_other_errors_do_not_rotate(tmp_path, status):
    pool = make_pool(tmp_path)
    calls = []
    result = pool.evaluate(lambda key: (calls.append(key['id']) or JevResult(error='failure', http_status=status)))
    assert result.error == 'failure'
    assert calls == [pool.keys[0]['id']]
    assert pool.select() == pool.keys[0]


def test_concurrent_late_402_does_not_skip_keys(tmp_path):
    pool = make_pool(tmp_path)
    barrier = threading.Barrier(8)
    def worker(_):
        other = KeyPool(pool.vault)
        def call(key):
            if key == pool.keys[0]:
                barrier.wait(timeout=10)
                return JevResult(error='credits exhausted', http_status=402)
            assert key == pool.keys[1]
            return JevResult(input_tokens=10)
        return other.evaluate(call)
    with ThreadPoolExecutor(max_workers=8) as workers:
        results = list(workers.map(worker, range(8)))
    assert all(not r.error for r in results)
    assert pool.select() == pool.keys[1]
    assert pool.status()['keys'][1]['successful_calls'] == 8


def test_all_exhausted_bounded(tmp_path):
    pool = make_pool(tmp_path)
    calls = []
    with pytest.raises(PoolExhausted):
        pool.evaluate(lambda key: (calls.append(key['id']) or JevResult(error='credits exhausted', http_status=402)))
    assert calls == [k['id'] for k in pool.keys]
    with pytest.raises(PoolExhausted):
        KeyPool(pool.vault).select()


def test_sdk_rotation_preserves_input_and_client_keys(tmp_path, monkeypatch):
    import typesafe_sdk
    from types import SimpleNamespace
    pool = make_pool(tmp_path)
    seen = []
    class FakeSDK:
        def __init__(self, api_key, **kwargs):
            self.key = api_key
        def system_one(self, **kwargs):
            seen.append((self.key, kwargs))
            if self.key == pool.keys[0]['secret']:
                raise typesafe_sdk.TypeSafeAPIError(402, {'message': 'credits exhausted'}, {})
            return SimpleNamespace(answers={}, usage=SimpleNamespace(input_tokens=33, output_tokens=2), model='jev-test')
    monkeypatch.setattr(typesafe_sdk, 'TypeSafeClient', FakeSDK)
    monkeypatch.setenv('JEV_API_KEY_POOL', str(pool.vault))
    client = JevClient(transport='official')
    result = client.evaluate({'full': 'state'}, {'q': {'type': 'noul'}})
    assert not result.error
    assert result.input_tokens == 33
    assert seen[0][1] == seen[1][1]
    assert [s[0] for s in seen] == [k['secret'] for k in pool.keys[:2]]
    assert client._pool_clients[pool.keys[0]['id']].api_key == pool.keys[0]['secret']
    local = JevClient(transport='local')
    assert local._key_pool is None and local.api_key is None
    with pytest.raises(ValueError, match='official TypeSafe endpoint'):
        JevClient(transport='official', base_url='http://localhost:1234')


def test_rest_402_does_not_repeat_on_exhausted_key(tmp_path, monkeypatch):
    import requests
    from types import SimpleNamespace
    pool = make_pool(tmp_path)
    monkeypatch.delenv('JEV_API_KEY_POOL', raising=False)
    client = JevClient(transport='official', api_key='apikey_test', max_retries=3, key_pool=False)
    client._sdk_client = None
    calls = []
    def post(*args, **kwargs):
        calls.append(kwargs)
        response = SimpleNamespace(status_code=402, headers={})
        def fail():
            raise requests.HTTPError('402 Payment Required', response=response)
        response.raise_for_status = fail
        return response
    monkeypatch.setattr(requests, 'post', post)
    result = client.evaluate({'input': 'complete'}, {'q': {'type': 'noul'}})
    assert result.http_status == 402
    assert len(calls) == 1


def test_vault_permissions_and_duplicate_migration(tmp_path):
    pool = make_pool(tmp_path)
    source = tmp_path / 'more.txt'
    source.write_text('apikey_new')
    with pytest.raises(ValueError, match='already exists'):
        migrate(source, pool.vault, pool.ledger)
    assert source.read_text() == 'apikey_new'
    pool.vault.chmod(0o644)
    with pytest.raises(ValueError, match='0600'):
        KeyPool(pool.vault)
