"""Ordered TypeSafe credentials; private vault, transactional public usage ledger.

Only HTTP 402 advances the pool. USD balances are never inferred from tokens.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VAULT = ROOT / '.local/api/jev_keys.json'
DEFAULT_LEDGER = ROOT / 'results/jev-api/credentials/usage.sqlite3'
POOL_ENV = 'JEV_API_KEY_POOL'


class PoolExhausted(RuntimeError):
    pass


class KeyPool:
    def __init__(self, vault, ledger=None):
        self.vault = Path(vault).resolve()
        if self.vault.stat().st_mode & 0o077:
            raise ValueError('Credential vault must have mode 0600')
        data = json.loads(self.vault.read_text())
        if data.get('version') != 1 or not data.get('keys'):
            raise ValueError('Unsupported or empty credential vault')
        self.keys = data['keys']
        if len({k['id'] for k in self.keys}) != len(self.keys):
            raise ValueError('Duplicate credential identifiers')
        self.ledger = Path(ledger or data['ledger']).resolve()
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS keys (id TEXT PRIMARY KEY, exhausted INTEGER NOT NULL DEFAULT 0)')
            db.execute('''CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY, time REAL, pid INTEGER, key_id TEXT,
                status INTEGER, success INTEGER, input_tokens INTEGER, output_tokens INTEGER,
                latency REAL, error TEXT)''')
            db.executemany('INSERT OR IGNORE INTO keys(id) VALUES (?)', [(k['id'],) for k in self.keys])

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.ledger, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def select(self):
        with self.connect() as db:
            exhausted = {r['id'] for r in db.execute('SELECT id FROM keys WHERE exhausted=1')}
        for key in self.keys:
            if key['id'] not in exhausted:
                return key
        raise PoolExhausted('All configured TypeSafe API keys have exhausted credits (HTTP 402)')

    def record(self, key, result):
        # Store neither credentials nor request headers, even if an exception echoes a key.
        error = result.error
        if error:
            for entry in self.keys:
                error = error.replace(entry['secret'], '[REDACTED]')
            result.error = error
        with self.connect() as db:
            db.execute('INSERT INTO attempts(time,pid,key_id,status,success,input_tokens,output_tokens,latency,error) VALUES (?,?,?,?,?,?,?,?,?)',
                       (time.time(), os.getpid(), key['id'], result.http_status, int(not result.error),
                        result.input_tokens, result.output_tokens, result.latency, error))
            if result.http_status == 402:
                # Mark the key used by THIS call, never blindly advance a shared pointer.
                # Late concurrent failures for an old key cannot skip the next key.
                db.execute('UPDATE keys SET exhausted=1 WHERE id=?', (key['id'],))

    def evaluate(self, call):
        started = time.time()
        for attempt in range(1, len(self.keys) + 1):
            key = self.select()
            result = call(key)
            result.credential_id = key["id"]
            result.credential_attempts = attempt
            self.record(key, result)
            if result.http_status != 402:
                result.latency = time.time() - started
                return result
        raise PoolExhausted('All configured TypeSafe API keys have exhausted credits (HTTP 402)')

    def status(self):
        with self.connect() as db:
            states = {r['id']: r['exhausted'] for r in db.execute('SELECT * FROM keys')}
            usage = {r['key_id']: dict(r) for r in db.execute('''SELECT key_id, COUNT(*) AS calls,
                SUM(success) AS successful_calls, SUM(input_tokens) AS input_tokens,
                SUM(output_tokens) AS output_tokens FROM attempts GROUP BY key_id''')}
        active = next((k['id'] for k in self.keys if not states[k['id']]), None)
        return {'version': 1, 'policy': 'sequential_until_http_402', 'active_key': active,
                'balance_query': 'not_queried',
                'keys': [{'id': k['id'], 'declared_initial_usd': k.get('declared_initial_usd'),
                          'balance_source': 'http_402' if states[k['id']] else 'unknown',
                          'remaining_usd': None,  # HTTP 402 establishes unavailability, not an exact USD balance.
                          'state': 'exhausted' if states[k['id']] else ('active' if k['id'] == active else 'standby'),
                          **{n: v for n, v in usage.get(k['id'], {}).items() if n != 'key_id'}} for k in self.keys]}


def migrate(source, vault=DEFAULT_VAULT, ledger=DEFAULT_LEDGER, declared_usd=None):
    source, vault = Path(source), Path(vault)
    secrets = list(dict.fromkeys(re.findall(r'apikey_[A-Za-z0-9_]+', source.read_text())))
    if not secrets:
        raise ValueError('No TypeSafe credentials found in source')
    if vault.exists():
        raise ValueError('Vault already exists; refusing to overwrite credentials or reset order')
    vault.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(vault.parent, 0o700)
    data = {'version': 1, 'ledger': str(Path(ledger).resolve()),
            'keys': [{'id': 'key-' + hashlib.sha256(s.encode()).hexdigest()[:12],
                      'secret': s, 'declared_initial_usd': declared_usd} for s in secrets]}
    fd = os.open(vault, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    pool = KeyPool(vault)
    if [k['secret'] for k in pool.keys] != secrets:
        raise RuntimeError('Credential migration verification failed')
    # Replace the old plaintext credential list only after verified durable storage.
    os.chmod(source, 0o600)
    source.write_text('Credentials migrated to jev_keys.json (0600).\n'
                      'Status: uv run python -m jevlight.api_key_pool status\n'
                      'Usage ledger: results/jev-api/credentials/usage.sqlite3\n')
    return pool.status()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['status', 'migrate'])
    parser.add_argument('--vault', type=Path, default=DEFAULT_VAULT)
    parser.add_argument('--source', type=Path, default=ROOT / '.local/api/jev.txt')
    args = parser.parse_args()
    status = migrate(args.source, args.vault) if args.action == 'migrate' else KeyPool(args.vault).status()
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
