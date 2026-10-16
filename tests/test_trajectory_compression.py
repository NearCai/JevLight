import gzip
import json
import os

import pytest

from scripts import compress_trajectories as compressor
from scripts.compress_trajectories import compress_file, eligible


def trace(tmp_path, pid=999999999):
    run = tmp_path / f'run_pid{pid}'
    run.mkdir()
    (run / 'summary.json').write_text(json.dumps({'final/duration': 4200}))
    (tmp_path / 'trajectory_compression').mkdir()
    path = run / 'inter_0.pkl'
    path.write_bytes(b'complete state and action\x00' * 1000)
    return path


def test_round_trip_and_audit(tmp_path):
    path = trace(tmp_path)
    expected = path.read_bytes()
    record = compress_file(path, tmp_path, min_age=0)
    assert not path.exists()
    with gzip.open(path.with_suffix('.pkl.gz'), 'rb') as archive:
        assert archive.read() == expected
    assert record['compressed_bytes'] < record['original_bytes']
    assert json.loads((tmp_path / 'trajectory_compression/files.jsonl').read_text()) == record


def test_live_runner_and_fresh_files_are_skipped(tmp_path):
    path = trace(tmp_path, os.getpid())
    assert not eligible(path, tmp_path, 0)
    assert compress_file(path, tmp_path, min_age=0) is None
    assert path.exists()


def test_summary_required_and_age_checked(tmp_path):
    path = trace(tmp_path)
    assert not eligible(path, tmp_path, 300)
    (path.parent / 'summary.json').unlink()
    assert not eligible(path, tmp_path, 0)


def test_conflicting_archive_retains_original(tmp_path):
    path = trace(tmp_path)
    with gzip.open(path.with_suffix('.pkl.gz'), 'wb') as archive:
        archive.write(b'wrong data')
    with pytest.raises(ValueError, match='differs'):
        compress_file(path, tmp_path, min_age=0)
    assert path.exists()
    assert not list(path.parent.glob('.*.tmp'))


def test_resume_verified_archive(tmp_path):
    path = trace(tmp_path)
    with gzip.open(path.with_suffix('.pkl.gz'), 'wb') as archive:
        archive.write(path.read_bytes())
    assert compress_file(path, tmp_path, min_age=0)
    assert not path.exists()


def test_symlink_not_compressed(tmp_path):
    path = trace(tmp_path)
    link = path.parent / 'inter_1.pkl'
    link.symlink_to(path)
    assert not eligible(link, tmp_path, 0)


def test_source_changed_during_verification_is_retained(tmp_path, monkeypatch):
    path = trace(tmp_path)
    original_digest = compressor.digest
    calls = 0

    def change_source(stream):
        nonlocal calls
        result = original_digest(stream)
        calls += 1
        if calls == 2:
            path.write_bytes(b'new data written during compression')
        return result

    monkeypatch.setattr(compressor, 'digest', change_source)
    with pytest.raises(ValueError, match='Source changed'):
        compress_file(path, tmp_path, min_age=0)
    assert path.read_bytes() == b'new data written during compression'
    assert not path.with_suffix('.pkl.gz').exists()
    assert not list(path.parent.glob('.*.tmp'))
