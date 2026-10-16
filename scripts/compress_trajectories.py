"""Periodically archive completed SUMO intersection traces as verified gzip files."""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
STOP = threading.Event()
CHUNK = 1024 * 1024


def now():
    return datetime.now().astimezone().isoformat()


def fingerprint(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def eligible(path, results, min_age):
    """A summary precedes batch_log, so require a dead runner PID as well."""
    if not re.fullmatch(r"inter_\d+\.pkl", path.name):
        return False
    if not path.resolve().is_relative_to(results):
        return False
    if any(p.is_symlink() for p in (path, *path.parents) if p != results.parent):
        return False
    match = re.search(r"_pid(\d+)$", path.parent.name)
    if not match or (Path('/proc') / match[1]).exists():
        return False
    try:
        summary = json.loads((path.parent / 'summary.json').read_text())
        duration = summary.get('final/duration')
        if not isinstance(duration, (int, float)) or duration <= 0:
            return False
        return time.time() - path.stat().st_mtime >= min_age
    except (OSError, ValueError, TypeError):
        return False


def digest(stream):
    result = hashlib.sha256()
    size = 0
    while block := stream.read(CHUNK):
        result.update(block)
        size += len(block)
    return result.hexdigest(), size


def compress_file(path, results, min_age=300, level=3):
    """Publish without overwriting; delete source only after round-trip verification."""
    if not eligible(path, results, min_age):
        return None
    original = fingerprint(path)
    original_stat = path.stat()
    target = path.with_suffix('.pkl.gz')
    temporary = None
    try:
        with path.open('rb') as source:
            expected = digest(source)
        if not target.exists():
            fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
            temporary = Path(name)
            with os.fdopen(fd, 'wb') as raw:
                with gzip.GzipFile(filename='', fileobj=raw, mode='wb', compresslevel=level, mtime=0) as archive:
                    with path.open('rb') as source:
                        while block := source.read(CHUNK):
                            archive.write(block)
                raw.flush()
                os.fsync(raw.fileno())
            os.chmod(temporary, original_stat.st_mode & 0o777)
            os.utime(temporary, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
            candidate = temporary
        else:
            if target.is_symlink() or not target.is_file():
                raise ValueError('Existing archive is not a regular file')
            candidate = target
        with gzip.open(candidate, 'rb') as archive:
            if digest(archive) != expected:
                raise ValueError('Archive differs from source; original retained')
        if fingerprint(path) != original or not eligible(path, results, min_age):
            raise ValueError('Source changed or runner is active; original retained')
        if temporary is not None:
            # link is atomic and refuses to overwrite an existing destination.
            os.link(temporary, target)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
            target_stat = target.stat()
            record = dict(time=now(), source=str(path.relative_to(results)),
                          archive=str(target.relative_to(results)), sha256=expected[0],
                          original_bytes=original_stat.st_size, compressed_bytes=target_stat.st_size,
                          released_allocated_bytes=(original_stat.st_blocks - target_stat.st_blocks) * 512)
            # Persist evidence before removing the original, including on interruption.
            with (results / 'trajectory_compression' / 'files.jsonl').open('a') as audit:
                audit.write(json.dumps(record) + '\n')
                audit.flush()
                os.fsync(audit.fileno())
            path.unlink()
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return record
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_status(output, status):
    temporary = output / 'status.json.tmp'
    temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(output / 'status.json')


def scan(results, output, status, min_age, level, dry_run=False):
    status.update(state='scanning', scan_started_at=now(), candidates=0, skipped=0, scan_errors=0)
    write_status(output, status)
    for path in results.rglob('inter_*.pkl'):
        if STOP.is_set():
            break
        try:
            if not eligible(path, results, min_age):
                status['skipped'] += 1
                continue
            status['candidates'] += 1
            if dry_run:
                continue
            record = compress_file(path, results, min_age, level)
            if record:
                status['compressed_files'] += 1
                status['original_bytes'] += record['original_bytes']
                status['compressed_bytes'] += record['compressed_bytes']
                status['released_allocated_bytes'] += record['released_allocated_bytes']
                status['last_compressed'] = record['archive']
                if status['compressed_files'] % 20 == 0:
                    write_status(output, status)
        except (OSError, ValueError, EOFError) as error:
            status['scan_errors'] += 1
            with (output / 'errors.jsonl').open('a') as log:
                log.write(json.dumps(dict(time=now(), path=str(path), error=str(error))) + '\n')
    status.update(state='waiting', scan_finished_at=now())
    write_status(output, status)
    print(json.dumps(status), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/maintenance/trajectory_compression.json')
    parser.add_argument('--interval', type=int, default=600)
    parser.add_argument('--min_age', type=int, default=300)
    parser.add_argument('--compression_level', type=int, default=3)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    preliminary, _ = parser.parse_known_args(argv)
    config = json.loads(preliminary.config.read_text())
    if set(config) - {'interval', 'min_age', 'compression_level'}:
        parser.error('Unknown configuration field')
    parser.set_defaults(**config)
    args = parser.parse_args(argv)
    if args.interval < 1 or args.min_age < 0 or not 0 <= args.compression_level <= 9:
        parser.error('Invalid interval, min_age or compression_level')
    results = (ROOT / 'results').resolve()
    output = results / 'trajectory_compression'
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'compressor.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(1, 'A trajectory compressor is already running.\n')
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: STOP.set())
        os.nice(10)
        status = dict(pid=os.getpid(), started_at=now(), interval_seconds=args.interval,
                      min_age_seconds=args.min_age, compression_level=args.compression_level,
                      dry_run=args.dry_run, compressed_files=0, original_bytes=0,
                      compressed_bytes=0, released_allocated_bytes=0)
        try:
            while not STOP.is_set():
                scan(results, output, status, args.min_age, args.compression_level, args.dry_run)
                if args.once or args.dry_run:
                    break
                STOP.wait(args.interval)
        finally:
            status.update(state='stopped', stopped_at=now())
            write_status(output, status)


if __name__ == '__main__':
    main()
