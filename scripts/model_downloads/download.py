"""Pinned, resumable model downloads; run with uv from the repository root."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import fcntl
import json
from pathlib import Path
import shutil
import time
from urllib.parse import quote
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parents[2]
CATALOG = Path(__file__).with_name("models.json")
ENDPOINTS = ("https://hf-mirror.com", "https://huggingface.co")


class RangeUnsupported(ValueError):
    pass


def get(url, **kwargs):
    """Fetch model files with a fallback for proxy connection failures."""
    session = requests.Session()
    session.trust_env = False
    if url.startswith("https://modelscope.cn/"):
        try:
            return requests.get(url, **kwargs)
        except requests.exceptions.ProxyError:
            return session.get(url, **kwargs)
    try:
        return session.get(url, **kwargs)
    except requests.RequestException:
        return requests.get(url, **kwargs)


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def model_keys(requested, catalog):
    if requested == ["all"]:
        return list(catalog)
    return requested


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_blob(path, entry):
    if entry.get("lfs") or not entry.get("blobId"):
        return True
    body = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest() == entry["blobId"]


def save(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def metadata(model_id, revision):
    # Frozen repository metadata is reusable; moving refs such as main are not.
    cache = ROOT / ".local/models/_cache/metadata" / model_id.replace("/", "--") / f"{revision}.json"
    if len(revision) == 40:
        candidates = [cache]
        candidates.extend((ROOT / ".local/models/_bases" / model_id.replace("/", "--")).glob(f"{revision}/repository_metadata.json"))
        candidates.extend((ROOT / ".local/models").glob("*/repository_metadata.json"))
        for candidate in candidates:
            if candidate.exists():
                stored = json.loads(candidate.read_text())
                if stored.get("sha") == revision and stored.get("id") == model_id:
                    return stored, "verified_frozen_metadata_cache"
    errors = []
    for endpoint in ENDPOINTS:
        try:
            response = get(
                f"{endpoint}/api/models/{model_id}/revision/{revision}",
                params={"blobs": "true"}, timeout=(10, 20),
            )
            response.raise_for_status()
            body = response.json()
            save(cache.parent / f"{body['sha']}.json", body)
            return body, endpoint
        except requests.RequestException as error:
            errors.append(f"{endpoint}: {type(error).__name__}")
    raise RuntimeError("metadata unavailable: " + "; ".join(errors))


def selected(name, key):
    # Preserve runtime code, heads, calibration and tokenizer; omit media,
    # historical evaluation artifacts and other Laya variants.
    if name.startswith(("assets/", "eval/", "evaluation/")):
        return False
    if key == "laya" and name.startswith(("multilingual/", "typed-decisions/")):
        return False
    return not name.endswith((".png", ".gif", ".webp", ".svg", ".log"))


def parallel_transfer(url, partial, size, workers=8):
    """Download bounded ranges and retain chunks after an interrupted job."""
    chunk_size = 32 * 1024**2
    chunks = partial.with_name(partial.name + ".chunks")
    chunks.mkdir(exist_ok=True)
    def get_chunk(start):
        end = min(start + chunk_size, size) - 1
        path = chunks / str(start)
        required = end - start + 1
        if path.exists() and path.stat().st_size == required:
            return path
        for attempt in range(3):
            try:
                with get(url, headers={"Range": f"bytes={start}-{end}"},
                                  stream=True, timeout=(20, 120)) as response:
                    response.raise_for_status()
                    if response.status_code == 200:
                        raise RangeUnsupported("server does not support range downloads")
                    if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"):
                        raise ValueError("server does not support the requested range")
                    with path.open("wb") as stream:
                        for block in response.iter_content(1024**2):
                            stream.write(block)
                if path.stat().st_size != required:
                    raise ValueError("incomplete chunk")
                return path
            except RangeUnsupported:
                raise
            except (requests.RequestException, OSError, ValueError):
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        paths = list(pool.map(get_chunk, range(0, size, chunk_size)))
    with partial.open("wb") as stream:
        for path in paths:
            with path.open("rb") as source:
                shutil.copyfileobj(source, stream, 8 * 1024**2)
    shutil.rmtree(chunks)


def transfer(model_id, revision, entry, destination, modelscope_files=None):
    locks = ROOT / ".local/models/_cache/locks"
    locks.mkdir(parents=True, exist_ok=True)
    lock_name = hashlib.sha256(str((destination / entry["rfilename"]).resolve()).encode()).hexdigest()
    with (locks / lock_name).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _transfer(model_id, revision, entry, destination, modelscope_files)


def _transfer(model_id, revision, entry, destination, modelscope_files=None):
    name = entry["rfilename"]
    target = destination / name
    if not target.resolve().is_relative_to(destination.resolve()):
        raise ValueError("unsafe repository filename")
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_size = entry.get("size")
    expected_hash = (entry.get("lfs") or {}).get("sha256")
    if target.is_file() and (expected_size is None or target.stat().st_size == expected_size):
        hashed = digest(target)
        if (expected_hash is None or hashed == expected_hash) and check_blob(target, entry):
            return {"path": name, "size": target.stat().st_size, "sha256": hashed, "source": "verified_local"}
    partial = target.with_name(target.name + ".part")
    errors = []
    ms_entry = (modelscope_files or {}).get(name)
    sources = []
    if expected_hash and ms_entry and ms_entry.get("Sha256") == expected_hash:
        ms_url = (f"https://modelscope.cn/api/v1/models/{model_id}/repo?"
                  f"Revision={ms_entry['Revision']}&FilePath={quote(name, safe='')}")
        sources.append(("https://modelscope.cn", ms_url))
    sources.extend((endpoint, f"{endpoint}/{model_id}/resolve/{revision}/{quote(name, safe='/')}")
                   for endpoint in ENDPOINTS)
    for endpoint, url in sources:
        for attempt in range(3):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                if expected_size and expected_size >= 128 * 1024**2 and not offset:
                    try:
                        parallel_transfer(url, partial, expected_size)
                    except RangeUnsupported:
                        with get(url, stream=True, timeout=(20, 120)) as response:
                            response.raise_for_status()
                            with partial.open("wb") as stream:
                                for block in response.iter_content(4 * 1024**2):
                                    stream.write(block)
                else:
                    with get(url, headers={"Range": f"bytes={offset}-"} if offset else {},
                                  stream=True, timeout=(20, 120)) as response:
                        response.raise_for_status()
                        if offset and response.status_code == 206:
                            if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                                raise ValueError("invalid resume offset")
                            mode = "ab"
                        else:
                            mode = "wb"
                        with partial.open(mode) as stream:
                            for block in response.iter_content(4 * 1024 * 1024):
                                stream.write(block)
                if expected_size is not None and partial.stat().st_size != expected_size:
                    raise ValueError("file size mismatch")
                hashed = digest(partial)
                if expected_hash and hashed != expected_hash:
                    partial.unlink()
                    raise ValueError("LFS SHA256 mismatch")
                if not check_blob(partial, entry):
                    partial.unlink()
                    raise ValueError("Git blob hash mismatch")
                partial.replace(target)
                print(f"downloaded {model_id}/{name} ({target.stat().st_size} bytes) via {endpoint}", flush=True)
                return {"path": name, "size": target.stat().st_size, "sha256": hashed, "source": endpoint}
            except (requests.RequestException, OSError, ValueError) as error:
                errors.append(f"{endpoint}: {type(error).__name__}")
                time.sleep(attempt + 1)
    raise RuntimeError(f"download failed: {model_id}/{name}: {'; '.join(errors)}")


def snapshot(key, model_id, revision="main", base=False, workers=4):
    model_root = ROOT / ".local/models"
    pinned = model_root / key / "manifest.json"
    if not base and pinned.exists():
        revision = json.loads(pinned.read_text())["revision"]
    meta, source = metadata(model_id, revision)
    revision = meta["sha"]
    destination = (model_root / "_bases" / model_id.replace("/", "--") / revision
                   if base else model_root / key / "weights")
    destination.mkdir(parents=True, exist_ok=True)
    files = [entry for entry in meta["siblings"] if selected(entry["rfilename"], key)]
    modelscope_files = {}
    try:
        response = get(f"https://modelscope.cn/api/v1/models/{model_id}/repo/files",
                                params={"Revision": "master", "Recursive": "true"}, timeout=(10, 30))
        response.raise_for_status()
        modelscope_files = {entry["Path"]: entry for entry in response.json()["Data"]["Files"]
                           if entry["Type"] == "blob"}
    except (requests.RequestException, KeyError, TypeError):
        pass  # An absent domestic copy does not change the pinned HF identity.
    required = sum(entry.get("size", 0) for entry in files)
    if shutil.disk_usage(destination).free < required + 4 * 1024**3:
        raise RuntimeError("insufficient free disk space")
    manifest = {"model_id": model_id, "revision": revision, "metadata_source": source,
                "started_at": now(), "status": "downloading", "weights_path": str(destination),
                "files": [], "bases": []}
    manifest_path = destination.parent / "manifest.json" if not base else destination / "manifest.json"
    save(manifest_path, manifest)
    save(destination.parent / "repository_metadata.json" if not base else destination / "repository_metadata.json", meta)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        manifest["files"] = list(pool.map(lambda entry: transfer(model_id, revision, entry, destination, modelscope_files), files))
    for config_path in destination.rglob("adapter_config.json"):
        config = json.loads(config_path.read_text())
        base_id = config.get("base_model_name_or_path")
        checkpoint_metadata = config_path.parent.parent / "model.json"
        checkpoint = json.loads(checkpoint_metadata.read_text()) if checkpoint_metadata.exists() else {}
        base_id = base_id or checkpoint.get("model_id")
        if base_id and "/" in base_id and not base_id.startswith(("/", ".")):
            base_revision = config.get("revision")
            checkpoint_metadata = config_path.parent.parent / "model.json"
            if not base_revision and checkpoint_metadata.exists():
                checkpoint = json.loads(checkpoint_metadata.read_text())
                if checkpoint.get("model_id") == base_id:
                    base_revision = checkpoint.get("revision")
            if not base_revision:
                for provenance_name in ("training_config.json", "provenance.json", "schema_config.json"):
                    provenance_path = destination / provenance_name
                    if provenance_path.exists():
                        provenance = json.loads(provenance_path.read_text())
                        base_revision = provenance.get("base_revision") or provenance.get("args", {}).get("base_revision")
                        if not base_revision and provenance.get("model") == base_id:
                            base_revision = provenance.get("revision")
                        if base_revision:
                            break
            base_manifest = snapshot(base_id.replace("/", "--"), base_id, base_revision or "main", True, workers)
            manifest["bases"].append({"model_id": base_id, "revision": base_manifest["revision"],
                                      "weights_path": base_manifest["weights_path"]})
    if key == "agent-jev":
        base_manifest = snapshot("Qwen--Qwen3-0.6B", "Qwen/Qwen3-0.6B", "main", True, workers)
        manifest["bases"].append({"model_id": "Qwen/Qwen3-0.6B", "revision": base_manifest["revision"],
                                  "weights_path": base_manifest["weights_path"]})
    manifest.update(status="downloaded", finished_at=now())
    save(manifest_path, manifest)
    return manifest


def main(default_models=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=default_models, required=default_models is None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--catalog", type=Path, default=CATALOG,
                        help="Frozen model catalog")
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text())
    keys = model_keys(args.models, catalog)
    for key in keys:
        if key not in catalog:
            parser.error(f"unknown model: {key}")
    run_id = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    failed = False
    for key in keys:
        identity = catalog[key]
        report = {"model_key": key, "model_id": identity["model_id"], "started_at": now()}
        try:
            manifest = snapshot(key, identity["model_id"], identity["revision"], workers=args.workers)
            report.update(status="downloaded", revision=manifest["revision"])
        except Exception as error:
            failed = True
            report.update(status="blocked", reason=f"{type(error).__name__}: {error}")
        report["finished_at"] = now()
        save(ROOT / "results" / key / "preparation" / run_id / "download_report.json", report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
