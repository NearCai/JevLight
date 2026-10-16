"""Verify frozen model manifests, LFS hashes and ordinary Git blob identities."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

if __package__:
    from .download import CATALOG, ROOT, check_blob, digest, model_keys, save
else:
    from download import CATALOG, ROOT, check_blob, digest, model_keys, save


def native_tokenizer_normalization(path, snapshot, item, entry):
    """Recognize Laya's exact list-to-mapping fix against the published bytes."""
    if (snapshot['model_id'] not in ('convaiinnovations/laya', 'convaiinnovations/laya-multilingual')
            or item['path'] != 'tokenizer/tokenizer_config.json' or not entry.get('blobId')):
        return None
    body = path.read_bytes()
    try:
        current = json.loads(body)
        extra = current.get('extra_special_tokens')
        if (not isinstance(extra, dict) or not extra
                or list(extra) != [f'extra_{i}' for i in range(len(extra))]
                or body != json.dumps(current, indent=2).encode()):
            return None
        original = dict(current, extra_special_tokens=list(extra.values()))
        published = json.dumps(original, indent=2).encode()
    except (ValueError, TypeError, AttributeError):
        return None
    published_sha = hashlib.sha256(published).hexdigest()
    blob = hashlib.sha1(b'blob ' + str(len(published)).encode() + b'\0' + published).hexdigest()
    if (len(published) != item['size'] or published_sha != item['sha256']
            or blob != entry['blobId']
            or ((entry.get('lfs') or {}).get('sha256') not in (None, published_sha))):
        return None
    # Author laya.agent._fix_tokenizer_config performs this conversion on load.
    # Neither the checkpoint nor its frozen manifest is rewritten here.
    return dict(path=str(path), published_sha256=published_sha,
                actual_sha256=hashlib.sha256(body).hexdigest(),
                normalization='author extra_special_tokens list-to-mapping compatibility fix')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text())
    keys = model_keys(args.models, catalog)
    checked = {}
    reports = {}
    for key in keys:
        try:
            normalizations = []
            manifest = json.loads((ROOT / ".local/models" / key / "manifest.json").read_text())
            if manifest["revision"] != catalog[key]["revision"]:
                raise ValueError("checkpoint revision differs from frozen catalog")
            snapshots = [manifest] + [json.loads((Path(base["weights_path"]) / "manifest.json").read_text()) for base in manifest["bases"]]
            for snapshot in snapshots:
                if snapshot["status"] != "downloaded":
                    raise ValueError("snapshot incomplete")
                weights = Path(snapshot["weights_path"])
                metadata_path = weights / "repository_metadata.json" if weights.parent.name == snapshot["model_id"].replace("/", "--") else weights.parent / "repository_metadata.json"
                metadata = json.loads(metadata_path.read_text())
                if metadata["sha"] != snapshot["revision"]:
                    raise ValueError("repository metadata revision differs")
                entries = {entry["rfilename"]: entry for entry in metadata["siblings"]}
                for item in snapshot["files"]:
                    path = weights / item["path"]
                    if not path.resolve().is_relative_to(weights.resolve()):
                        raise ValueError("unsafe file path")
                    if path not in checked:
                        entry = entries[item["path"]]
                        hashed = digest(path)
                        if path.stat().st_size != item["size"] or hashed != item["sha256"]:
                            normalization = native_tokenizer_normalization(path, snapshot, item, entry)
                            if normalization is None:
                                raise ValueError(f"manifest mismatch: {path}")
                            normalizations.append(normalization)
                        else:
                            expected = (entry.get("lfs") or {}).get("sha256")
                            if (expected and hashed != expected) or not check_blob(path, entry):
                                raise ValueError(f"repository identity mismatch: {path}")
                        checked[path] = hashed
            reports[key] = {"status": "verified", "revision": manifest["revision"], "base_count": len(manifest["bases"])}
            if normalizations:
                reports[key]['native_metadata_normalizations'] = normalizations
        except Exception as error:
            reports[key] = {"status": "failed", "reason": f"{type(error).__name__}: {error}"}
        print(key, reports[key]["status"], flush=True)
    output = ROOT / "results/decision_models/preparation" / datetime.now().strftime("integrity_%Y%m%d_%H%M%S.json")
    save(output, {"models": reports, "unique_verified_files": len(checked)})
    raise SystemExit(int(any(report["status"] != "verified" for report in reports.values())))


if __name__ == "__main__":
    main()
