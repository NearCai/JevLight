"""Bind qualification to model files, native sources, code and environment locks."""
import hashlib
import json
from pathlib import Path
import tomllib

from scripts.experiments.preparation_cases import source_hashes, FLOWS


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def acceptance_identity(root, key, profile=None):
    root = Path(root)
    catalog = json.loads((root / 'scripts/model_downloads/models.json').read_text())
    if key not in catalog:
        raise ValueError('Model is absent from the frozen catalog')
    config_path = root / 'config' / key / 'serving.toml'
    config = tomllib.loads(config_path.read_text())
    manifest_path = root / config['weights_path']
    manifest_path = manifest_path.parent / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('status') != 'downloaded' or manifest['revision'] != catalog[key]['revision']:
        raise ValueError('Model snapshot is incomplete or differs from the frozen revision')
    snapshots = [manifest]
    paths = [config_path, manifest_path, root / 'pyproject.toml', root / 'uv.lock']
    for base in manifest.get('bases', []):
        path = Path(base['weights_path']) / 'manifest.json'
        if not path.is_absolute():
            path = root / path
        snapshots.append(json.loads(path.read_text()))
        paths.append(path)
    files = {}
    for snapshot in snapshots:
        if snapshot.get('status') != 'downloaded':
            raise ValueError('A complete frozen backbone is required')
        weights = root / snapshot['weights_path']
        for entry in snapshot['files']:
            path = weights / entry['path']
            if not path.resolve().is_relative_to(weights.resolve()):
                raise ValueError('Snapshot file escapes its model directory')
            stat = path.stat()
            if stat.st_size != entry['size']:
                # Some authors normalize tokenizer metadata during loading;
                # verify_snapshots handles that explicitly before qualification.
                if entry['path'] != 'tokenizer/tokenizer_config.json':
                    raise ValueError(f'Frozen file size changed: {entry["path"]}')
            files[str(path.relative_to(root)) if path.is_relative_to(root) else str(path)] = {
                'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                'manifest_sha256': entry['sha256'],
            }
            if path.suffix in ('.py', '.json', '.toml', '.jinja'):
                paths.append(path)
    # Changes to adapters or orchestration require new qualification. These
    # conservative hashes never rebind an old measurement to changed code.
    for folder in ('jevlight', 'scripts/experiments', 'scripts/model_downloads'):
        paths.extend(path for path in (root / folder).rglob('*') if path.is_file()
                     and path.suffix in ('.py', '.cpp', '.h', '.toml', '.lock', '.json')
                     and not any(part in ('.git', '.venv', '__pycache__') for part in path.parts))
    sources = json.loads((root / 'scripts/model_downloads/runtime_sources.json').read_text())
    selected_sources = [sources[config['family']]] if config['family'] in sources else []
    if (profile or {}).get('selection'):
        selected_sources.append(sources['vllm-jev'])
    for source in selected_sources:
        directory = root / source['path']
        if not directory.is_dir():
            raise ValueError('Frozen author sources have not been fetched')
        paths.extend(path for path in directory.rglob('*.py') if not any(
            part in ('.git', '.venv', '__pycache__') for part in path.parts))
    extras = config.get('batch_fla_package_path')
    if extras:
        paths.extend((root / extras).rglob('*.py'))
    for directory in ('fixed_cases', 'cooperative_cases'):
        paths.append(root / 'results/decision_models/preparation' / directory / 'prompt_cases.jsonl')
    paths.extend((root / 'config' / key).glob('*.json'))
    for _, directory, route in FLOWS.values():
        paths.append(root / directory / route)
        paths.extend((root / directory).glob('*.net.xml'))
    checkpoint = (profile or {}).get('selection', {}) or {}
    export_path = checkpoint.get('checkpoint') or config.get('engine_weights_path')
    if export_path:
        export = root / export_path
        if not export.is_dir():
            raise ValueError('Engine checkpoint is missing')
        paths.extend(path for path in export.rglob('*') if path.is_file()
                     and path.suffix in ('.json', '.py', '.jinja'))
        for path in export.rglob('*.safetensors'):
            stat = path.stat()
            files[str(path)] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    return {
        'model': catalog[key], 'config': config, 'benchmark': source_hashes(),
        'files': {str(path.relative_to(root)) if path.is_relative_to(root) else str(path): digest(path)
                  for path in sorted(set(paths))},
        'snapshot_files': files,
    }
