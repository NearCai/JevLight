"""CPU-only frozen Open-Jev export using the author's PEFT merge operation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def export(key):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from peft.tuners.lora.layer import Linear
    from jevlight.backends.decision_model_service import local_base
    torch.set_num_threads(8)
    config = tomllib.loads((ROOT/f'config/{key}/serving.toml').read_text())
    checkpoint = ROOT/config['weights_path']/'package/checkpoint'
    metadata = json.loads((checkpoint/'model.json').read_text())
    base = Path(local_base(config, metadata['model_id'], metadata['revision']))
    adapter = checkpoint/'adapter/adapter_model.safetensors'
    options = json.loads((adapter.parent/'adapter_config.json').read_text())
    if any(options.get(k) for k in ('rank_pattern', 'alpha_pattern', 'use_dora',
                                  'fan_in_fan_out', 'lora_bias', 'use_rslora')):
        raise ValueError('The reviewed author merge requires plain uniform LoRA')
    output = ROOT/f'.local/models/_engine_exports/{key}/author_peft_merged'
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    with safe_open(adapter, framework='pt', device='cpu') as archive:
        tensors = {name: archive.get_tensor(name) for name in archive.keys()}
    pairs = {}
    for name, tensor in tensors.items():
        if tensor.dtype != torch.float32:
            raise ValueError('The frozen released adapter tensor precision differs')
        stem, kind = name.removeprefix('base_model.model.').rsplit('.lora_', 1)
        if kind not in ('A.weight', 'B.weight'):
            raise ValueError(f'Unreviewed adapter tensor: {name}')
        pairs.setdefault('model.language_model.'+stem+'.weight', {})[kind[0]] = tensor
    index = json.loads((base/'model.safetensors.index.json').read_text())
    if not pairs.keys() <= index['weight_map'].keys():
        raise ValueError('A released adapter target is absent from the frozen base')
    retained = {name: shard for name, shard in index['weight_map'].items()
                if name.startswith('model.language_model.')}
    exported, files, verified = {}, {}, []
    with torch.inference_mode():
        for number, source_name in enumerate(sorted(set(retained.values()))):
            state = {}
            with safe_open(base/source_name, framework='pt', device='cpu') as archive:
                for name in archive.keys():
                    if name not in retained:
                        continue
                    weight = archive.get_tensor(name)
                    if name in pairs:
                        a, b = pairs[name]['A'], pairs[name]['B']
                        proxy = SimpleNamespace(lora_A={'frozen': SimpleNamespace(weight=a)},
                            lora_B={'frozen': SimpleNamespace(weight=b)}, fan_in_fan_out=False,
                            scaling={'frozen': options['lora_alpha']/options['r']})
                        delta = Linear.get_delta_weight(proxy, 'frozen')
                        if delta.shape != weight.shape or weight.dtype != torch.bfloat16:
                            raise ValueError(f'Author merge shape/precision differs: {name}')
                        # Same two BF16 rounding points as PEFT safe_merge.
                        weight = weight.clone()
                        weight.add_(delta.to(weight.dtype))
                        if not torch.isfinite(weight).all():
                            raise ValueError(f'Nonfinite merged author target: {name}')
                        verified.append(name)
                    state[name] = weight.contiguous()
            if not state:
                continue
            destination = output/f'model-{number:05d}.safetensors'
            save_file(state, str(destination), metadata={'format': 'pt'})
            exported.update({name: destination.name for name in state})
            stat = destination.stat()
            files[destination.name] = dict(sha256=digest(destination), size=stat.st_size,
                                           mtime_ns=stat.st_mtime_ns)
            print(json.dumps(dict(shard=destination.name, merged_targets=len(verified))), flush=True)
    if set(verified) != set(pairs):
        raise ValueError('Not every released adapter target was merged')
    for path in base.iterdir():
        if path.name != 'manifest.json' and path.suffix in ('.json', '.jinja', '.txt'):
            if path.name != 'model.safetensors.index.json':
                shutil.copyfile(path, output/path.name)
    (output/'model.safetensors.index.json').write_text(json.dumps(
        dict(metadata={'total_size': sum(v['size'] for v in files.values())}, weight_map=exported), indent=2)+'\n')
    manifest = dict(status='ready', model_key=key, base=str(base), checkpoint=str(checkpoint),
        source_adapter_sha256=digest(adapter), source_head_sha256=digest(checkpoint/'head.pt'),
        source_temperature_sha256=digest(checkpoint/'temperature.json'),
        merge_implementation='PEFT Linear.get_delta_weight + safe_merge BF16 addition',
        author_export_reference='https://github.com/mode-io/vllm-jev/blob/main/vllm_jev/export.py',
        adapter_source_dtype='float32', merged_backbone_dtype='bfloat16',
        released_adapter_tensors_verified=len(tensors), merged_targets=verified, files=files)
    (output/'engine_export_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(status='ready', output=str(output), tensors=len(tensors))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    export(parser.parse_args().model)
