"""Resolve complete native context budgets across model families."""
import json
import ast
from pathlib import Path
import re
import tomllib


def prepare_context(root, key):
    root = Path(root)
    serving = root / 'config' / key / 'serving.toml'
    config = tomllib.loads(serving.read_text())
    weights = root / config['weights_path']
    manifest = json.loads((weights.parent / 'manifest.json').read_text())
    release, base = {}, None
    family = config['family']
    state_budget = None
    if family == 'open-jev':
        release = json.loads((weights / 'package/checkpoint/model.json').read_text())
        base = next(b for b in manifest['bases'] if b['model_id'] == release['model_id']
                    and b['revision'].startswith(release['revision']))
        native = json.loads((Path(base['weights_path']) / 'config.json').read_text())
        supported = native.get('text_config', native)['max_position_embeddings']
    elif family == 'valen':
        native = json.loads((weights / 'config.json').read_text())
        supported = native['valen_max_length']  # Dedicated compiler cap, not backbone window.
    elif family in ('laya', 'von', 'agent-jev', 'nimble', 'gliner2'):
        if family == 'nimble':
            native = json.loads((weights / 'schema_config.json').read_text())
            supported = native['max_length']
        elif family == 'gliner2':
            native = json.loads((weights / 'config.json').read_text())
            # Relative-attention position buckets are not an absolute token cap.
            # Keep the existing serving budget, validated by full native encoding/GPU gates.
            encoder = native['encoder_config']
            if encoder.get('position_biased_input', True):
                supported = encoder['max_position_embeddings']
            else:
                supported = config['max_length']
        else:
            path = weights / ('encoder/config.json' if family == 'laya' else 'config.json')
            native = json.loads(path.read_text())
            supported = native.get('text_config', native)['max_position_embeddings']
    elif family == 'kev':
        source = json.loads((root / 'scripts/model_downloads/runtime_sources.json').read_text())['kev']['path']
        tree = ast.parse((root / source / 'kev/model.py').read_text())
        values = {}
        def constant(node):
            if isinstance(node, ast.Constant) and type(node.value) is int:
                return node.value
            if isinstance(node, ast.Name):
                return values[node.id]
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                return constant(node.left) + constant(node.right)
            raise ValueError('Unsupported native context constant')
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in ('SERVE_MAX_STATE', 'SERVE_MAX_BRANCH'):
                    values[name] = constant(node.value)
        state_budget, supported = values['SERVE_MAX_STATE'], values['SERVE_MAX_BRANCH']
    elif family in ('decider', 'clef', 'lev'):
        config_path = weights / 'config.json'
        if family == 'lev':
            adapter = json.loads((weights / 'adapter_config.json').read_text())
            base = next(b for b in manifest['bases'] if b['model_id'] == adapter['base_model_name_or_path'])
            config_path = Path(base['weights_path']) / 'config.json'
        native = json.loads(config_path.read_text())
        supported = native.get('text_config', native)['max_position_embeddings']
        if family == 'decider':
            state_budget = supported
    else:
        raise ValueError(f'Native context policy not audited for {family}')
    if type(supported) is not int or supported < 1:
        raise ValueError('Invalid model-supported context')
    body, count = re.subn(r'(?m)^max_length\s*=\s*\d+\s*$', f'max_length = {supported}', serving.read_text())
    if count != 1:
        raise ValueError('Expected one explicit serving context field')
    if state_budget is not None:
        body, replaced = re.subn(r'(?m)^max_state_tokens\s*=\s*\d+\s*$', f'max_state_tokens = {state_budget}', body)
        if replaced != 1: raise ValueError('Missing native state budget field')
    if body != serving.read_text():
        serving.write_text(body)
    return dict(model_key=key, supported_window=supported, previous_window=config['max_length'],
                release_training_window=release.get('max_length'), native_state_budget=state_budget, base=base,
                validation='Window resolution only; complete encoding and real GPU acceptance must follow.')
