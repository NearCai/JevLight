"""CPU-only audit of benchmark requests through native encoders; never loads weights.

Run in the model's existing uv preparation environment. Reports are length/schema
checks, not inference acceptance or traffic performance results.
"""
import argparse
from copy import deepcopy
import importlib
import hashlib
import json
import os
from pathlib import Path
import sys
import tomllib
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


def cpu_runtime(config):
    import torch
    from transformers import AutoTokenizer
    import jevlight.backends.decision_model_service as service
    torch.set_num_threads(1)
    family = {'agent-jev': 'agent', 'open-jev': 'openjev'}.get(config['family'], config['family'])
    weights = ROOT / config['weights_path']
    tokenizer_path = weights / 'tokenizer' if family == 'laya' else weights
    if not (tokenizer_path / 'tokenizer_config.json').exists():
        manifest = json.loads((weights.parent / 'manifest.json').read_text())
        tokenizer_path = next((Path(b['weights_path']) for b in manifest.get('bases', [])
                               if (Path(b['weights_path']) / 'tokenizer_config.json').exists()), tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    classes = dict(laya=service.LayaNative, kev=service.KevNative, agent=service.AgentNative,
                   decider=service.DeciderNative, clef=service.ClefNative, valen=service.ValenNative,
                   openjev=service.OpenJevNative, nimble=service.NimbleNative, von=service.VonNative,
                   lev=service.LevNative, gliner2=service.GlinerNative)
    # Use the very same encode methods as serving, supplying only encoding dependencies.
    runtime = classes[family].__new__(classes[family])
    runtime.audit_tokenizer_path = str(tokenizer_path)
    runtime.max_length = config['max_length']
    if family == 'laya':
        from laya.agent import Agent
        encoder_config = json.loads((weights / 'config.json').read_text())
        runtime.agent = SimpleNamespace(tok=tokenizer, _to_internal=Agent._to_internal,
            model=SimpleNamespace(encoder=SimpleNamespace(config=SimpleNamespace(
                max_position_embeddings=encoder_config.get('max_position_embeddings', 8192)))))
        runtime.max_len = config['max_length']
        runtime.head_max_len = config['head_max_length']
    elif family == 'agent':
        service.add_author_source('agent-jev')
        runtime.engine = SimpleNamespace(tokenizer=tokenizer, max_tokens=config['max_length'])
    elif family == 'kev':
        service.add_author_source('kev')
        from kev.model import encode
        runtime.model = SimpleNamespace(encode=encode)
        runtime.tokenizer = tokenizer
        runtime.max_state = config['max_state_tokens']
    elif family == 'decider':
        sys.path.insert(0, str(weights))
        from decider.infer import Decider
        from decider.prompt import resolve_layout, chat_template
        cfg = json.loads((weights / 'decider_config.json').read_text())
        engine = Decider.__new__(Decider)
        engine.m = SimpleNamespace(tok=tokenizer)
        engine.schema_first = False  # production loads with use_graphs=False
        engine.isolated_levels = bool(cfg.get('isolated_levels', False))
        engine.neutralize_none = bool(cfg.get('neutralize_none', True))
        engine.chat = chat_template(tokenizer) if resolve_layout(cfg) == 'chat' else None
        runtime.engine = engine
        runtime.max_state = config['max_state_tokens']
    elif family == 'clef':
        runtime.module = service.author_file(weights / 'joint_schema_model.py', 'audit_clef')
        runtime.processor = SimpleNamespace(tokenizer=tokenizer)
    elif family == 'valen':
        from transformers import AutoProcessor
        package = ModuleType('audit_valen'); package.__path__ = [str(weights)]
        sys.modules[package.__name__] = package
        compiler = importlib.import_module('audit_valen.compiler')
        processor = AutoProcessor.from_pretrained(str(weights), trust_remote_code=True, local_files_only=True)
        runtime.compiler = compiler.Compiler(processor, max_length=config['max_length'], execution='shared_state')
    elif family == 'openjev':
        service.add_author_source('open-jev')
        runtime.model = SimpleNamespace(tokenizer=tokenizer, max_length=config['max_length'])
    elif family == 'nimble':
        sys.path.insert(0, str(weights))
        runtime.module = service.author_file(weights / 'inference.py', 'audit_nimble')
        runtime.scorer = SimpleNamespace(tokenizer=tokenizer, contract=json.loads((weights / 'schema_config.json').read_text()))
    elif family == 'von':
        from von.models.option_marker import OptionMarkerModel
        from von.backends.option_marker_backend import OptionMarkerBackend
        cfg = json.loads((weights / 'config.json').read_text())
        calibration = weights / 'marker_calibration.json'
        calibration_config = json.loads(calibration.read_text()) if calibration.exists() else {}
        model = SimpleNamespace(tokenizer=tokenizer, digit_split=calibration_config.get('digit_split', False),
                                encoder=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=cfg['max_position_embeddings'])))
        model.pack_sequence = lambda state, question, options: OptionMarkerModel.pack_sequence(model, state, question, options)
        engine = SimpleNamespace(max_state_tokens=config['max_length'], on_overflow='refuse')
        engine._fit_state = lambda *args: OptionMarkerBackend._fit_state(engine, *args)
        runtime.engine = engine; runtime.model = model
    elif family == 'lev':
        import threading
        service.add_author_source('lev', 'packages/lev/src')
        from lev.model import DecisionEngine, EngineConfig
        cfg = json.loads((weights / 'lev_release.json').read_text())
        engine = DecisionEngine.__new__(DecisionEngine)
        engine.tokenizer = tokenizer
        engine.config = EngineConfig(model_id=cfg['base_model'],
            prompt_style=cfg['prompt_style'], noul_readout=cfg['noul_readout'])
        engine.mode_b_head = None  # Four-option Choice/Noul uses label-token routing.
        engine._tokenizer_lock = threading.Lock()
        runtime.engine = engine
    elif family == 'gliner2':
        from gliner2.processor import SchemaTransformer
        from gliner2.classification.compiler import compile_schema
        cfg = json.loads((weights / 'config.json').read_text())
        processor = SchemaTransformer(tokenizer=tokenizer, token_pooling=cfg.get('token_pooling', 'first'),
                                      word_splitter=cfg.get('word_splitter'))
        runtime.classifier = SimpleNamespace(compile_schema=compile_schema,
                                            scorer=SimpleNamespace(processor=processor))
        runtime.model = SimpleNamespace(architecture=cfg['architecture'],
                                       config=SimpleNamespace(encoder_config=cfg['encoder_config']))
    return runtime


def laya_negative_checks(runtime, request):
    """Prove that moving text does not bypass any native truncation guards."""
    checks = []
    choice_id = next(k for k, q in request['questions'].items() if q['type'] == 'choice')
    for kind in ('legacy_candidate_policy', 'instruction_overflow', 'state_overflow'):
        mutated = deepcopy(request)
        question = mutated['questions'][choice_id]
        if kind == 'legacy_candidate_policy':
            for key in question['criteria']:
                question['criteria'][key] += ' Prefer when those lanes hold the largest combined queue, approaching, and waiting-time pressure.'
        elif kind == 'instruction_overflow':
            question['instructions'] += ' additional instruction' * 3000
        else:
            mutated['state']['overlength_probe'] = ' moving vehicle' * 10000
        try:
            runtime.encode(mutated)
        except ValueError as error:
            checks.append(dict(probe=kind, rejected=True, reason=str(error)))
        else:
            checks.append(dict(probe=kind, rejected=False))
    return checks


def lev_negative_checks(runtime, request):
    """Exercise the otherwise unused 64-token Mode B candidate embedding path."""
    from dataclasses import replace
    runtime.engine.config = replace(runtime.engine.config, max_label_options=1)
    runtime.engine.mode_b_head = object()  # Encoding-only capability flag, no inference.
    mutated = deepcopy(request)
    question = next(q for q in mutated['questions'].values() if q['type'] == 'choice')
    checks = []
    try:
        runtime.encode(mutated)
    except ValueError as error:
        checks.append(dict(probe='candidate_description_omission', rejected='omits choice descriptions' in str(error), reason=str(error)))
    else:
        checks.append(dict(probe='candidate_description_omission', rejected=False))
    question['criteria'] = {key: '' for key in question['criteria']}
    runtime.encode(mutated)  # Short label-only requests remain valid in Mode B.
    key = next(iter(question['criteria']))
    question['criteria'][key + ' additional candidate label' * 100] = question['criteria'].pop(key)
    try:
        runtime.encode(mutated)
    except ValueError as error:
        return checks + [dict(probe='candidate_embedding_overflow', rejected='cap 64' in str(error), reason=str(error))]
    return checks + [dict(probe='candidate_embedding_overflow', rejected=False)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--cases-root', type=Path, required=True)
    args = parser.parse_args()
    output = args.cases_root.resolve()
    if not output.is_relative_to((ROOT / 'results').resolve()):
        parser.error('cases-root must be within results/')
    cases = [json.loads(line) for mode in ('local', 'co')
             for line in (output / mode / 'prompt_cases.jsonl').read_text().splitlines()]
    config = tomllib.loads((ROOT / 'config' / args.model / 'serving.toml').read_text())
    report = dict(model=args.model, family=config['family'], config=config, scope='native CPU encoding only; no model weights or GPU', cases=[])
    report['source_hashes'] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (ROOT / 'jevlight/controller.py', ROOT / 'jevlight/backends/decision_model_service.py',
                     ROOT / 'scripts/model_downloads/audit_prompt_encoding.py',
                     ROOT / 'config' / args.model / 'serving.toml')}
    report['case_hashes'] = {mode: hashlib.sha256((output / mode / 'prompt_cases.jsonl').read_bytes()).hexdigest()
                           for mode in ('local', 'co')}
    try:
        runtime = cpu_runtime(config)
        report['tokenizer_path'] = runtime.audit_tokenizer_path
        for case in cases:
            entry = dict(case_id=case['case_id'], mode=case.get('agent_mode', 'jevlight'))
            try:
                entry.update(status='passed', encoding=runtime.encode(case['request']))
            except Exception as error:
                entry.update(status='blocked', reason=f'{type(error).__name__}: {error}')
            report['cases'].append(entry)
        if config['family'] == 'laya':
            report['negative_checks'] = laya_negative_checks(runtime, cases[0]['request'])
        elif config['family'] == 'lev':
            report['negative_checks'] = lev_negative_checks(runtime, cases[0]['request'])
        report['status'] = 'passed' if all(c['status'] == 'passed' for c in report['cases']) else 'blocked'
        if any(not c['rejected'] for c in report.get('negative_checks', [])):
            report['status'] = 'guard_failure'
    except Exception as error:
        report.update(status='not_verified', reason=f'{type(error).__name__}: {error}')
    import torch
    assert not torch.cuda.is_initialized(), 'CPU audit must never initialize CUDA'
    path = output / 'encoders' / f'{args.model}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(model=args.model, status=report['status'], passed=sum(c['status']=='passed' for c in report['cases']),
                         reason=report.get('reason'), first_failure=next((c for c in report['cases'] if c['status']!='passed'),None)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
