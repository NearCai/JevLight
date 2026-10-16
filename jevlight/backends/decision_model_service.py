"""Native decision heads behind the existing local System One transport."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import tomllib

from jevlight.local_server import make_server

ROOT = Path(__file__).resolve().parents[2]


def add_author_source(name, subdirectory=""):
    sources = json.loads((ROOT / "scripts/model_downloads/runtime_sources.json").read_text())
    identity = sources[name]
    source = ROOT / identity["path"]
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True)
    if revision != identity["revision"] or dirty:
        raise ValueError(f"{name}: author runtime revision changed")
    sys.path.insert(0, str(source / subdirectory))


def local_base(config, base_id, revision=None):
    manifest = json.loads((ROOT / config["weights_path"]).parent.joinpath("manifest.json").read_text())
    for base in manifest["bases"]:
        if base["model_id"] == base_id and (not revision or base["revision"].startswith(revision)):
            return base["weights_path"]
    raise ValueError(f"missing pinned base {base_id}@{revision}")


def prepare_native_cache(config):
    """Expose checked local bases through the author's normal HF cache lookup."""
    cache = ROOT / ".local/models/_cache/huggingface/hub"
    os.environ["HF_HOME"] = str(cache.parent)
    os.environ["HF_HUB_CACHE"] = str(cache)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    manifest_path = (ROOT / config["weights_path"]).parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for base in manifest["bases"]:
        source = Path(base["weights_path"])
        base_manifest = json.loads((source / "manifest.json").read_text())
        if base_manifest["status"] != "downloaded":
            raise ValueError("base snapshot incomplete")
        repository = cache / ("models--" + base["model_id"].replace("/", "--"))
        snapshot = repository / "snapshots" / base["revision"]
        for entry in base_manifest["files"]:
            target = snapshot / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            origin = source / entry["path"]
            if target.exists() or target.is_symlink():
                if target.resolve() != origin.resolve():
                    raise ValueError("existing HF cache file points to a different base")
            else:
                target.symlink_to(origin)
        refs = repository / "refs"
        refs.mkdir(exist_ok=True)
        (refs / "main").write_text(base["revision"])


def validate_answers(request, result, *, normalization_abs_tolerance=2.1e-4):
    answers = result.get("answers", {})
    if set(answers) != set(request["questions"]):
        raise ValueError("answer IDs differ from question IDs")
    for question_id, question in request["questions"].items():
        if question["type"] == "noul":
            answer = answers[question_id]
            value = answer.get("noul")
            if answer.get("type") != "noul" or type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("invalid native congestion probability")
            continue
        if question["type"] != "choice":
            raise ValueError("this preparation protocol requires choice questions")
        answer = answers[question_id]
        if answer.get("type") != "choice":
            raise ValueError("answer type differs from choice question")
        probabilities = answer.get("probabilities", {})
        if set(probabilities) != set(question["criteria"]):
            raise ValueError("candidate keys lost or changed")
        values = list(probabilities.values())
        if any(type(p) not in (int, float) or not math.isfinite(p) or p < 0 or p > 1 for p in values):
            raise ValueError("invalid candidate probability")
        # Laya and Clef APIs round each of four probabilities to 4 decimals;
        # retain the native values and allow only that known rounding envelope.
        if not math.isclose(sum(values), 1.0, rel_tol=0, abs_tol=normalization_abs_tolerance):
            raise ValueError("candidate probabilities are not normalized")
        if answer.get("choice") not in probabilities:
            raise ValueError("invalid phase choice")
        if probabilities[answer["choice"]] < max(values) - 1e-6:
            raise ValueError("chosen phase differs from probability argmax")
        confidence = answer.get("confidence")
        if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("invalid decision confidence")
    return result


class LayaNative:
    def __init__(self, config):
        import laya
        self.agent = laya.load(config["weights_path"], device="cuda", backend="eager")
        self.max_len = config["max_length"]
        self.head_max_len = config["head_max_length"]
        self.sort_batch_by_length = bool(config.get('batch_sort_by_length', True))
        supported = self.agent.model.encoder.config.max_position_embeddings
        if self.max_len > supported:
            raise ValueError(f"max_length={self.max_len} exceeds encoder limit={supported}")

    def encode(self, request):
        from laya.common import build_sequence, encode_text, render_options
        reports = []
        for qid, definition in request["questions"].items():
            question = self.agent._to_internal(definition)
            # The author's encoder caps each option at 48 regardless of the
            # configurable head budget. Reject rather than bypass that path.
            options = render_options(question)
            option_lengths = [len(encode_text(self.agent.tok, " " + option,
                                              add_special_tokens=False)["input_ids"]) for option in options]
            if max(option_lengths, default=0) > 48:
                raise ValueError(f"{qid}: native option cap 48 exceeded: {option_lengths}")
            actual = build_sequence(self.agent.tok, request["state"], question,
                                    self.max_len, self.head_max_len, return_stats=True, return_truncation_stats=True)
            unlimited = build_sequence(self.agent.tok, request["state"], question,
                                       1000000, 1000000, return_stats=True, return_truncation_stats=True)
            if actual[0] != unlimited[0] or actual[1] != unlimited[1]:
                raise ValueError(f"{qid}: complete request {len(unlimited[0])} tokens; native limits state={self.max_len}, head={self.head_max_len}; would truncate state, instructions or candidates")
            reports.append({"question_id": qid, "encoded_tokens": len(actual[0]),
                            "option_tokens": option_lengths, "markers": len(actual[1]),
                            "max_length": self.max_len, "head_max_length": self.head_max_len,
                            "encoder_limit": self.agent.model.encoder.config.max_position_embeddings,
                            "truncated": False})
        return reports

    def native(self, request):
        result = self.agent.predict(request["state"], request["questions"],
                                    max_len=self.max_len, head_max_len=self.head_max_len)
        if result.get("usage", {}).get("truncated"):
            raise ValueError("native runtime reported truncation")
        return result

    def native_batch(self, requests, batch_size, *, bucket_by_length=False):
        """Batch identical question definitions; preserve each state's IDs and content."""
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        groups = {}
        for index, request in enumerate(requests):
            definitions = list(request['questions'].values())
            signature = json.dumps(definitions, ensure_ascii=False)
            if bucket_by_length:
                # Exact lengths preserve the single-request padding shape and
                # avoid wasting quadratic attention on shorter states.
                encoded = self.encode(request)
                signature = (signature, max(row['encoded_tokens'] for row in encoded))
            groups.setdefault(signature, []).append((index, request))
        results = [None] * len(requests)
        for group in groups.values():
            canonical = {f'q{i}': value for i, value in
                         enumerate(group[0][1]['questions'].values())}
            outcomes = self.agent.predict_batch(
                [request['state'] for _, request in group], canonical,
                batch_size=batch_size, max_len=self.max_len,
                head_max_len=self.head_max_len,
                sort_by_length=getattr(self, 'sort_batch_by_length', False))
            if len(outcomes) != len(group):
                raise ValueError('native batch result count mismatch')
            for (index, request), outcome in zip(group, outcomes):
                if outcome.get('usage', {}).get('truncated'):
                    raise ValueError('native batch runtime reported truncation')
                outcome = dict(outcome)
                outcome['answers'] = {qid: outcome['answers'][f'q{i}']
                                      for i, qid in enumerate(request['questions'])}
                results[index] = validate_answers(request, outcome)
        return results


class VonNative:
    def __init__(self, config):
        from von.backends.option_marker_backend import OptionMarkerBackend
        self.engine = OptionMarkerBackend(checkpoint_dir=config["weights_path"], device="cuda")
        self.engine.on_overflow = "refuse"
        self.engine.max_state_tokens = config["max_length"]
        self.model = self.engine._get_model()

    def encode(self, request):
        from von.backends.option_marker_backend import _format_state
        from von.types import Choice, Noul
        reports = []
        for qid, definition in request["questions"].items():
            question = (Choice if definition["type"] == "choice" else Noul)(**definition)
            state = _format_state(request["state"])
            descriptions = list(question.criteria.values()) if definition["type"] == "choice" else [
                (question.criteria or {}).get("true") or "Yes, condition holds true.",
                (question.criteria or {}).get("false") or "No, condition is false."]
            packed = self.model.pack_sequence(state, question.instructions, descriptions)
            encoded = self.model.tokenizer(packed)["input_ids"]
            limit = self.model.encoder.config.max_position_embeddings
            if len(encoded) > limit:
                raise ValueError(f"{qid}: {len(encoded)} tokens exceed encoder limit {limit}")
            if self.engine._fit_state(self.model, state, question.instructions, descriptions) != state:
                raise ValueError("native runtime shortened the state")
            reports.append({"question_id": qid, "encoded_tokens": len(encoded), "encoder_limit": limit,
                            "max_state_tokens": self.engine.max_state_tokens, "truncated": False})
        return reports

    def native(self, request):
        result = self.engine.evaluate(**request).model_dump(mode="json")
        if result.get("truncation"):
            raise ValueError("native runtime reported truncation")
        return result


class GlinerNative:
    def __init__(self, config):
        from gliner2.classification import Classifier
        self.classifier = Classifier.from_pretrained(config["weights_path"], local_files_only=True,
                                                    device="cuda").to("cuda").eval()
        self.model = self.classifier.model
        self.max_length = config["max_length"]

    def rows(self, request):
        from gliner2.classification import ClassificationSchema
        for qid, definition in request["questions"].items():
            # Native typed schemas forbid parentheses in instructions. Retain
            # the entire original instruction in the text, alongside the full
            # serialized state; labels and descriptions stay in the schema.
            text = json.dumps(request["state"], ensure_ascii=False) + "\n" + definition["instructions"]
            criteria = definition["criteria"] if definition["type"] == "choice" else {"true": "True", "false": "False"}
            schema = ClassificationSchema().single(qid, criteria)
            yield qid, text, schema

    def encode(self, request):
        reports = []
        for qid, text, schema in self.rows(request):
            compiled = self.classifier.compile_schema(schema)
            processor = self.classifier.scorer.processor
            batch = processor.collate_fn_inference([(text, compiled.build())], max_len=None,
                                                   architecture=self.model.architecture)
            tokens = int(batch.attention_mask.sum().item())
            if tokens > self.max_length:
                raise ValueError(f"complete request {tokens} tokens exceeds serving limit {self.max_length}; no truncation")
            reports.append({"question_id": qid, "encoded_tokens": tokens,
                            "native_max_len": None, "encoder_position_config": self.model.config.encoder_config.get("max_position_embeddings")
                            if isinstance(self.model.config.encoder_config, dict) else None, "truncated": False})
        return reports

    def native(self, request):
        from gliner2.classification import ClassificationConfig
        answers = {}
        for qid, text, schema in self.rows(request):
            result = self.classifier.classify(text, schema, config=ClassificationConfig(max_len=None))
            if request["questions"][qid]["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": result.probabilities(qid)["true"]}
                continue
            answers[qid] = {"type": "choice", "choice": result.value(qid),
                            "probabilities": dict(result.probabilities(qid)), "confidence": result.confidence(qid)}
        return {"answers": answers, "usage": {"output_tokens": 0}}


class KevNative:
    def __init__(self, config):
        import torch
        add_author_source("kev")
        from kev.checkpoint import Checkpoint, LoadOptions
        checkpoint = Checkpoint(str(ROOT / config["weights_path"]))
        # Full checkpoints carry both their own backbone and the author's
        # tokenizer copy. Only adapters require the separately pinned base.
        base = checkpoint.path if checkpoint.full else local_base(
            config, checkpoint.meta.base, checkpoint.meta.base_revision)
        checkpoint.meta.base = base
        checkpoint.meta.base_revision = None
        self.tokenizer, self.model = checkpoint.load("cuda", LoadOptions(dtype=torch.bfloat16, merge=True,
                                                                         backend="torch", fused=False, cuda_graphs=False))
        self.max_length = config["max_length"]
        self.max_state = config["max_state_tokens"]

    def record(self, request):
        from kev.api import SystemOneRequest, to_record
        return to_record(SystemOneRequest(**request))

    def encode_native(self, request):
        record, meta = self.record(request)
        encoded = self.model.encode(self.tokenizer, record, max_state=self.max_state,
                                    max_branch=self.max_length, strict=True)
        if encoded["state_truncated"]:
            raise ValueError("native state truncated")
        if any(len(options) != (len(request["questions"][entry["id"]]["criteria"]) if request["questions"][entry["id"]]["type"] == "choice" else 2)
               for options, entry in zip(encoded["opt_idx"], meta)):
            raise ValueError("native option markers lost")
        return encoded, meta

    def encode(self, request):
        encoded, _ = self.encode_native(request)
        return [{"encoded_tokens": len(encoded["ids"]), "state_tokens": encoded["state_tokens"],
                 "max_state_tokens": self.max_state, "max_length": self.max_length, "truncated": False}]

    def native(self, request):
        import torch
        from kev.api import to_answers
        encoded, meta = self.encode_native(request)
        with torch.inference_mode():
            probabilities = self.model.probs(encoded)
        return {"answers": to_answers(probabilities, meta), "usage": {"input_tokens": len(encoded["ids"]), "output_tokens": 0}}

    def native_batch(self, requests, batch_size):
        import torch
        from kev.api import to_answers
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch_size must be positive')
        results = []
        with torch.inference_mode():
            for start in range(0, len(requests), batch_size):
                group = requests[start:start + batch_size]
                packed = [self.encode_native(request) for request in group]
                logits = self.model.forward_batch([encoded for encoded, meta in packed])
                if len(logits) != len(group): raise ValueError('native batch result count mismatch')
                for (encoded, meta), row in zip(packed, logits, strict=True):
                    probabilities = [torch.nn.functional.softmax(z, -1).cpu() for z in row]
                    results.append({'answers': to_answers(probabilities, meta),
                                    'usage': {'input_tokens': len(encoded['ids']), 'output_tokens': 0}})
        return [validate_answers(request, result) for request, result in zip(requests, results, strict=True)]



class AgentNative:
    def __init__(self, config):
        import torch
        from safetensors.torch import load_file
        add_author_source("agent-jev")
        from jev_service.engine import DecisionEngine
        weights = ROOT / config["weights_path"]
        checkpoint = weights.parent / "adapter/agentjev.pt"
        checkpoint.parent.mkdir(exist_ok=True)
        if not checkpoint.exists():
            torch.save({"state_dict": load_file(str(weights / "model.safetensors"))}, checkpoint)
        self.engine = DecisionEngine(str(checkpoint), local_base(config, "Qwen/Qwen3-0.6B"),
                                     max_tokens=config["max_length"], path_batch=1, encoder="path",
                                     temperatures=str(weights / "temperatures.json"))
        self.model = self.engine.model

    def payload(self, request):
        return {"state": request["state"], "questions": [
            dict({"id": qid, "type": "boolean" if definition["type"] == "noul" else "choice", "question": definition["instructions"]},
                 **({"options": definition["criteria"]} if definition["type"] == "choice" else {}))
            for qid, definition in request["questions"].items()]}

    def encode(self, request):
        from jev_service.contract import prepare, encode_paths
        paths, locations, _ = encode_paths(prepare(self.payload(request)), self.engine.tokenizer, self.engine.max_tokens)
        return [{"candidate_path_tokens": [len(path) for path in paths], "candidate_path_count": len(locations),
                 "max_length": self.engine.max_tokens, "truncated": False}]

    def native(self, request):
        native = self.engine.evaluate(self.payload(request))
        if native["usage"].get("truncated_inputs"):
            raise ValueError("native input truncated")
        answers = {answer["id"]: ({"type": "noul", "noul": answer["probability"]} if answer["type"] == "boolean" else {"type": "choice", "choice": answer["value"],
            "probabilities": answer["distribution"], "confidence": answer["top_probability"]}
            )
            for answer in native["results"][0]["answers"]}
        return {"answers": answers, "usage": native["usage"]}

    def native_batch(self, requests, batch_size):
        if type(batch_size) is not int or not 1 <= batch_size <= 32:
            raise ValueError('Author request batch requires 1..32 states')
        trained = self.engine.trained_config
        previous = trained.get('eval_batch_states') if trained else None
        previous_path_batch = self.engine.path_batch
        results = []
        try:
            if trained: trained['eval_batch_states'] = batch_size
            else: self.engine.path_batch = max(previous_path_batch, batch_size)
            for start in range(0, len(requests), batch_size):
                group = requests[start:start + batch_size]
                native = self.engine.evaluate({'requests': [self.payload(request) for request in group]})
                if native['usage'].get('truncated_inputs'): raise ValueError('native batch input truncated')
                if len(native['results']) != len(group): raise ValueError('native batch result count mismatch')
                for row in native['results']:
                    answers = {a['id']: ({'type': 'noul', 'noul': a['probability']} if a['type'] == 'boolean' else
                        {'type': 'choice', 'choice': a['value'], 'probabilities': a['distribution'], 'confidence': a['top_probability']})
                        for a in row['answers']}
                    results.append({'answers': answers, 'usage': {}})
        finally:
            self.engine.path_batch = previous_path_batch
            if trained:
                if previous is None: trained.pop('eval_batch_states', None)
                else: trained['eval_batch_states'] = previous
        return [validate_answers(request, result) for request, result in zip(requests, results, strict=True)]



class DeciderNative:
    def __init__(self, config):
        import torch
        sys.path.insert(0, str(ROOT / config["weights_path"]))
        from decider.infer import Decider
        self.engine = Decider(str(ROOT / config["weights_path"]), device="cuda", dtype=torch.bfloat16, use_graphs=False)
        self.model = self.engine.m
        self.max_length = config["max_length"]
        self.max_state = config["max_state_tokens"]

    def encode(self, request):
        rendered, _, actual = self.engine._system_one_items(**request, max_state_tokens=self.max_state)
        _, _, full = self.engine._system_one_items(**request, max_state_tokens=1000000)
        reports = []
        for row, unlimited, question in zip(actual, full, rendered.values(), strict=True):
            if row["ids"] != unlimited["ids"]:
                raise ValueError("native runtime would truncate state")
            if len(row["ids"]) > self.max_length:
                raise ValueError("complete native request exceeds serving limit")
            if row["nopts"] != [len(question["options"])]:
                raise ValueError("native candidate set changed")
            reports.append({"encoded_tokens": len(row["ids"]), "max_length": self.max_length,
                            "max_state_tokens": self.max_state, "truncated": False})
        return reports

    def native(self, request):
        return self.engine.system_one(**request, max_state_tokens=self.max_state)

    def native_batch(self, requests, batch_size):
        import torch
        from decider.systemone import assemble, unique_tokens
        from decider import temperature as temperatures
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch_size must be positive')
        if self.engine.eng is not None:
            raise ValueError('This adapter batches the native eager state-first path only')
        results = []
        with torch.inference_mode():
            for start in range(0, len(requests), batch_size):
                group = requests[start:start + batch_size]
                layout = 'schema_first' if self.engine.schema_first else 'state_first'
                prepared = [self.engine._system_one_items(**request, max_state_tokens=self.max_state,
                    layout=layout, isolated=self.engine.isolated_levels) for request in group]
                items = [item for _, _, local in prepared for item in local]
                calibration = temperatures.for_items(self.engine.T, self.engine.T_by_type, items)
                probabilities = self.engine._system_one_probs(items, layout, temperature=calibration)
                offset = 0
                for rendered, index, local in prepared:
                    count = sum(len(item['slots']) for item in local)
                    results.append({'model': self.engine.name,
                        'answers': assemble(rendered, index, probabilities[offset:offset + count]),
                        'usage': {'input_tokens': unique_tokens(local), 'output_tokens': 0}})
                    offset += count
                if offset != len(probabilities): raise ValueError('native batch slot mapping changed')
        return [validate_answers(request, result) for request, result in zip(requests, results, strict=True)]



def author_file(path, name):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


class ClefNative:
    def __init__(self, config):
        weights = ROOT / config["weights_path"]
        self.module = author_file(weights / "joint_schema_model.py", "jevlight_author_clef")
        self.model, self.processor = self.module.load_release_model(weights, device="cuda")
        self.max_length = config["max_length"]
        self.model_id = config["model_id"]

    def encode(self, request):
        encoded = self.module.encode_record(self.processor.tokenizer, request, max_length=self.max_length, processor=self.processor)
        full = self.module.encode_record(self.processor.tokenizer, request, max_length=1000000, processor=self.processor)
        if encoded != full:
            raise ValueError("native encoder would truncate request")
        for question in encoded.questions:
            definition = request["questions"][question.question_id]
            expected = len(definition["criteria"]) if definition["type"] == "choice" else 2
            if len(question.option_ids) != expected:
                raise ValueError("candidate IDs lost")
        return [{"encoded_tokens": len(encoded.input_ids), "max_length": self.max_length, "truncated": False}]

    def native(self, request):
        import torch
        with torch.inference_mode():
            return self.module.systemone(self.model, self.processor, dict(request, model=self.model_id), max_length=self.max_length)

    def native_batch(self, requests, batch_size):
        import torch
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch_size must be positive')
        results = []
        with torch.inference_mode():
            device = next(self.model.parameters()).device
            for start in range(0, len(requests), batch_size):
                group = requests[start:start + batch_size]
                encoded = [self.module.encode_record(self.processor.tokenizer, dict(request, model=self.model_id),
                    max_length=self.max_length, processor=self.processor) for request in group]
                logits = self.model(self.module.collate_records(encoded, self.processor.tokenizer.pad_token_id, device))
                if len(logits) != len(encoded): raise ValueError('native batch result count mismatch')
                for request, record, row in zip(group, encoded, logits, strict=True):
                    if len(record.questions) != len(row): raise ValueError('native batch lost decision questions')
                    results.append({'model': self.model_id, 'answers': {
                        question.question_id: self.module.systemone_answer(request['questions'][question.question_id],
                            dict(zip(question.option_ids, z.float().softmax(-1).tolist(), strict=True)))
                        for question, z in zip(record.questions, row, strict=True)},
                        'usage': {'input_tokens': len(record.input_ids), 'output_tokens': 0}})
        return [validate_answers(request, result) for request, result in zip(requests, results, strict=True)]



class ValenNative:
    def __init__(self, config):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(str(ROOT / config["weights_path"]), trust_remote_code=True,
                                              dtype="auto", attn_implementation="sdpa", local_files_only=True).to("cuda").eval()
        if config["max_length"] > self.model.config.valen_max_length:
            raise ValueError("serving limit exceeds published native compiler limit")
        self.compiler = self.model.make_compiler(execution="shared_state")
        self.compiler.max_length = config["max_length"]

    def request(self, request):
        return {"state": json.dumps(request["state"], ensure_ascii=False), "questions": request["questions"]}

    def encode(self, request):
        compiled = self.compiler.compile({"request": self.request(request)})
        for question in compiled.questions:
            definition = request["questions"][question.qid]
            if definition["type"] == "choice" and list(question.keys) != list(definition["criteria"]):
                raise ValueError("native candidate mapping changed")
        return [{"logical_tokens": compiled.logical_tokens, "compute_tokens": compiled.compute_tokens,
                 "max_length": self.compiler.max_length, "truncated": False}]

    def native(self, request):
        return self.model.predict(self.request(request), execution="shared_state")

    def native_batch(self, requests, batch_size):
        """Use the author's shared-state batch and trained readouts."""
        import importlib
        import torch
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch_size must be positive')
        answer = importlib.import_module(type(self.model).__module__.rsplit('.', 1)[0] + '.responses').answer
        results = []
        with torch.inference_mode():
            for start in range(0, len(requests), batch_size):
                group = [self.compiler.compile({'request': self.request(request)}) for request in requests[start:start + batch_size]]
                rows = self.model.forward_state_batch(group, max_tokens=32768)
                if len(rows) != len(group):
                    raise ValueError('native batch result count mismatch')
                for state, logits in zip(group, rows, strict=True):
                    if len(logits) != len(state.questions):
                        raise ValueError('native batch lost decision questions')
                    results.append({'model': self.model.model_name,
                        'answers': {q.qid: answer(q, z, 1.0) for q, z in zip(state.questions, logits, strict=True)},
                        'usage': {'input_tokens': state.logical_tokens, 'output_tokens': 0},
                        'internal_usage': {'compute_tokens': state.compute_tokens}})
        return [validate_answers(request, result) for request, result in zip(requests, results, strict=True)]


class OpenJevNative:
    def __init__(self, config):
        import torch
        from peft import PeftModel
        add_author_source("open-jev")
        from jev.model import DecisionModel
        weights = ROOT / config["weights_path"] / "package/checkpoint"
        metadata = json.loads((weights / "model.json").read_text())
        # Same constructor and readout as DecisionModel.load, with the pinned
        # base resolved locally. The checkpoint files remain byte-identical.
        self.model = DecisionModel(local_base(config, metadata["model_id"], metadata["revision"]), None,
                                   device="cuda:0", lora_rank=0, max_length=config["max_length"])
        self.model.backbone = PeftModel.from_pretrained(self.model.backbone, weights / "adapter")
        self.model.lora_rank = metadata["lora_rank"]
        self.model.head.load_state_dict(torch.load(weights / "head.pt", map_location="cuda:0", weights_only=True))
        self.model._validate_loaded_parameters()
        self.model.eval()
        self.temperature = json.loads((weights / "temperature.json").read_text())["temperature"]

        if config.get('inference_engine', 'author') == 'author_prefix':
            from jevlight.backends.open_jev_prefix_batch import enable_open_prefix_batch
            enable_open_prefix_batch(self, merged=config.get('batch_prefix_merged', False),
                fused=config.get('batch_prefix_fused', False),
                package=ROOT/config.get('batch_fla_package_path', '.local/environments/open-jev-fla/batch_packages'))

    def encode(self, request):
        from jev.api import compile_request, candidate_prompts
        lengths = []
        for record in compile_request(**request):
            for prompt in candidate_prompts(record):
                rendered = self.model.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
                # Match the author's forward exactly. Transformers 5 may return
                # a BatchEncoding for tokenize=True; len(mapping) is not tokens.
                ids = self.model.tokenizer(rendered, truncation=False)["input_ids"]
                if len(ids) > self.model.max_length:
                    raise ValueError(f"complete native candidate path {len(ids)} tokens exceeds limit {self.model.max_length}")
                lengths.append(len(ids))
        return [{"candidate_path_tokens": lengths, "max_length": self.model.max_length, "truncated": False}]

    def native(self, request):
        import torch
        from jev.api import compile_request, format_response
        from jev.metrics import softmax
        records = compile_request(**request)
        with torch.inference_mode():
            logits = self.model(records)
        return format_response(records, [softmax(row.float().cpu().tolist(), temperature=self.temperature) for row in logits])

    def native_batch(self, requests, batch_size):
        import torch
        from jev.api import compile_request, format_response
        from jev.metrics import softmax
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch_size must be positive')
        results = []
        with torch.inference_mode():
            for start in range(0, len(requests), batch_size):
                groups = [compile_request(**request) for request in requests[start:start + batch_size]]
                records = [record for group in groups for record in group]
                logits = self.model(records)
                if len(logits) != len(records): raise ValueError('native batch record mapping changed')
                probabilities = [softmax(row.float().cpu().tolist(), temperature=self.temperature) for row in logits]
                offset = 0
                for group in groups:
                    results.append(format_response(group, probabilities[offset:offset + len(group)]))
                    offset += len(group)
        return [validate_answers(request, result) for request, result in zip(requests, results, strict=True)]



class NimbleNative:
    def __init__(self, config):
        weights = ROOT / config["weights_path"]
        sys.path.insert(0, str(weights))
        self.module = author_file(weights / "inference.py", "jevlight_author_nimble")
        self.scorer = self.module.NimbleModel(str(weights))
        self.model = self.scorer.model

    def schema(self, request):
        return {qid: ({"type": "boolean", "description": definition["instructions"]} if definition["type"] == "noul" else {"type": "enum", "description": definition["instructions"],
                      "choices": list(definition["criteria"]), "choice_descriptions": definition["criteria"]}
                    )
                for qid, definition in request["questions"].items()}

    def encode(self, request):
        prepared = self.module.prepare_prompts(self.scorer.tokenizer,
            json.dumps(request["state"], ensure_ascii=False), self.schema(request), self.scorer.contract["max_length"])
        if any(len(keys) != (4 if definition["type"] == "choice" else 2)
               for keys, definition in zip(prepared.candidate_ids, request["questions"].values())):
            raise ValueError("candidate codebook lost a phase")
        return [{"candidate_path_tokens": [len(ids) for ids in prepared.full_ids],
                 "max_length": self.scorer.contract["max_length"], "truncated": False}]

    def native(self, request):
        result = self.scorer.score(json.dumps(request["state"], ensure_ascii=False), self.schema(request))
        return {"answers": {qid: ({"type": "noul", "noul": field["probability_true"]} if request["questions"][qid]["type"] == "noul" else {"type": "choice", "choice": field["prediction"],
                                 "probabilities": field["probabilities"], "confidence": max(field["probabilities"].values())}
                                 )
                            for qid, field in result["fields"].items()}, "usage": {"output_tokens": 0}}


class LevNative:
    def __init__(self, config):
        add_author_source("lev", "packages/lev/src")
        from lev.model import load
        self.engine = load(str(ROOT / config["weights_path"]), compile=False)
        self.model = self.engine.model
        self.max_length = config["max_length"]

    def encode(self, request):
        from lev.model import candidate_texts
        from lev.router import Mode
        prepared = self.engine.prepare(**request)
        # The author's Mode B embeds each candidate with max_length=64.
        # This is separate from the full prompt window, just like Laya's 48.
        for qid, route in prepared.routes.items():
            if route.mode is Mode.CANDIDATE_PATH:
                definition = request["questions"][qid]
                if definition["type"] == "choice" and any(
                    value is not None and value != "" for value in definition["criteria"].values()
                ):
                    raise ValueError(f"{qid}: native candidate path omits choice descriptions; incomplete input")
                sizes = [len(self.engine.tokenizer(text, add_special_tokens=False)["input_ids"])
                         for text in candidate_texts(prepared.questions[qid])]
                if max(sizes, default=0) > 64:
                    raise ValueError(f"{qid}: native candidate embedding cap 64 exceeded: {sizes}")
        lengths = [len(prepared.prefix_ids) + len(variant.suffix_ids) for variant in prepared.variants]
        if max(lengths) > self.max_length:
            raise ValueError("complete native request exceeds serving limit")
        return [{"candidate_path_tokens": lengths, "max_length": self.max_length, "truncated": False}]

    def native(self, request):
        return self.engine.system_one(**request).model_dump(mode="json")

    def native_batch(self, requests, batch_size):
        """Author calibrated DecisionEngine.answer performs the shared GPU forward."""
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch_size must be positive')
        prepared = [self.engine.prepare(**request) for request in requests]
        results = []
        for start in range(0, len(prepared), batch_size):
            results.extend(result.model_dump(mode='json') for result in
                           self.engine.answer(prepared[start:start + batch_size]))
        if len(results) != len(requests):
            raise ValueError('native batch result count mismatch')
        return [validate_answers(request, result) for request, result in zip(requests, results, strict=True)]



class NativeBackend:
    def __init__(self, config):
        self.config = config
        self.lock = threading.Lock()
        self.reported_model = config["model_id"]
        prepare_native_cache(config)
        if config['family'] == 'open-jev':
            if config.get('inference_engine', 'author') == 'vllm_pooling':
                from jevlight.backends.vllm_decision_runtime import OpenJevPoolingNative
                self.runtime = OpenJevPoolingNative(config)
            else:
                self.runtime = {"laya": LayaNative, "von": VonNative, "gliner2": GlinerNative,
                        "kev": KevNative, "agent-jev": AgentNative, "decider": DeciderNative,
                        "clef": ClefNative, "valen": ValenNative, "open-jev": OpenJevNative,
                        "nimble": NimbleNative, "lev": LevNative}[config["family"]](config)
        else:
            if config.get('inference_engine', 'author') == 'vllm_marker':
                from jevlight.backends.marker_vllm_runtime import NativeMarkerPooling
                self.runtime = NativeMarkerPooling(config)
            else:
                self.runtime = {"laya": LayaNative, "von": VonNative, "gliner2": GlinerNative,
                        "kev": KevNative, "agent-jev": AgentNative, "decider": DeciderNative,
                        "clef": ClefNative, "valen": ValenNative, "open-jev": OpenJevNative,
                        "nimble": NimbleNative, "lev": LevNative}[config["family"]](config)
        self.log_path = None

    def evaluate_batch(self, requests, batch_size):
        """Independent complete requests, one lock and native batch invocation."""
        if not hasattr(self.runtime, 'native_batch'):
            raise ValueError('this native runtime has no verified batch interface')
        maximum = int(self.config.get('batch_size', 16))
        if not 1 <= batch_size <= maximum:
            raise ValueError(f'batch_size must be between 1 and {maximum}')
        with self.lock:
            started = time.monotonic()
            encodings = [self.runtime.encode(request) for request in requests]
            results = self.runtime.native_batch(requests, batch_size)
            if len(results) != len(requests):
                raise ValueError('batch result count mismatch')
            for request, result in zip(requests, results):
                getattr(self, 'answer_validator', validate_answers)(request, result)
            elapsed_ms = (time.monotonic() - started) * 1000
            if self.log_path:
                with self.log_path.open('a') as stream:
                    for request, encoding, result in zip(requests, encodings, results):
                        stream.write(json.dumps(dict(request=request, encoding=encoding,
                            result=result, batch_requests=len(requests), batch_size=batch_size,
                            batch_wall_time_ms=elapsed_ms)) + '\n')
            return results

    def evaluate(self, state, questions):
        # Serialize HTTP concurrency; one GPU model is resident at a time.
        with self.lock:
            request = {"state": state, "questions": questions}
            started = time.monotonic()
            try:
                lengths = self.runtime.encode(request)
                result = getattr(self, 'answer_validator', validate_answers)(request, self.runtime.native(request))
            except Exception as error:
                if self.log_path:
                    with self.log_path.open("a") as stream:
                        stream.write(json.dumps({"request": request, "error": f"{type(error).__name__}: {error}"}) + "\n")
                raise
            if self.log_path:
                with self.log_path.open("a") as stream:
                    stream.write(json.dumps({"request": request, "encoding": lengths, "result": result,
                                             "wall_time_ms": (time.monotonic() - started) * 1000}) + "\n")
            return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--request-log", type=Path, required=True)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    root = Path(__file__).resolve().parents[2]
    log_path = args.request_log.resolve()
    if not log_path.is_relative_to(root / "results"):
        parser.error("request log must be inside results/")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    backend = NativeBackend(config)
    backend.log_path = log_path
    server = make_server(backend, config.get("host", "127.0.0.1"), config.get("port", 8123), config["model_id"])
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
