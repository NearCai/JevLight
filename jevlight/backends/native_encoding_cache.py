"""Reuse validated CPU encodings only inside one independent GPU batch call."""
import json
import threading
from collections import OrderedDict
from types import MethodType


def enable_native_encoding_cache(runtime, family):
    """Keep scalar encoding unchanged and release every batch's prepared inputs."""
    if hasattr(runtime, '_step_encoding_context'):
        return
    if family not in ('lev', 'clef'):
        raise ValueError('Native encoding reuse supports Lev and Clef text requests')
    context = threading.local()
    runtime._step_encoding_context = context

    def entries():
        if not hasattr(context, 'entries'):
            context.entries = OrderedDict()
        return context.entries

    def key(value):
        try:
            # Preserve question and candidate order, every state field and ID.
            return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
        except (TypeError, ValueError):
            return None

    def remember(identity, result):
        if identity is None:
            return
        cache = entries()
        cache[identity] = result
        cache.move_to_end(identity)
        while len(cache) > 512:
            cache.popitem(last=False)

    if family == 'lev':
        engine = runtime.engine
        original_prepare = engine.prepare

        def prepare(self, state, questions):
            identity = key(dict(state=state, questions=questions))
            if getattr(context, 'active', False) and identity in entries():
                return entries()[identity]
            result = original_prepare(state, questions)
            if getattr(context, 'active', False) or getattr(context, 'capture', False):
                remember(identity, result)
            return result

        engine.prepare = MethodType(prepare, engine)
    else:
        module = runtime.module
        original_encode = module.encode_record

        def encode_record(tokenizer, record, max_length=16384, max_state_tokens=None, processor=None):
            # Media encodings hold mutable tensors; the traffic adapter is text.
            identity = None
            if not record.get('images') and not record.get('videos'):
                # The author's encoder never reads `model`; it is response
                # metadata added by native_batch after complete-input validation.
                identity = key(dict(record={k: v for k, v in record.items() if k != 'model'},
                                    maximum=max_length, state_maximum=max_state_tokens,
                                    tokenizer=id(tokenizer), processor=id(processor)))
            if getattr(context, 'active', False) and identity in entries():
                return entries()[identity]
            result = original_encode(tokenizer, record, max_length=max_length,
                                     max_state_tokens=max_state_tokens, processor=processor)
            if getattr(context, 'active', False) or getattr(context, 'capture', False):
                remember(identity, result)
            return result

        module.encode_record = encode_record

    original_encoding = runtime.encode
    original_native = runtime.native
    original_batch = runtime.native_batch

    def encoding(self, request):
        previous = getattr(context, 'capture', False)
        context.capture = True
        try:
            return original_encoding(request)
        except BaseException:
            entries().clear()
            raise
        finally:
            context.capture = previous

    def native(self, request):
        previous = getattr(context, 'active', False)
        context.active = False
        try:
            return original_native(request)
        finally:
            context.active = previous
            entries().clear()

    def call(function, *args):
        previous = getattr(context, 'active', False)
        context.active = True
        try:
            return function(*args)
        finally:
            context.active = previous
            entries().clear()

    def batch(self, requests, batch_size):
        return call(original_batch, requests, batch_size)

    runtime.encode = MethodType(encoding, runtime)
    runtime.native = MethodType(native, runtime)
    runtime.native_batch = MethodType(batch, runtime)
    if hasattr(runtime, 'probe_step_capacity'):
        original_probe = runtime.probe_step_capacity

        def probe(self, requests, requested):
            return call(original_probe, requests, requested)

        runtime.probe_step_capacity = MethodType(probe, runtime)
