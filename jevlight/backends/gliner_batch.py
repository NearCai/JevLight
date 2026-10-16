"""Author-native GLiNER junction batching, enabled at a model boundary."""
from types import MethodType


def native_batch(self, requests, batch_size):
    from gliner2.classification import ClassificationConfig
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    outputs = [{'answers': {}, 'usage': {'output_tokens': 0}} for _ in requests]
    rows = [(index, qid, text, schema) for index, request in enumerate(requests)
            for qid, text, schema in self.rows(request)]
    if not rows:
        return outputs
    # Each text keeps its own author schema, including label order. Question IDs
    # may repeat across junctions; restore by request index rather than ID alone.
    # A whole request group includes every optional question row in one forward.
    row_batch_size = len(rows)
    config = ClassificationConfig(max_len=None, batch_size=row_batch_size)
    compiled = [self.classifier.compile_schema(row[3]) for row in rows]
    # The pinned 2.0.0 high-level batch_classify accepts only one shared schema.
    # Its native scorer supports per-row compiled schemas; preserve that exact
    # scorer and the author's decoder instead of changing the installed runtime.
    scores = self.classifier.scorer.batch_score(
        [row[2] for row in rows], compiled, max_len=None, batch_size=row_batch_size)
    if len(scores) != len(rows):
        raise ValueError('GLiNER batch result count mismatch')
    results = [self.classifier.decode(score, schema, config=config)
               for score, schema in zip(scores, compiled, strict=True)]
    if len(results) != len(rows):
        raise ValueError('GLiNER batch result count mismatch')
    for (index, qid, _, _), result in zip(rows, results, strict=True):
        if requests[index]['questions'][qid]['type'] == 'noul':
            answer = {'type': 'noul', 'noul': result.probabilities(qid)['true']}
        else:
            answer = {'type': 'choice', 'choice': result.value(qid),
                      'probabilities': dict(result.probabilities(qid)),
                      'confidence': result.confidence(qid)}
        outputs[index]['answers'][qid] = answer
    return outputs


def enable_gliner_batch(runtime):
    """Keep the shared loaded service unchanged while adding its native API."""
    from jevlight.backends.decision_model_service import GlinerNative
    if isinstance(runtime, GlinerNative):
        runtime.native_batch = MethodType(native_batch, runtime)
