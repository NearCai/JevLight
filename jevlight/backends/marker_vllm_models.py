"""Text backbone pooling for frozen native marker decision heads."""
from jevlight.backends.vllm_decision_models import OpenJevEngineForEmbedding


class NativeMarkerEngine(OpenJevEngineForEmbedding):
    def load_weights(self, weights):
        def mapped():
            for name, value in weights:
                if name.startswith('model.language_model.'):
                    name = 'model.' + name.removeprefix('model.language_model.')
                elif name.startswith(('layers.', 'embed_tokens.', 'norm.')):
                    name = 'model.' + name
                if name.startswith(('model.layers.', 'model.embed_tokens.', 'model.norm.')):
                    yield name, value
        return super(OpenJevEngineForEmbedding, self).load_weights(mapped())


class ValenMarkerEngine(NativeMarkerEngine):
    def load_weights(self,weights):
        mapped=(('model.'+name.removeprefix('backbone.language_model.'),value)
                for name,value in weights if name.startswith('backbone.language_model.'))
        return super().load_weights(mapped)
