"""vLLM pooling adapter; frozen author heads stay external."""
from vllm.model_executor.models.adapters import as_embedding_model
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration
from vllm.model_executor.models.interfaces import IsHybrid, SupportsMRoPE
from vllm.model_executor.models.utils import WeightsMapper


class OpenJevEngineForEmbedding(as_embedding_model(Qwen3_5ForCausalLM), IsHybrid, SupportsMRoPE):
    # The text-only CausalLM class omits the hybrid cache metadata exposed by
    # the multimodal wrapper. Its text backbone needs those same state sizes.
    get_mamba_state_dtype_from_config = classmethod(
        Qwen3_5ForConditionalGeneration.get_mamba_state_dtype_from_config.__func__)
    get_mamba_state_shape_from_config = classmethod(
        Qwen3_5ForConditionalGeneration.get_mamba_state_shape_from_config.__func__)
    get_mamba_state_copy_func = classmethod(
        Qwen3_5ForConditionalGeneration.get_mamba_state_copy_func.__func__)

    def get_mrope_input_positions(self, input_tokens, mm_features):
        import torch
        if mm_features:
            raise ValueError('The frozen decision head accepts complete text-only candidate paths')
        # Text-only positions use the same token index in all three planes.
        return torch.arange(len(input_tokens), dtype=torch.long).expand(3, -1), 0
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={
        'model.language_model.': 'model.',
        'language_model.': 'model.',
        'layers.': 'model.layers.',
    })

    def load_weights(self, weights):
        # Text-only author backbone. Frozen visual/vocabulary heads are not
        # decision weights and the pooling engine never generates text.
        mapped = ((self.hf_to_vllm_mapper._map_name(name), value)
                  for name, value in weights
                  if name.startswith('model.language_model.'))
        return super().load_weights(mapped)
