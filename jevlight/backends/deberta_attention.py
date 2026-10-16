"""Bounded-memory DeBERTa inference retaining its trained relative attention."""
import math
from types import MethodType


def _forward(self, hidden_states, attention_mask, output_attentions=False,
             query_states=None, relative_pos=None, rel_embeddings=None):
    import torch
    from torch.nn import functional as F
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from transformers.models.deberta_v2.modeling_deberta_v2 import make_log_bucket_position
    if self.training or output_attentions or query_states is not None:
        if attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :] * attention_mask[:, None, :, None]
        return self._native_forward(hidden_states, attention_mask, output_attentions,
                                 query_states, relative_pos, rel_embeddings)
    batch, length, _ = hidden_states.shape
    heads = self.num_attention_heads
    query = self.transpose_for_scores(self.query_proj(hidden_states), heads)
    key = self.transpose_for_scores(self.key_proj(hidden_states), heads)
    value = self.transpose_for_scores(self.value_proj(hidden_states), heads)
    factor = 1 + int('c2p' in self.pos_att_type) + int('p2c' in self.pos_att_type)
    scale = math.sqrt(query.shape[-1] * factor)
    span = self.pos_ebd_size
    positional_query = positional_key = None
    if self.relative_attention:
        embeddings = self.pos_dropout(rel_embeddings)[:2 * span].unsqueeze(0)
        if self.share_att_key:
            positional_query = self.transpose_for_scores(self.query_proj(embeddings), heads).repeat(batch, 1, 1)
            positional_key = self.transpose_for_scores(self.key_proj(embeddings), heads).repeat(batch, 1, 1)
        else:
            if 'p2c' in self.pos_att_type:
                positional_query = self.transpose_for_scores(self.pos_query_proj(embeddings), heads).repeat(batch, 1, 1)
            if 'c2p' in self.pos_att_type:
                positional_key = self.transpose_for_scores(self.pos_key_proj(embeddings), heads).repeat(batch, 1, 1)
    p2c = torch.bmm(key, positional_query.transpose(-1, -2)) if positional_query is not None else None
    key_positions = torch.arange(length, device=hidden_states.device)
    # Bucket only the O(L) possible distances, rather than evaluating logarithms
    # for every Q x K tile in every layer. All original relative positions remain.
    distances = torch.arange(1-length, length, device=hidden_states.device)
    if self.position_buckets > 0:
        distances = make_log_bucket_position(distances, self.position_buckets, self.max_relative_positions)
    position_indices = (distances.long() + span).clamp(0, 2 * span - 1)
    query_chunk = min(length, max(self._native_query_chunk, 1024**3 // (batch * heads * length * 4)))
    output = torch.empty_like(query)
    for start in range(0, length, query_chunk):
        stop = min(length, start + query_chunk)
        relative = torch.arange(start, stop, device=hidden_states.device)[:, None] - key_positions[None, :]
        indices = position_indices[relative + length - 1]
        bias = torch.zeros((batch * heads, stop-start, length), device=query.device, dtype=query.dtype)
        if positional_key is not None:
            c2p = torch.bmm(query[:, start:stop], positional_key.transpose(-1, -2))
            bias += c2p.gather(-1, indices[None].expand(batch * heads, -1, -1)) / scale
        if p2c is not None:
            # The author's key-row gather is transposed back to query/key order.
            bias += p2c.gather(-1, indices.T[None].expand(batch * heads, -1, -1)).transpose(-1, -2) / scale
        bias = bias.view(batch, heads, stop-start, length)
        if attention_mask.ndim == 2:
            allowed = attention_mask[:, None, start:stop, None].bool() & attention_mask[:, None, None, :].bool()
        else:
            allowed = attention_mask[..., start:stop, :].bool()
        bias.masked_fill_(~allowed, torch.finfo(query.dtype).min)
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            context = F.scaled_dot_product_attention(
                query[:, start:stop].view(batch, heads, stop-start, -1),
                key.view(batch, heads, length, -1), value.view(batch, heads, length, -1),
                attn_mask=bias, dropout_p=0.0, scale=1 / scale)
        output[:, start:stop] = context.reshape(batch * heads, stop-start, -1)
    context = output.view(batch, heads, length, -1).permute(0, 2, 1, 3).contiguous()
    return context.view(batch, length, -1), None


def enable_deberta_attention(model, query_chunk=256):
    """Keep FP32 weights and all key tokens; only tile attention's query axis."""
    installed = 0
    from jevlight.backends.deberta_preprojected_attention import flash_forward
    for module in model.modules():
        if type(module).__name__ == 'DisentangledSelfAttention' and not hasattr(module, '_native_forward'):
            module._native_forward = module.forward
            module._native_query_chunk = query_chunk
            module.forward = MethodType(flash_forward, module)
            installed += 1
        elif type(module).__name__ == 'DebertaV2Encoder' and not hasattr(module, '_deberta_attention_mask'):
            module._deberta_attention_mask = module.get_attention_mask
            module._native_relative_pos = module.get_rel_pos
            # The attention layer constructs exactly the same masks/relative
            # positions per query tile rather than allocating B x L x L first.
            module.get_attention_mask = lambda mask: mask
            module.get_rel_pos = lambda *args, **kwargs: None
    return installed
