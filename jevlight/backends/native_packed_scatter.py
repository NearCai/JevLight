"""Copy independent native projection outputs into one padded GPU tensor."""
import torch
import triton
import triton.language as tl


@triton.jit
def _scatter(Flat, Lengths, Offsets, Padded, T, C, N, BLOCK: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = index < N
    row = index // (T * C)
    position = (index // C) % T
    channel = index % C
    length = tl.load(Lengths + row, valid, 0)
    offset = tl.load(Offsets + row, valid, 0)
    value = tl.load(Flat + (offset + position) * C + channel,
                    valid & (position < length), 0)
    tl.store(Padded + index, value, valid)


def scatter_native_rows(flat, layout, width, context):
    """One direct copy/zero-fill; every native value keeps its original bits."""
    layout = tuple(layout)
    if (flat.ndim != 2 or not flat.is_cuda or not flat.is_contiguous()
            or width != max(length for _, length in layout)
            or flat.shape[0] != sum(rows * length for rows, length in layout)):
        raise ValueError('Packed scatter requires the complete contiguous native row layout')
    key = (layout, flat.device)
    if getattr(context, 'scatter_key', None) != key:
        lengths, offsets, cursor = [], [], 0
        for rows, length in layout:
            for row in range(rows):
                lengths.append(length)
                offsets.append(cursor + row * length)
            cursor += rows * length
        context.scatter_lengths = torch.tensor(lengths, dtype=torch.long, device=flat.device)
        context.scatter_offsets = torch.tensor(offsets, dtype=torch.long, device=flat.device)
        context.scatter_key = key
    padded = flat.new_empty((sum(rows for rows, _ in layout), width, flat.shape[1]))
    _scatter[(triton.cdiv(padded.numel(), 256),)](
        flat, context.scatter_lengths, context.scatter_offsets, padded,
        width, flat.shape[1], padded.numel(), BLOCK=256)
    return padded
