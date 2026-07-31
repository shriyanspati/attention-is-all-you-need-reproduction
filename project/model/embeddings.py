"""Embedding layer construction and three-way weight tying (paper Section 3.4).

"we use learned embeddings to convert the input tokens and output tokens to
vectors of dimension d_model. We also use the usual learned linear
transformation and softmax function to convert the decoder output to predicted
next-token probabilities. In our model, we share the same weight matrix between
the two embedding layers and the pre-softmax linear transformation ... In the
embedding layers, we multiply those weights by sqrt(d_model)."

Three distinct requirements, all implemented here:

  1. Learned token embeddings of dimension d_model.
  2. The SAME weight matrix object shared by (a) the source embedding,
     (b) the target embedding, and (c) the pre-softmax projection. Sharing
     requires a shared source-target vocabulary, which is why the paper's
     ~37k joint BPE vocabulary (Section 5.1) and this tying are a package.
     tests/test_architecture.py asserts object identity (`is`), not merely
     equal shapes -- a same-shape-but-distinct-tensor bug would train
     without error and silently triple the embedding parameter count.
  3. Multiplication by sqrt(d_model) before the positional encoding is added.

Why the sqrt(d_model) factor: embedding rows are initialized ~N(0, d_model^-0.5),
so after scaling their entries are O(1) and commensurate with the positional
encodings, which live in [-1, 1]. Without it the position signal would dominate
the token identity. The factor is applied at use-time (in Transformer.encode /
.decode), not baked into the stored weights, so the pre-softmax projection --
which shares those weights -- is NOT scaled. That asymmetry is deliberate and
matches the paper's wording ("in the embedding layers").

This module deliberately exposes *builders* rather than a wrapper Module, so
that parameter names in the state_dict remain `src_embed.weight` /
`tgt_embed.weight` / `generator.weight`. Introducing a wrapper Module here
would rename every key and invalidate previously written checkpoints.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


def build_embeddings(
    src_vocab_size: int,
    tgt_vocab_size: int,
    d_model: int,
    pad_id: int,
    tie: bool,
) -> Tuple[nn.Embedding, nn.Embedding, nn.Linear]:
    """Return (src_embed, tgt_embed, generator), tied per Section 3.4 if `tie`.

    The generator is bias-free so that, when tied, it is exactly the transpose
    of the embedding matrix.
    """
    if tie and src_vocab_size != tgt_vocab_size:
        raise ValueError("three-way tying requires a shared source-target vocabulary")

    src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_id)
    tgt_embed = src_embed if tie else nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_id)
    generator = nn.Linear(d_model, tgt_vocab_size, bias=False)
    if tie:
        generator.weight = tgt_embed.weight
    return src_embed, tgt_embed, generator


def init_embeddings(
    src_embed: nn.Embedding,
    tgt_embed: nn.Embedding,
    generator: nn.Linear,
    d_model: int,
    pad_id: int,
    tie: bool,
) -> None:
    """N(0, d_model^-0.5) init; PAD row zeroed. See module docstring."""
    nn.init.normal_(src_embed.weight, mean=0.0, std=d_model ** -0.5)
    with torch.no_grad():
        src_embed.weight[pad_id].zero_()
    if not tie:
        nn.init.normal_(tgt_embed.weight, mean=0.0, std=d_model ** -0.5)
        with torch.no_grad():
            tgt_embed.weight[pad_id].zero_()
        nn.init.normal_(generator.weight, mean=0.0, std=d_model ** -0.5)


def embedding_scale(d_model: int) -> float:
    """sqrt(d_model), the Section 3.4 multiplier."""
    return math.sqrt(d_model)


def verify_tying(model) -> dict:
    """Report whether the three matrices are the same object (not just equal)."""
    return {
        "src_is_tgt": model.src_embed.weight is model.tgt_embed.weight,
        "tgt_is_generator": model.generator.weight is model.tgt_embed.weight,
        "shapes_equal": (model.src_embed.weight.shape == model.generator.weight.shape),
        "embedding_scale": model.emb_scale,
    }


__all__ = ["build_embeddings", "init_embeddings", "embedding_scale", "verify_tying"]
