"""Mask construction (paper Sections 3.1 and 3.2.3).

Convention used everywhere in this project: **True means "may attend"**,
False means "suppress". Attention consumes masks broadcastable to
(B, H, Lq, Lk).

Two distinct kinds of mask are combined:

1. Padding mask. Batching variable-length sentences requires PAD filler.
   PAD positions must never contribute to any other position's
   representation. This is a *key-side* constraint, so its natural shape
   is (B, 1, 1, Lk) -- broadcast over heads and over all queries.

   The paper never discusses padding (it describes the model, not the
   batching), but omitting this mask silently changes the model: attention
   distributions would place probability mass on filler tokens and results
   would depend on batch composition. We treat it as a required part of a
   correct implementation and test it explicitly
   (tests/test_masking.py::test_padding_invariance).

2. Causal / subsequent mask. In the decoder self-attention, position i may
   attend to positions <= i. This is the lower-triangular mask, shape
   (1, 1, L, L). Paper: "masking out (setting to -inf) all values in the
   input of the softmax which correspond to illegal connections."

The decoder self-attention mask is the elementwise AND of the two.
"""

from __future__ import annotations

import torch


def padding_mask(tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """(B, L) token ids -> (B, 1, 1, L) bool mask, True where not PAD."""
    return (tokens != pad_id).unsqueeze(1).unsqueeze(2)


def causal_mask(length: int, device: torch.device | None = None) -> torch.Tensor:
    """(1, 1, L, L) lower-triangular bool mask, True on and below diagonal."""
    return torch.ones(length, length, dtype=torch.bool, device=device).tril().view(1, 1, length, length)


def decoder_mask(tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Combined causal + padding mask for decoder self-attention.

    tokens: (B, Lt) -> (B, 1, Lt, Lt)
    """
    L = tokens.size(1)
    return padding_mask(tokens, pad_id) & causal_mask(L, tokens.device)


__all__ = ["padding_mask", "causal_mask", "decoder_mask"]
