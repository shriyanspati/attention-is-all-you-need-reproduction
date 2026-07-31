"""Transformer decoder stack (paper Section 3.1).

"The decoder is also composed of a stack of N = 6 identical layers. In
addition to the two sub-layers in each encoder layer, the decoder inserts
a third sub-layer, which performs multi-head attention over the output of
the encoder stack. ... We also modify the self-attention sub-layer in the
decoder stack to prevent positions from attending to subsequent
positions."

Sub-layer order per layer (Figure 1, bottom to top):
    1. masked multi-head self-attention over decoder states
    2. multi-head encoder-decoder ("cross") attention: queries from the
       decoder, keys and values from the encoder output (Section 3.2.3)
    3. position-wise feed-forward network

Each wrapped as LayerNorm(x + Dropout(Sublayer(x))) in the paper's
post-norm form; see encoder.py for the post-norm / pre-norm discussion.

The causal mask together with the one-position right shift of the output
embeddings gives the auto-regressive property: "the predictions for
position i can depend only on the known outputs at positions less than i".
The shift is applied in the data pipeline (decoder input = [BOS] + y[:-1],
target = y), not here; see data/dataset.py.

Shapes: B = batch, Lt = target length, Ls = source length, D = d_model.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    """One decoder layer: masked self-attn, cross-attn, feed-forward."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        relu_dropout: float = 0.0,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        norm_first: bool = False,
        layer_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model, num_heads, d_k=d_k, d_v=d_v, attention_dropout=attention_dropout
        )
        self.cross_attn = MultiHeadAttention(
            d_model, num_heads, d_k=d_k, d_v=d_v, attention_dropout=attention_dropout
        )
        self.ff = PositionwiseFeedForward(d_model, d_ff, relu_dropout=relu_dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(
        self,
        x: torch.Tensor,                            # (B, Lt, D)
        memory: torch.Tensor,                       # (B, Ls, D) encoder output
        tgt_mask: Optional[torch.Tensor] = None,    # (B, 1, Lt, Lt) causal & padding
        memory_mask: Optional[torch.Tensor] = None, # (B, 1, 1, Ls) source padding
        store_attention: bool = False,
    ) -> torch.Tensor:                              # (B, Lt, D)
        if self.norm_first:
            h = self.norm1(x)
            x = x + self.drop1(self.self_attn(h, h, h, tgt_mask))
            h = self.norm2(x)
            x = x + self.drop2(self.cross_attn(h, memory, memory, memory_mask, store_attention))
            x = x + self.drop3(self.ff(self.norm3(x)))
        else:
            x = self.norm1(x + self.drop1(self.self_attn(x, x, x, tgt_mask)))
            x = self.norm2(
                x + self.drop2(self.cross_attn(x, memory, memory, memory_mask, store_attention))
            )
            x = self.norm3(x + self.drop3(self.ff(x)))
        return x


class Decoder(nn.Module):
    """Stack of N identical decoder layers."""

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        relu_dropout: float = 0.0,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        norm_first: bool = False,
        layer_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            DecoderLayer(
                d_model, num_heads, d_ff, dropout, attention_dropout,
                relu_dropout, d_k, d_v, norm_first, layer_norm_eps,
            )
            for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps) if norm_first else None

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        store_attention: bool = False,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask, store_attention)
        return self.norm(x) if self.norm is not None else x


__all__ = ["Decoder", "DecoderLayer"]
