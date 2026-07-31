"""Transformer encoder stack (paper Section 3.1).

"The encoder is composed of a stack of N = 6 identical layers. Each layer
has two sub-layers. The first is a multi-head self-attention mechanism,
and the second is a simple, position-wise fully connected feed-forward
network. We employ a residual connection around each of the two
sub-layers, followed by layer normalization. That is, the output of each
sub-layer is LayerNorm(x + Sublayer(x))."

Residual dropout (Section 5.4): "We apply dropout to the output of each
sub-layer, before it is added to the sub-layer input and normalized."
So the exact composition the paper describes is

    x <- LayerNorm(x + Dropout(Sublayer(x)))                     [POST-NORM]

POST-NORM vs PRE-NORM -- a deliberate reproduction decision
-----------------------------------------------------------
The formula quoted above is unambiguous: normalization is applied *after*
the residual addition. However, the authors' own reference implementation
(tensor2tensor) applies normalization to the sub-layer *input* and leaves
the residual path un-normalized:

    x <- x + Dropout(Sublayer(LayerNorm(x)))                     [PRE-NORM]

with a single LayerNorm applied after the whole stack. Later analyses
(e.g. Xiong et al. 2020, "On Layer Normalization in the Transformer
Architecture") showed the two behave quite differently in optimization:
post-norm has large gradients at initialization near the output layers
and genuinely *requires* learning-rate warmup, whereas pre-norm trains
stably without it.

We implement POST-NORM as the default because this study reproduces the
*paper*, and we expose `norm_first=True` for the pre-norm variant. Both
are exercised in the Phase 8 ablations, which lets us test the claim that
the paper's warmup schedule is load-bearing for the post-norm formulation
it describes. This is precisely the kind of discrepancy that "do not
blindly trust existing implementations" is meant to catch.

Shapes: B = batch, L = source length, D = d_model.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """One encoder layer: self-attention sub-layer + feed-forward sub-layer."""

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
        self.ff = PositionwiseFeedForward(d_model, d_ff, relu_dropout=relu_dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(
        self,
        x: torch.Tensor,                      # (B, L, D)
        src_mask: Optional[torch.Tensor] = None,  # (B, 1, 1, L) or (B, 1, L, L), True = attend
        store_attention: bool = False,
    ) -> torch.Tensor:                        # (B, L, D)
        if self.norm_first:
            h = self.norm1(x)
            x = x + self.drop1(self.self_attn(h, h, h, src_mask, store_attention))
            x = x + self.drop2(self.ff(self.norm2(x)))
        else:
            # Paper form: LayerNorm(x + Dropout(Sublayer(x)))
            x = self.norm1(x + self.drop1(self.self_attn(x, x, x, src_mask, store_attention)))
            x = self.norm2(x + self.drop2(self.ff(x)))
        return x


class Encoder(nn.Module):
    """Stack of N identical encoder layers.

    A final LayerNorm is applied only in the pre-norm variant, where it is
    required (the stack output is otherwise an unnormalized residual sum).
    The post-norm variant needs none: its last operation is already a
    LayerNorm.
    """

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
            EncoderLayer(
                d_model, num_heads, d_ff, dropout, attention_dropout,
                relu_dropout, d_k, d_v, norm_first, layer_norm_eps,
            )
            for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps) if norm_first else None

    def forward(
        self,
        x: torch.Tensor,                          # (B, L, D)
        src_mask: Optional[torch.Tensor] = None,
        store_attention: bool = False,
    ) -> torch.Tensor:                            # (B, L, D)
        for layer in self.layers:
            x = layer(x, src_mask, store_attention)
        return self.norm(x) if self.norm is not None else x


__all__ = ["Encoder", "EncoderLayer"]
