"""Scaled dot-product attention and multi-head attention.

Implements Vaswani et al. (2017), Sections 3.2.1 and 3.2.2.

Equation (1) of the paper:

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

Equation for multi-head attention (Section 3.2.2):

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    head_i             = Attention(Q W_i^Q, K W_i^K, V W_i^V)

with W_i^Q, W_i^K in R^{d_model x d_k}, W_i^V in R^{d_model x d_v},
W^O in R^{h*d_v x d_model}, and h = 8, d_k = d_v = d_model / h = 64
for the base model.

Design decisions and paper-fidelity notes
-----------------------------------------
1. The h per-head projections W_i^Q are implemented as a single fused
   Linear(d_model, h*d_k) followed by a reshape. This is mathematically
   identical to h separate projections (the heads do not interact before
   the softmax) but is far faster. Verified numerically in
   tests/test_attention.py::test_fused_projection_equals_per_head_loop.

2. Masking uses the convention `True = attend`, `False = suppress`.
   Suppressed logits are set to torch.finfo(dtype).min rather than
   -float('inf'). Both give identical softmax output in fp32 here, but
   the finite value avoids inf - inf = NaN if a row were ever fully
   masked, and avoids NaN under autocast. Verified in
   tests/test_masking.py::test_masked_positions_receive_zero_weight.

3. Dropout on the *attention weights* is NOT part of the paper's
   Section 5.4, which lists only residual dropout, embedding dropout,
   and label smoothing. The reference tensor2tensor code exposes a
   separate `attention_dropout` hyperparameter. We therefore default
   attention_dropout=0.0 (literal paper) and expose it as an option.
   This discrepancy is recorded in the reproduction report.

Shape conventions used throughout this file
-------------------------------------------
    B  = batch size
    Lq = query sequence length
    Lk = key/value sequence length
    H  = number of heads
    Dk = per-head key/query dimension
    Dv = per-head value dimension
    D  = d_model
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    query: torch.Tensor,          # (B, H, Lq, Dk)
    key: torch.Tensor,            # (B, H, Lk, Dk)
    value: torch.Tensor,          # (B, H, Lk, Dv)
    mask: Optional[torch.Tensor] = None,  # broadcastable to (B, H, Lq, Lk), True = attend
    dropout: Optional[nn.Dropout] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V.

    The 1/sqrt(d_k) factor is the paper's counter-measure (Section 3.2.1,
    footnote 4) against dot products growing with variance d_k and pushing
    the softmax into saturated, low-gradient regions.

    Returns
    -------
    output : (B, H, Lq, Dv)
    weights : (B, H, Lq, Lk)  post-softmax attention distribution, rows sum to 1
    """
    d_k = query.size(-1)

    # (B, H, Lq, Dk) @ (B, H, Dk, Lk) -> (B, H, Lq, Lk)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # Set illegal connections to the most negative representable value so
        # that softmax assigns them ~0 probability (paper: "setting to -inf").
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

    weights = torch.softmax(scores, dim=-1)          # (B, H, Lq, Lk)
    if dropout is not None:
        weights = dropout(weights)

    output = torch.matmul(weights, value)            # (B, H, Lq, Dv)
    return output, weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention (paper Section 3.2.2).

    Parameters
    ----------
    d_model : int
        Model dimension (512 for base, 1024 for big).
    num_heads : int
        h. Must divide d_model so that d_k = d_v = d_model / h, which keeps
        total compute comparable to single-head full-dimensional attention.
    d_k, d_v : int, optional
        Explicit per-head dimensions. Needed to reproduce Table 3 row (B),
        where d_k is reduced *without* changing h. Default d_model // h.
    attention_dropout : float
        Dropout on attention weights. 0.0 by default (see module docstring).
    bias : bool
        The paper writes the projections as parameter matrices with no bias
        term. tensor2tensor also uses bias-free projections. Default False.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        attention_dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_k is None or d_v is None:
            if d_model % num_heads != 0:
                raise ValueError(
                    f"d_model={d_model} must be divisible by num_heads={num_heads} "
                    "when d_k/d_v are not given explicitly"
                )
        self.d_model = d_model
        self.h = num_heads
        self.d_k = d_k if d_k is not None else d_model // num_heads
        self.d_v = d_v if d_v is not None else d_model // num_heads

        # W^Q, W^K in R^{d_model x h*d_k}; W^V in R^{d_model x h*d_v}
        self.w_q = nn.Linear(d_model, self.h * self.d_k, bias=bias)
        self.w_k = nn.Linear(d_model, self.h * self.d_k, bias=bias)
        self.w_v = nn.Linear(d_model, self.h * self.d_v, bias=bias)
        # W^O in R^{h*d_v x d_model}
        self.w_o = nn.Linear(self.h * self.d_v, d_model, bias=bias)

        self.attn_dropout = nn.Dropout(attention_dropout) if attention_dropout > 0 else None
        self.last_attention_weights: Optional[torch.Tensor] = None

    def _split_heads(self, x: torch.Tensor, d_head: int) -> torch.Tensor:
        """(B, L, H*d_head) -> (B, H, L, d_head)."""
        B, L, _ = x.shape
        return x.view(B, L, self.h, d_head).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, L, Dv) -> (B, L, H*Dv). This is the Concat of Figure 2."""
        B, H, L, Dv = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * Dv)

    def forward(
        self,
        query: torch.Tensor,                  # (B, Lq, D)
        key: torch.Tensor,                    # (B, Lk, D)
        value: torch.Tensor,                  # (B, Lk, D)
        mask: Optional[torch.Tensor] = None,  # broadcastable to (B, H, Lq, Lk)
        store_attention: bool = False,
    ) -> torch.Tensor:                        # (B, Lq, D)
        q = self._split_heads(self.w_q(query), self.d_k)   # (B, H, Lq, Dk)
        k = self._split_heads(self.w_k(key), self.d_k)     # (B, H, Lk, Dk)
        v = self._split_heads(self.w_v(value), self.d_v)   # (B, H, Lk, Dv)

        if mask is not None and mask.dim() == 3:
            # (B, Lq, Lk) -> (B, 1, Lq, Lk) so it broadcasts across heads
            mask = mask.unsqueeze(1)

        ctx, weights = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.attn_dropout
        )                                                  # (B, H, Lq, Dv)

        if store_attention:
            self.last_attention_weights = weights.detach()

        return self.w_o(self._merge_heads(ctx))            # (B, Lq, D)


__all__ = ["scaled_dot_product_attention", "MultiHeadAttention"]
