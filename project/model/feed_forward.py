"""Position-wise feed-forward network (paper Section 3.3).

    FFN(x) = max(0, x W_1 + b_1) W_2 + b_2                          (eq. 2)

Applied "to each position separately and identically": it is a shared
two-layer MLP broadcast over the sequence axis, equivalently "two
convolutions with kernel size 1". Parameters differ from layer to layer
but not from position to position.

Dimensions: d_model = 512 in and out, d_ff = 2048 inner (base model).
Table 3 row (C) varies d_ff over {1024, 4096}, so it is a constructor
argument, not a constant.

Note on bias terms: unlike the attention projections, equation (2)
explicitly includes b_1 and b_2, so these Linear layers DO use bias.

Note on dropout: the paper's Section 5.4 does not describe dropout
between the two linear layers ("relu_dropout" in tensor2tensor). We
default it to 0.0 to follow the paper text and expose it as an option.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    """Two linear transformations with a ReLU in between."""

    def __init__(self, d_model: int, d_ff: int, relu_dropout: float = 0.0) -> None:
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff, bias=True)    # (D -> d_ff)
        self.w_2 = nn.Linear(d_ff, d_model, bias=True)    # (d_ff -> D)
        self.relu_dropout = nn.Dropout(relu_dropout) if relu_dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) -> (B, L, D)."""
        h = F.relu(self.w_1(x))              # (B, L, d_ff)
        if self.relu_dropout is not None:
            h = self.relu_dropout(h)
        return self.w_2(h)                   # (B, L, D)


__all__ = ["PositionwiseFeedForward"]
