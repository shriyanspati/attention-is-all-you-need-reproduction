"""Sinusoidal positional encoding (paper Section 3.5).

The paper defines

    PE(pos, 2i)   = sin(pos / 10000^{2i / d_model})
    PE(pos, 2i+1) = cos(pos / 10000^{2i / d_model})

where `pos` is the position and `i` indexes the dimension pair. Note that
the *same* exponent 2i/d_model is used for the sine at column 2i and the
cosine at column 2i+1: columns come in (sin, cos) pairs sharing one
frequency. For d_model = 512 there are 256 distinct frequencies.

Two paper claims are asserted here and verified in
tests/test_positional_encoding.py:

  (a) "The wavelengths form a geometric progression from 2*pi to
       10000 * 2*pi."  With omega_i = 1 / 10000^{2i/d} and wavelength
       lambda_i = 2*pi / omega_i, we get lambda_0 = 2*pi and
       lambda_{d/2-1} = 2*pi * 10000^{(d-2)/d}, which approaches
       10000 * 2*pi as d grows (it is 10000^{510/512} * 2*pi for d=512,
       i.e. 0.982 of the stated endpoint). The paper's "to 10000 * 2*pi"
       is therefore an asymptotic statement, not an exact endpoint --
       a small imprecision we flag rather than silently "fix".

  (b) "for any fixed offset k, PE_{pos+k} can be represented as a linear
       function of PE_pos."  This is exactly true: each (sin, cos) pair
       rotates by angle omega_i * k, so PE_{pos+k} = R(k) @ PE_pos with
       R(k) a fixed block-diagonal matrix of 2x2 rotations, independent
       of pos. The test recovers R(k) and checks it numerically.

Implementation notes
--------------------
* Computed in float64 then cast, so that the largest arguments
  (pos up to 10^4, frequencies down to 10^-4) do not accumulate
  avoidable fp32 error in the sin/cos arguments.
* Registered as a non-persistent buffer: it is a deterministic function
  of (max_len, d_model) and carries no learned state, so keeping it out
  of the checkpoint keeps checkpoints faithful to "parameter-free
  position representation" (paper, author-contribution footnote).
* Applied as `dropout(x * sqrt(d_model) + PE)`. The sqrt(d_model)
  embedding scaling belongs to Section 3.4 and is applied in
  transformer.py; the dropout on the *sum* of embeddings and positional
  encodings is Section 5.4 ("we apply dropout to the sums of the
  embeddings and the positional encodings").
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal position signal, added to token embeddings.

    Parameters
    ----------
    d_model : int
    dropout : float
        Rate for the dropout applied to the embedding + PE sum (P_drop = 0.1).
    max_len : int
        Table size. Sinusoids extrapolate beyond it in principle (that is the
        paper's stated motivation for choosing them over learned embeddings),
        but the buffer must be pre-sized; `extend_to` grows it on demand.
    base : float
        The 10000 constant.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.base = base
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pe", self._build(max_len), persistent=False)

    def _build(self, max_len: int) -> torch.Tensor:
        """Return the (1, max_len, d_model) position table."""
        d = self.d_model
        pos = torch.arange(max_len, dtype=torch.float64).unsqueeze(1)   # (L, 1)
        # i = 0, 1, ..., floor(d/2)-1  ->  exponent 2i/d
        two_i = torch.arange(0, d, 2, dtype=torch.float64)              # (ceil(d/2),)
        inv_freq = torch.pow(self.base, -two_i / d)                     # (ceil(d/2),)

        pe = torch.zeros(max_len, d, dtype=torch.float64)               # (L, D)
        angles = pos * inv_freq                                         # (L, ceil(d/2))
        pe[:, 0::2] = torch.sin(angles)
        # For odd d_model the cosine block is one column shorter.
        pe[:, 1::2] = torch.cos(angles[:, : pe[:, 1::2].size(1)])
        return pe.to(torch.float32).unsqueeze(0)                        # (1, L, D)

    def extend_to(self, length: int) -> None:
        if length > self.pe.size(1):
            self.pe = self._build(length).to(self.pe.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) already scaled by sqrt(d_model). Returns (B, L, D)."""
        L = x.size(1)
        self.extend_to(L)
        return self.dropout(x + self.pe[:, :L].to(x.dtype))

    def wavelengths(self) -> torch.Tensor:
        """lambda_i = 2*pi * base^{2i/d_model}; used to check paper claim (a)."""
        d = self.d_model
        two_i = torch.arange(0, d, 2, dtype=torch.float64)
        return 2 * math.pi * torch.pow(self.base, two_i / d)


class LearnedPositionalEmbedding(nn.Module):
    """Learned position embeddings, for Table 3 row (E).

    The paper reports "nearly identical results" to the sinusoidal version;
    this module exists so that claim can be tested rather than assumed.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.emb = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.emb.weight, mean=0.0, std=d_model ** -0.5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        pos = torch.arange(L, device=x.device)
        return self.dropout(x + self.emb(pos).unsqueeze(0))


class NoPositionalEncoding(nn.Module):
    """Ablation: no position information at all (Phase 8 experiment 2).

    Keeps the embedding-sum dropout so the ablation isolates *only* the
    position signal.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, **_: object) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x)


def build_position_encoding(kind: str, d_model: int, dropout: float, max_len: int = 5000) -> nn.Module:
    kinds = {
        "sinusoidal": SinusoidalPositionalEncoding,
        "learned": LearnedPositionalEmbedding,
        "none": NoPositionalEncoding,
    }
    if kind not in kinds:
        raise ValueError(f"unknown positional encoding {kind!r}; choose from {sorted(kinds)}")
    return kinds[kind](d_model=d_model, dropout=dropout, max_len=max_len)


__all__ = [
    "SinusoidalPositionalEncoding",
    "LearnedPositionalEmbedding",
    "NoPositionalEncoding",
    "build_position_encoding",
]
