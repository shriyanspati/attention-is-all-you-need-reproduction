import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with a fixed maximum sequence length."""

    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")

        position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_seq_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, sequence, d_model].")
        if x.size(1) > self.pe.size(1):
            raise ValueError("Input sequence length exceeds configured max_seq_len.")
        return self.dropout(x + self.pe[:, : x.size(1)].to(dtype=x.dtype))


class MultiHeadAttention(nn.Module):
    """Multi-head scaled dot-product attention."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)
        q = self._shape(self.q_proj(query), batch_size)
        k = self._shape(self.k_proj(key), batch_size)
        v = self._shape(self.v_proj(value), batch_size)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        context = torch.matmul(attention, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, query.size(1), self.d_model)
        return self.out_proj(context), attention

    def _shape(self, tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
        return tensor.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)


class PositionwiseFeedForward(nn.Module):
    """Position-wise feed-forward network used inside Transformer blocks."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
