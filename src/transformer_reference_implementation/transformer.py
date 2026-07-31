import math

import torch
from torch import nn

from transformer_reference_implementation.config import TransformerConfig
from transformer_reference_implementation.masks import make_causal_mask, make_padding_mask
from transformer_reference_implementation.modules import (
    MultiHeadAttention,
    PositionalEncoding,
    PositionwiseFeedForward,
)


class EncoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(config.d_model, config.num_heads, config.dropout)
        self.feed_forward = PositionwiseFeedForward(config.d_model, config.d_ff, config.dropout)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None) -> torch.Tensor:
        attention_output, _ = self.self_attention(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attention_output))
        ff_output = self.feed_forward(x)
        return self.norm2(x + self.dropout(ff_output))


class DecoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(config.d_model, config.num_heads, config.dropout)
        self.cross_attention = MultiHeadAttention(config.d_model, config.num_heads, config.dropout)
        self.feed_forward = PositionwiseFeedForward(config.d_model, config.d_ff, config.dropout)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None,
        memory_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        self_attention_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attention_output))
        cross_attention_output, _ = self.cross_attention(x, memory, memory, memory_mask)
        x = self.norm2(x + self.dropout(cross_attention_output))
        ff_output = self.feed_forward(x)
        return self.norm3(x + self.dropout(ff_output))


class TransformerModel(nn.Module):
    """Encoder-decoder Transformer for sequence-to-sequence token modeling."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.src_embedding = nn.Embedding(
            config.src_vocab_size,
            config.d_model,
            padding_idx=config.pad_token_id,
        )
        self.tgt_embedding = nn.Embedding(
            config.tgt_vocab_size,
            config.d_model,
            padding_idx=config.pad_token_id,
        )
        self.src_positional_encoding = PositionalEncoding(
            config.d_model,
            config.max_seq_len,
            config.dropout,
        )
        self.tgt_positional_encoding = PositionalEncoding(
            config.d_model,
            config.max_seq_len,
            config.dropout,
        )
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(config.num_encoder_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_decoder_layers)]
        )
        self.output_projection = nn.Linear(config.d_model, config.tgt_vocab_size)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        if src_mask is None:
            src_mask = self._expand_padding_mask(make_padding_mask(src, self.config.pad_token_id))

        x = self.src_embedding(src) * math.sqrt(self.config.d_model)
        x = self.src_positional_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tgt_mask is None:
            tgt_mask = self._make_decoder_self_mask(tgt)

        x = self.tgt_embedding(tgt) * math.sqrt(self.config.d_model)
        x = self.tgt_positional_encoding(x)
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return x

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_padding_mask = make_padding_mask(src, self.config.pad_token_id)
        memory_mask = self._expand_padding_mask(src_padding_mask)
        memory = self.encode(src, memory_mask)
        decoded = self.decode(tgt, memory, self._make_decoder_self_mask(tgt), memory_mask)
        return self.output_projection(decoded)

    def _make_decoder_self_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        padding_mask = self._expand_padding_mask(make_padding_mask(tgt, self.config.pad_token_id))
        causal_mask = make_causal_mask(tgt.size(1), device=tgt.device).unsqueeze(0).unsqueeze(0)
        return padding_mask | causal_mask

    @staticmethod
    def _expand_padding_mask(mask: torch.Tensor) -> torch.Tensor:
        return mask.unsqueeze(1).unsqueeze(2)
