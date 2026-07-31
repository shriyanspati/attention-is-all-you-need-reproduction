from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerConfig:
    """Configuration for an encoder-decoder Transformer."""

    src_vocab_size: int
    tgt_vocab_size: int
    d_model: int = 512
    num_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 5000
    pad_token_id: int = 0

    def __post_init__(self) -> None:
        if self.src_vocab_size <= 0 or self.tgt_vocab_size <= 0:
            raise ValueError("Vocabulary sizes must be positive.")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive.")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if self.num_encoder_layers <= 0 or self.num_decoder_layers <= 0:
            raise ValueError("Layer counts must be positive.")
        if self.d_ff <= 0:
            raise ValueError("d_ff must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the interval [0, 1).")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")
