import torch

from transformer_reference_implementation import (
    MultiHeadAttention,
    PositionalEncoding,
    TransformerConfig,
    TransformerModel,
)


def small_config() -> TransformerConfig:
    return TransformerConfig(
        src_vocab_size=17,
        tgt_vocab_size=19,
        d_model=16,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=32,
        dropout=0.0,
        max_seq_len=32,
        pad_token_id=0,
    )


def test_transformer_forward_shape() -> None:
    model = TransformerModel(small_config())
    src = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
    tgt = torch.tensor([[1, 6, 7], [1, 8, 0]])

    logits = model(src, tgt)

    assert logits.shape == (2, 3, 19)


def test_multi_head_attention_returns_expected_shapes() -> None:
    attention = MultiHeadAttention(d_model=16, num_heads=4, dropout=0.0)
    x = torch.randn(2, 5, 16)

    output, weights = attention(x, x, x)

    assert output.shape == (2, 5, 16)
    assert weights.shape == (2, 4, 5, 5)


def test_positional_encoding_is_deterministic_in_eval_mode() -> None:
    encoding = PositionalEncoding(d_model=8, max_seq_len=16, dropout=0.0)
    x = torch.zeros(1, 4, 8)

    first = encoding(x)
    second = encoding(x)

    torch.testing.assert_close(first, second)
