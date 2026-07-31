import torch

from transformer_reference_implementation import TransformerConfig, TransformerModel, greedy_decode


def test_greedy_decode_returns_bos_prefixed_sequence() -> None:
    config = TransformerConfig(
        src_vocab_size=11,
        tgt_vocab_size=13,
        d_model=8,
        num_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        d_ff=16,
        dropout=0.0,
        max_seq_len=16,
    )
    model = TransformerModel(config)
    src = torch.tensor([[1, 2, 0]])

    generated = greedy_decode(
        model,
        src,
        bos_token_id=1,
        eos_token_id=2,
        max_new_tokens=3,
    )

    assert generated.shape[0] == 1
    assert generated.shape[1] >= 2
    assert generated[0, 0].item() == 1
