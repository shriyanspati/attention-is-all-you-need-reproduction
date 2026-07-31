import torch

from transformer_reference_implementation.masks import make_causal_mask, make_padding_mask


def test_padding_mask_marks_pad_tokens() -> None:
    tokens = torch.tensor([[1, 0, 2], [0, 0, 3]])
    mask = make_padding_mask(tokens, pad_token_id=0)

    assert mask.tolist() == [[False, True, False], [True, True, False]]


def test_causal_mask_excludes_future_positions() -> None:
    mask = make_causal_mask(4)

    assert mask.tolist() == [
        [False, True, True, True],
        [False, False, True, True],
        [False, False, False, True],
        [False, False, False, False],
    ]
