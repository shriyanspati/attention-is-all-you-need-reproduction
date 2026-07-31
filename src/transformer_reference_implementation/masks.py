import torch


def make_padding_mask(tokens: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """Return a boolean mask with True at padding positions."""

    if tokens.ndim != 2:
        raise ValueError("tokens must have shape [batch, sequence].")
    return tokens.eq(pad_token_id)


def make_causal_mask(sequence_length: int, device: torch.device | None = None) -> torch.Tensor:
    """Return an upper-triangular causal mask with True above the diagonal."""

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")
    return torch.triu(
        torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device),
        diagonal=1,
    )
