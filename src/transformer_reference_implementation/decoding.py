import torch

from transformer_reference_implementation.masks import make_padding_mask
from transformer_reference_implementation.transformer import TransformerModel


@torch.no_grad()
def greedy_decode(
    model: TransformerModel,
    src: torch.Tensor,
    bos_token_id: int,
    eos_token_id: int,
    max_new_tokens: int,
) -> torch.Tensor:
    """Generate target tokens by greedy autoregressive decoding."""

    if src.ndim != 2:
        raise ValueError("src must have shape [batch, sequence].")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")

    was_training = model.training
    model.eval()

    src_mask = model._expand_padding_mask(make_padding_mask(src, model.config.pad_token_id))
    memory = model.encode(src, src_mask)
    generated = torch.full(
        (src.size(0), 1),
        bos_token_id,
        dtype=torch.long,
        device=src.device,
    )

    for _ in range(max_new_tokens):
        decoded = model.decode(generated, memory, memory_mask=src_mask)
        logits = model.output_projection(decoded[:, -1])
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        if torch.all(next_token.squeeze(1).eq(eos_token_id)):
            break

    model.train(was_training)
    return generated
