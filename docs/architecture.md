# Architecture

This implementation follows the encoder-decoder Transformer architecture described by Vaswani et al. The model maps source token ids and target-prefix token ids to target-vocabulary logits.

## Embeddings and Positional Encodings

Source and target tokens are embedded independently. Each embedding is multiplied by `sqrt(d_model)` before sinusoidal positional encodings are added. The positional encoding matrix is deterministic and is registered as a non-trainable buffer.

For even channel indices, the encoding uses sine. For odd channel indices, it uses cosine. Frequencies are spaced geometrically as in the original formulation.

## Multi-Head Attention

Multi-head attention projects queries, keys, and values into `num_heads` subspaces of width `d_model / num_heads`. Attention logits are scaled by the inverse square root of the per-head dimension. Boolean masks are applied before softmax, where `True` denotes a masked position.

The implementation supports:

- Encoder self-attention with source padding masks.
- Decoder self-attention with target padding masks and causal masks.
- Decoder cross-attention with source padding masks.

## Encoder Layer

Each encoder layer contains:

1. Multi-head self-attention.
2. Residual connection and layer normalization.
3. Position-wise feed-forward network.
4. Residual connection and layer normalization.

The feed-forward network applies a linear projection from `d_model` to `d_ff`, ReLU, dropout, and a second projection back to `d_model`.

## Decoder Layer

Each decoder layer contains:

1. Masked multi-head self-attention.
2. Residual connection and layer normalization.
3. Encoder-decoder cross-attention.
4. Residual connection and layer normalization.
5. Position-wise feed-forward network.
6. Residual connection and layer normalization.

## Output Projection

Decoder hidden states are projected to the target vocabulary with a learned linear layer. The repository does not tie input and output embeddings by default because the original architecture description can be implemented and studied without assuming vocabulary sharing.
