"""The Transformer (Vaswani et al., 2017), assembled end to end.

Covers paper Sections 3.1-3.5 plus the inference procedure of Section 6.1
(beam size 4, length penalty alpha = 0.6, max output length = input + 50).

Embeddings and softmax (Section 3.4)
------------------------------------
"we share the same weight matrix between the two embedding layers and the
pre-softmax linear transformation ... In the embedding layers, we multiply
those weights by sqrt(d_model)."

Three-way weight tying is the default here, which is only coherent because
the paper uses a *shared* source-target BPE vocabulary of ~37k tokens for
EN-DE (Section 5.1). `tie_embeddings=False` is available for the separate
-vocabulary case (their EN-FR setup uses a 32k word-piece vocabulary).

The sqrt(d_model) factor: embedding rows are initialized ~N(0, d_model^-1/2)
so that after multiplication by sqrt(d_model) their entries are O(1) and
comparable in scale to the positional encodings, which live in [-1, 1].
Without the factor the position signal would swamp the token identity.

Initialization -- an unspecified detail
---------------------------------------
The paper does not state its initialization scheme. We use Xavier/Glorot
uniform for all projection matrices (the tensor2tensor default of that
era) and N(0, d_model^-0.5) for the shared embedding, and record this as
an under-determined choice in the reproduction report rather than
presenting it as "the paper's" scheme.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder
from .masking import causal_mask, decoder_mask, padding_mask
from .positional_encoding import build_position_encoding


@dataclass
class TransformerConfig:
    """All architectural hyperparameters in one serializable place.

    Defaults are the paper's *base* model (Table 3, row "base").
    """

    src_vocab_size: int = 37000
    tgt_vocab_size: int = 37000
    num_encoder_layers: int = 6          # N
    num_decoder_layers: int = 6          # N
    d_model: int = 512                   # d_model
    num_heads: int = 8                   # h
    d_ff: int = 2048                     # d_ff
    d_k: Optional[int] = None            # default d_model // h = 64
    d_v: Optional[int] = None            # default d_model // h = 64
    dropout: float = 0.1                 # P_drop (residual + embedding sums)
    attention_dropout: float = 0.0       # not in paper text; see attention.py
    relu_dropout: float = 0.0            # not in paper text; see feed_forward.py
    positional_encoding: str = "sinusoidal"   # "sinusoidal" | "learned" | "none"
    max_len: int = 512
    tie_embeddings: bool = True          # Section 3.4 three-way tying
    norm_first: bool = False             # False = paper's post-norm
    layer_norm_eps: float = 1e-6
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def base(cls, **kw) -> "TransformerConfig":
        return cls(**kw)

    @classmethod
    def big(cls, **kw) -> "TransformerConfig":
        """Table 3, row "big": N=6, d_model=1024, d_ff=4096, h=16, P_drop=0.3."""
        defaults = dict(d_model=1024, d_ff=4096, num_heads=16, dropout=0.3)
        defaults.update(kw)
        return cls(**defaults)


class Transformer(nn.Module):
    """Encoder-decoder Transformer."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = c = config

        self.src_embed = nn.Embedding(c.src_vocab_size, c.d_model, padding_idx=c.pad_id)
        if c.tie_embeddings:
            if c.src_vocab_size != c.tgt_vocab_size:
                raise ValueError("tie_embeddings requires a shared source-target vocabulary")
            self.tgt_embed = self.src_embed
        else:
            self.tgt_embed = nn.Embedding(c.tgt_vocab_size, c.d_model, padding_idx=c.pad_id)

        self.src_pos = build_position_encoding(c.positional_encoding, c.d_model, c.dropout, c.max_len)
        self.tgt_pos = build_position_encoding(c.positional_encoding, c.d_model, c.dropout, c.max_len)

        self.encoder = Encoder(
            c.num_encoder_layers, c.d_model, c.num_heads, c.d_ff, c.dropout,
            c.attention_dropout, c.relu_dropout, c.d_k, c.d_v, c.norm_first, c.layer_norm_eps,
        )
        self.decoder = Decoder(
            c.num_decoder_layers, c.d_model, c.num_heads, c.d_ff, c.dropout,
            c.attention_dropout, c.relu_dropout, c.d_k, c.d_v, c.norm_first, c.layer_norm_eps,
        )

        # Pre-softmax linear transformation. Bias-free so that tying makes the
        # projection exactly the transpose of the embedding matrix.
        self.generator = nn.Linear(c.d_model, c.tgt_vocab_size, bias=False)
        if c.tie_embeddings:
            self.generator.weight = self.tgt_embed.weight

        self._init_parameters()
        self.emb_scale = math.sqrt(c.d_model)

    def _init_parameters(self) -> None:
        for name, p in self.named_parameters():
            if "embed" in name or "generator" in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        nn.init.normal_(self.src_embed.weight, mean=0.0, std=self.config.d_model ** -0.5)
        with torch.no_grad():
            self.src_embed.weight[self.config.pad_id].zero_()
        if not self.config.tie_embeddings:
            nn.init.normal_(self.tgt_embed.weight, mean=0.0, std=self.config.d_model ** -0.5)
            with torch.no_grad():
                self.tgt_embed.weight[self.config.pad_id].zero_()
            nn.init.normal_(self.generator.weight, mean=0.0, std=self.config.d_model ** -0.5)

    # ---------------------------------------------------------------- forward

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """src: (B, Ls) ids -> memory: (B, Ls, D)."""
        if src_mask is None:
            src_mask = padding_mask(src, self.config.pad_id)
        x = self.src_pos(self.src_embed(src) * self.emb_scale)   # (B, Ls, D)
        return self.encoder(x, src_mask)

    def decode(
        self,
        tgt_in: torch.Tensor,                        # (B, Lt) decoder input (already shifted)
        memory: torch.Tensor,                        # (B, Ls, D)
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        store_attention: bool = False,
    ) -> torch.Tensor:                               # (B, Lt, D)
        if tgt_mask is None:
            tgt_mask = decoder_mask(tgt_in, self.config.pad_id)
        x = self.tgt_pos(self.tgt_embed(tgt_in) * self.emb_scale)
        return self.decoder(x, memory, tgt_mask, memory_mask, store_attention)

    def forward(
        self,
        src: torch.Tensor,                           # (B, Ls)
        tgt_in: torch.Tensor,                        # (B, Lt)
    ) -> torch.Tensor:                               # (B, Lt, V) logits
        src_mask = padding_mask(src, self.config.pad_id)
        memory = self.encode(src, src_mask)
        h = self.decode(tgt_in, memory, memory_mask=src_mask)
        return self.generator(h)

    # -------------------------------------------------------------- inference

    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor, max_extra: int = 50) -> torch.Tensor:
        """Argmax decoding. src: (B, Ls) -> (B, <=Ls+max_extra) generated ids.

        Max output length follows Section 6.1: "input length + 50".
        """
        c = self.config
        self.eval()
        B = src.size(0)
        src_mask = padding_mask(src, c.pad_id)
        memory = self.encode(src, src_mask)
        max_len = int(src.size(1)) + max_extra

        ys = torch.full((B, 1), c.bos_id, dtype=torch.long, device=src.device)
        finished = torch.zeros(B, dtype=torch.bool, device=src.device)
        for _ in range(max_len - 1):
            logits = self.generator(self.decode(ys, memory, memory_mask=src_mask))  # (B, t, V)
            nxt = logits[:, -1].argmax(-1)                                          # (B,)
            nxt = torch.where(finished, torch.full_like(nxt, c.pad_id), nxt)
            ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
            finished |= nxt == c.eos_id
            if bool(finished.all()):
                break
        return ys

    @torch.no_grad()
    def beam_search(
        self,
        src: torch.Tensor,          # (1, Ls) -- single sentence
        beam_size: int = 4,
        length_penalty_alpha: float = 0.6,
        max_extra: int = 50,
    ) -> torch.Tensor:
        """Beam search with the GNMT length penalty used by the paper.

        Section 6.1: "We used beam search with a beam size of 4 and length
        penalty alpha = 0.6 [31]". Reference [31] is Wu et al. (GNMT), whose
        penalty is

            lp(Y) = ((5 + |Y|) / (5 + 1))^alpha

        and hypotheses are ranked by log P(Y|X) / lp(Y). The paper gives only
        the value of alpha, so the functional form is inherited from [31];
        this is recorded as an inferred (not stated) detail.

        Returns the best hypothesis as (1, L) including BOS and EOS.
        """
        c = self.config
        self.eval()
        assert src.size(0) == 1, "beam_search decodes one sentence at a time"
        device = src.device
        src_mask = padding_mask(src, c.pad_id)
        memory = self.encode(src, src_mask)                       # (1, Ls, D)
        max_len = int(src.size(1)) + max_extra

        mem = memory.expand(beam_size, -1, -1)                    # (K, Ls, D)
        mmask = src_mask.expand(beam_size, -1, -1, -1)
        ys = torch.full((beam_size, 1), c.bos_id, dtype=torch.long, device=device)
        scores = torch.full((beam_size,), float("-inf"), device=device)
        scores[0] = 0.0                                           # only one live beam at t=0
        finished: list[tuple[float, torch.Tensor]] = []

        for _ in range(max_len - 1):
            logits = self.generator(self.decode(ys, mem, memory_mask=mmask))[:, -1]   # (K, V)
            logp = torch.log_softmax(logits.float(), dim=-1)                          # (K, V)
            cand = scores.unsqueeze(1) + logp                                         # (K, V)
            flat = cand.view(-1)
            top_scores, top_idx = flat.topk(beam_size)
            beam_idx = torch.div(top_idx, logp.size(-1), rounding_mode="floor")
            token_idx = top_idx % logp.size(-1)

            ys = torch.cat([ys[beam_idx], token_idx.unsqueeze(1)], dim=1)
            scores = top_scores

            eos = token_idx == c.eos_id
            if bool(eos.any()):
                for i in eos.nonzero(as_tuple=False).flatten().tolist():
                    length = ys.size(1)
                    lp = ((5.0 + length) / 6.0) ** length_penalty_alpha
                    finished.append((float(scores[i]) / lp, ys[i].clone()))
                    scores[i] = float("-inf")
            if len(finished) >= beam_size or bool(torch.isinf(scores).all()):
                break

        if not finished:
            length = ys.size(1)
            lp = ((5.0 + length) / 6.0) ** length_penalty_alpha
            finished = [(float(scores[0]) / lp, ys[0])]
        best = max(finished, key=lambda t: t[0])[1]
        return best.unsqueeze(0)

    # ----------------------------------------------------------------- counts

    def count_parameters(self, trainable_only: bool = True) -> int:
        ps = self.parameters()
        return sum(p.numel() for p in ps if p.requires_grad or not trainable_only)

    def parameter_breakdown(self) -> dict:
        """Per-component parameter counts, for comparison with Table 3."""
        out: dict[str, int] = {}
        seen: set[int] = set()
        for name, p in self.named_parameters():
            if id(p) in seen:          # tied weights counted once
                continue
            seen.add(id(p))
            top = name.split(".")[0]
            out[top] = out.get(top, 0) + p.numel()
        out["total"] = sum(out.values())
        return out


__all__ = ["Transformer", "TransformerConfig"]
