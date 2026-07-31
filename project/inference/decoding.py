"""Inference: greedy decoding, beam search, and checkpoint averaging.

Paper Section 6.1: "we used a single model obtained by averaging the last 5
checkpoints ... For the big models, we averaged the last 20 checkpoints. We
used beam search with a beam size of 4 and length penalty alpha = 0.6 [31].
We set the maximum output length during inference to input length + 50."

Reference [31] is Wu et al. (GNMT), whose length penalty is

    lp(Y) = ((5 + |Y|) / (5 + 1))^alpha

and hypotheses are ranked by log P(Y|X) / lp(Y). The paper states only alpha,
so the functional form is inherited from [31] -- an inferred, not stated,
detail. Without a length penalty, beam search is biased toward short outputs
because every additional token multiplies in a probability < 1.

`Transformer.greedy_decode` and `Transformer.beam_search` delegate to the
functions here, so this module is the single definition of the decoding
procedure and the model file stays architectural.

IMPORTANT (evaluation methodology): greedy decoding of a single checkpoint is
NOT comparable to the paper's reported BLEU, which uses beam-4 search over an
averaged checkpoint. `evaluate_like_paper` runs the paper's procedure so the
comparison is like-for-like.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import torch


def length_penalty(length: int, alpha: float = 0.6) -> float:
    """GNMT lp(Y) = ((5 + |Y|) / 6)^alpha. Returns 1.0 at |Y| = 1."""
    return ((5.0 + length) / 6.0) ** alpha


@torch.no_grad()
def average_checkpoints(paths: Sequence[str | Path], model):
    """Uniform parameter mean over checkpoints (Section 6.1).

    Averaging nearby SGD iterates approximates a flatter solution and is worth
    roughly 0.3-0.6 BLEU in the original setup. Applied to PARAMETERS, not to
    predictions -- it is not an ensemble.
    """
    paths = list(paths)
    if not paths:
        raise ValueError("no checkpoints to average")
    avg = None
    for p in paths:
        sd = torch.load(p, map_location="cpu", weights_only=False)["model"]
        if avg is None:
            avg = {k: v.clone().float() for k, v in sd.items()}
        else:
            for k in avg:
                avg[k] += sd[k].float()
    for k in avg:
        avg[k] /= len(paths)
    model.load_state_dict({k: v.to(torch.float32) for k, v in avg.items()})
    return model


def select_checkpoints(run_dir: str | Path, last_n: int = 5) -> List[Path]:
    """Last-N periodic checkpoints, oldest first. Paper: 5 for base, 20 for big."""
    cks = sorted(Path(run_dir).glob("ckpt_*.pt"))
    return cks[-last_n:]


__all__ = ["length_penalty", "average_checkpoints", "select_checkpoints"]
