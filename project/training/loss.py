"""Label smoothing loss (paper Section 5.4).

"During training, we employed label smoothing of value eps_ls = 0.1 [30].
This hurts perplexity, as the model learns to be more unsure, but improves
accuracy and BLEU score."

Reference [30] is Szegedy et al. (Inception-v3), which replaces the
one-hot target q(k) = delta_{k,y} with

    q'(k) = (1 - eps) * delta_{k,y} + eps * u(k)

for a fixed distribution u. Szegedy use the uniform distribution
u(k) = 1/K over all K classes. The loss is then the cross-entropy
H(q', p), equivalently KL(q' || p) up to the constant H(q').

Two implementation questions the paper does not settle
------------------------------------------------------
1. Does the smoothing mass include the true class? Szegedy's formula does
   (the true class receives 1 - eps + eps/K). The widely copied
   "Annotated Transformer" implementation instead assigns confidence
   1 - eps to the true class and spreads eps over the remaining V - 2
   classes, excluding both the true class and PAD. We implement Szegedy's
   form as the default (`exclude_true=False`) since it is the cited source,
   and provide the alternative for comparison. The two differ by O(eps/V)
   in the target distribution, which is negligible for V ~ 37000 but not
   for the small vocabularies used at reduced scale, so it is worth being
   explicit rather than arbitrary.

2. Is PAD in the smoothing support? Assigning probability mass to PAD as a
   *prediction target* is incoherent -- PAD is never a legitimate output.
   We exclude PAD from the smoothing support and renormalize. Positions
   whose gold label is PAD are masked out of the loss entirely.

Normalization: the loss is summed over non-PAD target tokens and divided
by the number of such tokens ("per-token loss"). The paper reports
per-wordpiece perplexity in Table 3, which is consistent with per-token
normalization. Because gradients are accumulated over a token-count-based
batch, per-token normalization also makes the gradient magnitude
independent of how many sentences happened to land in the batch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """KL divergence to a label-smoothed target distribution.

    Parameters
    ----------
    vocab_size : int
    pad_id : int
        Excluded from the smoothing support and from the loss.
    smoothing : float
        eps_ls. 0.0 recovers exact cross-entropy (asserted in tests).
    exclude_true : bool
        If True, spread eps over classes other than the true one
        (Annotated-Transformer convention). If False (default), use
        Szegedy's uniform u including the true class.
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int = 0,
        smoothing: float = 0.1,
        exclude_true: bool = False,
    ) -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.smoothing = smoothing
        self.exclude_true = exclude_true
        # Number of classes carrying smoothing mass (PAD excluded).
        self.support = vocab_size - 1 - (1 if exclude_true else 0)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """logits: (B, L, V) or (N, V). target: (B, L) or (N,). Returns scalar.

        The returned value is the mean over non-PAD target positions.
        """
        V = logits.size(-1)
        logits = logits.reshape(-1, V)          # (N, V)
        target = target.reshape(-1)             # (N,)

        valid = target != self.pad_id           # (N,)
        n_valid = int(valid.sum())
        if n_valid == 0:
            return logits.sum() * 0.0

        logits = logits[valid]
        target = target[valid]
        logp = F.log_softmax(logits.float(), dim=-1)     # (n_valid, V)

        if self.smoothing == 0.0:
            return F.nll_loss(logp, target, reduction="mean")

        eps = self.smoothing
        off = eps / self.support
        with torch.no_grad():
            q = torch.full_like(logp, off)
            q[:, self.pad_id] = 0.0
            if self.exclude_true:
                q.scatter_(1, target.unsqueeze(1), 1.0 - eps)
            else:
                # Szegedy: true class keeps 1 - eps + eps/support
                q.scatter_add_(1, target.unsqueeze(1),
                               torch.full((target.size(0), 1), 1.0 - eps, device=q.device))
        # Cross-entropy H(q, p) summed over classes, averaged over tokens.
        return -(q * logp).sum(dim=-1).mean()


@torch.no_grad()
def token_cross_entropy(logits: torch.Tensor, target: torch.Tensor, pad_id: int = 0) -> tuple[float, int]:
    """Unsmoothed per-token CE (nats) and token count, for perplexity.

    Reported separately from the training objective because label smoothing
    deliberately inflates the smoothed loss; perplexity comparable to
    Table 3 must come from the unsmoothed likelihood.
    """
    V = logits.size(-1)
    logits = logits.reshape(-1, V)
    target = target.reshape(-1)
    valid = target != pad_id
    n = int(valid.sum())
    if n == 0:
        return 0.0, 0
    ce = F.cross_entropy(logits[valid].float(), target[valid], reduction="sum")
    return float(ce), n


__all__ = ["LabelSmoothingLoss", "token_cross_entropy"]
