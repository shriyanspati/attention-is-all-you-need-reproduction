"""Learning rate schedule (paper Section 5.3, equation 3).

    lrate = d_model^{-0.5} * min(step_num^{-0.5}, step_num * warmup_steps^{-1.5})

with warmup_steps = 4000. "This corresponds to increasing the learning rate
linearly for the first warmup_steps training steps, and decreasing it
thereafter proportionally to the inverse square root of the step number."

Properties that follow from the formula and are asserted in
tests/test_scheduler.py:

* step_num is 1-based. At step_num = 0 the formula gives lr = 0, and
  step_num^{-0.5} would divide by zero, so the first optimizer step must be
  step 1.
* The two branches cross exactly at step_num = warmup_steps:
      step^{-0.5} = step * warmup^{-1.5}  <=>  step^{1.5} = warmup^{1.5}.
  So the peak learning rate is d_model^{-0.5} * warmup_steps^{-0.5}.
  For the base model: 512^{-0.5} * 4000^{-0.5} = 6.99e-4.
* Strictly increasing on [1, warmup], strictly decreasing after.

Note that peak LR depends on d_model: the big model (d_model = 1024) gets
a peak of 4.94e-4, i.e. the schedule automatically shrinks the step size
for wider models. This coupling is easy to lose when reimplementing the
schedule as a bare "warmup then inverse-sqrt" and is worth preserving.

Scaling to a shortened budget
-----------------------------
This reproduction cannot run 100k steps. Keeping warmup_steps = 4000 while
training for only a few thousand steps would leave the model permanently
in the warmup ramp and never exercise the decay branch, so the *shape* of
the schedule -- not just its formula -- would be unreproduced. We therefore
preserve the ratio warmup_steps / total_steps = 4000 / 100000 = 4% and
document the substitution. `Adam` betas and epsilon are unchanged.
"""

from __future__ import annotations

from typing import Iterator

import torch


def noam_learning_rate(step: int, d_model: int, warmup_steps: int, factor: float = 1.0) -> float:
    """Equation (3). `step` is 1-based; returns 0.0 for step <= 0."""
    if step <= 0:
        return 0.0
    return factor * (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)


class NoamScheduler:
    """Wraps an optimizer and sets the LR from equation (3) before each step.

    Deliberately not a torch.optim.lr_scheduler._LRScheduler subclass: the
    formula is short enough that an explicit implementation is easier to
    verify against the paper than one relying on base-class step bookkeeping.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int,
        warmup_steps: int = 4000,
        factor: float = 1.0,
        last_step: int = 0,
    ) -> None:
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self._step = last_step
        self._lr = 0.0

    @property
    def step_num(self) -> int:
        return self._step

    @property
    def lr(self) -> float:
        return self._lr

    def step(self) -> float:
        """Advance the step counter, apply the new LR, then step the optimizer."""
        self._step += 1
        self._lr = noam_learning_rate(self._step, self.d_model, self.warmup_steps, self.factor)
        for group in self.optimizer.param_groups:
            group["lr"] = self._lr
        self.optimizer.step()
        return self._lr

    def peak_lr(self) -> float:
        return noam_learning_rate(self.warmup_steps, self.d_model, self.warmup_steps, self.factor)

    def state_dict(self) -> dict:
        return {"step": self._step, "d_model": self.d_model,
                "warmup_steps": self.warmup_steps, "factor": self.factor}

    def load_state_dict(self, sd: dict) -> None:
        self._step = sd["step"]
        self.d_model = sd["d_model"]
        self.warmup_steps = sd["warmup_steps"]
        self.factor = sd.get("factor", 1.0)


def build_optimizer(model: torch.nn.Module, betas=(0.9, 0.98), eps: float = 1e-9) -> torch.optim.Adam:
    """Adam with the paper's beta_1 = 0.9, beta_2 = 0.98, epsilon = 1e-9.

    beta_2 = 0.98 (rather than the usual 0.999) shortens the second-moment
    averaging window, and epsilon = 1e-9 (rather than 1e-8) is smaller than
    the Adam default; both are stated in Section 5.3. The initial `lr` here
    is a placeholder -- NoamScheduler overwrites it before every step.
    """
    return torch.optim.Adam(model.parameters(), lr=0.0, betas=betas, eps=eps)


__all__ = ["noam_learning_rate", "NoamScheduler", "build_optimizer"]
