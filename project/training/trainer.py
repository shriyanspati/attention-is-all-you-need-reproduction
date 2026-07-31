"""Training loop with the monitoring required by Phase 6.

Tracked every `log_every` steps and appended to a JSONL log:
  * training loss (label-smoothed objective, per target token)
  * learning rate from equation (3)
  * global gradient L2 norm, measured BEFORE any clipping
  * tokens processed per second (target tokens, wall-clock)
  * process resident memory (this container has no GPU, so
    torch.cuda.max_memory_allocated is unavailable; RSS is the honest
    substitute and is labelled as such)

Tracked every `eval_every` steps:
  * validation label-smoothed loss and unsmoothed per-token cross-entropy
  * validation perplexity = exp(unsmoothed CE), per BPE token, which is the
    quantity comparable to the "PPL (dev)" column of Table 3
  * optionally BLEU on a fixed validation subset via greedy decoding

Gradient clipping is NOT part of the paper and is off by default. It is
available (`clip_norm`) because the post-norm architecture the paper
describes can spike early in training; any run that used it is flagged in
its config so the deviation is never silent.

Checkpoint averaging (Section 6.1: "a single model obtained by averaging
the last 5 checkpoints") is implemented in `average_checkpoints`.
"""

from __future__ import annotations

import copy
import json
import math
import os
import resource
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterator, List, Optional

import torch
import torch.nn as nn

from ..data.dataset import DataLoader, infinite_loader
from ..model.transformer import Transformer, TransformerConfig
from .loss import LabelSmoothingLoss, token_cross_entropy
from .scheduler import NoamScheduler, build_optimizer


def rss_mb() -> float:
    """Resident set size in MiB (ru_maxrss is KiB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


@dataclass
class TrainConfig:
    total_steps: int = 100_000
    warmup_steps: int = 4000
    max_tokens: int = 25_000
    label_smoothing: float = 0.1
    betas: tuple = (0.9, 0.98)
    eps: float = 1e-9
    lr_factor: float = 1.0
    clip_norm: Optional[float] = None      # not in paper; None = disabled
    log_every: int = 25
    eval_every: int = 250
    save_every: int = 0                    # 0 = only at the end
    keep_last: int = 5                     # for checkpoint averaging
    seed: int = 1337
    accum_steps: int = 1                   # gradient accumulation to emulate
                                           # the paper's 25k-token batch
    out_dir: str = "experiments/run"
    notes: str = ""
    resume: bool = True                    # pick up from last.pt if present

    def to_dict(self) -> dict:
        d = asdict(self)
        d["betas"] = list(self.betas)
        return d


class Trainer:
    def __init__(
        self,
        model: Transformer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: TrainConfig,
        bleu_fn: Optional[Callable[[Transformer, int], dict]] = None,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.bleu_fn = bleu_fn

        torch.manual_seed(cfg.seed)

        self.criterion = LabelSmoothingLoss(
            vocab_size=model.config.tgt_vocab_size,
            pad_id=model.config.pad_id,
            smoothing=cfg.label_smoothing,
        )
        self.optimizer = build_optimizer(model, betas=cfg.betas, eps=cfg.eps)
        self.scheduler = NoamScheduler(
            self.optimizer, d_model=model.config.d_model,
            warmup_steps=cfg.warmup_steps, factor=cfg.lr_factor,
        )

        self.out = Path(cfg.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out / "log.jsonl"
        self.history: List[dict] = []
        self.start_step = 0
        if cfg.resume:
            self._maybe_resume()
        self._save_configs()

    def _maybe_resume(self) -> None:
        """Resume from last.pt so a long run can span several sessions.

        Caveat recorded honestly: the training data stream is restarted at the
        epoch implied by the step count rather than at the exact sample
        offset, so a resumed run does not see byte-identical batch ordering
        to an uninterrupted one. Model, optimizer (both Adam moments) and
        scheduler step are restored exactly, so the optimization trajectory
        is continuous; only the data permutation differs.
        """
        ck = self.out / "last.pt"
        if not ck.exists():
            return
        sd = torch.load(ck, map_location=self.device, weights_only=False)
        self.model.load_state_dict(sd["model"])
        self.optimizer.load_state_dict(sd["optimizer"])
        self.scheduler.load_state_dict(sd["scheduler"])
        self.start_step = int(sd["step"])
        if self.log_path.exists():
            for line in self.log_path.read_text().splitlines():
                if line.strip():
                    self.history.append(json.loads(line))
        print(f"resumed from {ck} at step {self.start_step}", flush=True)

    # ------------------------------------------------------------------ setup

    def _save_configs(self) -> None:
        (self.out / "config.json").write_text(json.dumps({
            "model": self.model.config.to_dict(),
            "train": self.cfg.to_dict(),
            "environment": {
                "torch": torch.__version__,
                "device": self.device,
                "num_threads": torch.get_num_threads(),
                "cuda_available": torch.cuda.is_available(),
            },
            "parameters": self.model.parameter_breakdown(),
            "peak_lr": self.scheduler.peak_lr(),
        }, indent=2))

    def _log(self, record: dict) -> None:
        self.history.append(record)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------- evaluation

    @torch.no_grad()
    def evaluate(self, max_batches: Optional[int] = None) -> dict:
        self.model.eval()
        tot_loss = 0.0
        tot_ce = 0.0
        tot_tok = 0
        nb = 0
        for batch in self.val_loader:
            src = batch["src"].to(self.device)
            tgt_in = batch["tgt_in"].to(self.device)
            tgt_out = batch["tgt_out"].to(self.device)
            logits = self.model(src, tgt_in)
            loss = self.criterion(logits, tgt_out)
            ce_sum, n = token_cross_entropy(logits, tgt_out, self.model.config.pad_id)
            tot_loss += float(loss) * n
            tot_ce += ce_sum
            tot_tok += n
            nb += 1
            if max_batches is not None and nb >= max_batches:
                break
        self.model.train()
        if tot_tok == 0:
            return {"val_loss": float("nan"), "val_ce": float("nan"), "val_ppl": float("nan")}
        ce = tot_ce / tot_tok
        return {
            "val_loss": tot_loss / tot_tok,
            "val_ce": ce,
            "val_ppl": math.exp(min(ce, 20.0)),
            "val_tokens": tot_tok,
        }

    def grad_norm(self) -> float:
        total = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total += float(p.grad.detach().pow(2).sum())
        return math.sqrt(total)

    # --------------------------------------------------------------- training

    def train(self) -> List[dict]:
        cfg = self.cfg
        self.model.train()
        n_batches = max(1, len(self.train_loader))
        stream: Iterator[dict] = infinite_loader(
            self.train_loader, start_epoch=self.start_step // n_batches)
        if self.start_step >= cfg.total_steps:
            print(f"already at step {self.start_step} >= {cfg.total_steps}; nothing to do",
                  flush=True)
            return self.history

        t0 = time.time()
        window_tokens = 0
        window_loss = 0.0
        window_tok_for_loss = 0
        window_t = time.time()
        best_val = float("inf")

        for step in range(self.start_step + 1, cfg.total_steps + 1):
            self.optimizer.zero_grad(set_to_none=True)
            step_tokens = 0
            step_loss_sum = 0.0

            # Gradient accumulation: `accum_steps` micro-batches form one
            # optimizer step, which is how a 25k-token batch is emulated on
            # hardware that cannot hold it in memory at once.
            for _ in range(cfg.accum_steps):
                batch = next(stream)
                src = batch["src"].to(self.device)
                tgt_in = batch["tgt_in"].to(self.device)
                tgt_out = batch["tgt_out"].to(self.device)
                logits = self.model(src, tgt_in)
                loss = self.criterion(logits, tgt_out)
                (loss / cfg.accum_steps).backward()
                n = batch["n_tokens"]
                step_tokens += n
                step_loss_sum += float(loss) * n

            gnorm = self.grad_norm()
            if cfg.clip_norm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.clip_norm)
            lr = self.scheduler.step()

            window_tokens += step_tokens
            window_loss += step_loss_sum
            window_tok_for_loss += step_tokens

            if step % cfg.log_every == 0 or step == 1:
                now = time.time()
                dt = max(now - window_t, 1e-9)
                rec = {
                    "step": step,
                    "train_loss": window_loss / max(window_tok_for_loss, 1),
                    "lr": lr,
                    "grad_norm": gnorm,
                    "tokens_per_sec": window_tokens / dt,
                    "tokens_seen": None,
                    "rss_mb": rss_mb(),
                    "elapsed_s": now - t0,
                    "sentences": batch["n_sentences"],
                    "src_len": int(src.size(1)),
                    "tgt_len": int(tgt_in.size(1)),
                }
                self._log(rec)
                print(f"step {step:6d} | loss {rec['train_loss']:.4f} | lr {lr:.2e} "
                      f"| gnorm {gnorm:7.2f} | tok/s {rec['tokens_per_sec']:7.0f} "
                      f"| rss {rec['rss_mb']:.0f}MB", flush=True)
                window_tokens = 0
                window_loss = 0.0
                window_tok_for_loss = 0
                window_t = now

            if cfg.eval_every and (step % cfg.eval_every == 0 or step == cfg.total_steps):
                ev = self.evaluate()
                ev["step"] = step
                ev["kind"] = "eval"
                if self.bleu_fn is not None:
                    ev.update(self.bleu_fn(self.model, step))
                self._log(ev)
                msg = (f"  eval @ {step}: val_loss {ev['val_loss']:.4f} "
                       f"ppl {ev['val_ppl']:.2f}")
                if "bleu" in ev:
                    msg += f" bleu {ev['bleu']:.2f}"
                print(msg, flush=True)
                if ev["val_loss"] < best_val:
                    best_val = ev["val_loss"]
                    self.save_checkpoint("best.pt", step)
                self.save_checkpoint("last.pt", step)

            if cfg.save_every and step % cfg.save_every == 0:
                self.save_checkpoint(f"ckpt_{step:06d}.pt", step)
                self._prune_checkpoints()

        self.save_checkpoint("final.pt", cfg.total_steps)
        self.save_checkpoint("last.pt", cfg.total_steps)
        (self.out / "history.json").write_text(json.dumps(self.history, indent=2))
        return self.history

    # ------------------------------------------------------------ checkpoints

    def save_checkpoint(self, name: str, step: int) -> Path:
        path = self.out / name
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "model_config": self.model.config.to_dict(),
            "train_config": self.cfg.to_dict(),
            "step": step,
        }, path)
        return path

    def _prune_checkpoints(self) -> None:
        cks = sorted(self.out.glob("ckpt_*.pt"))
        for p in cks[: max(0, len(cks) - self.cfg.keep_last)]:
            p.unlink()


def average_checkpoints(paths: List[str | Path], model: Transformer) -> Transformer:
    """Section 6.1 checkpoint averaging: uniform mean of the parameters.

    Averaging weights across nearby SGD iterates approximates a flatter
    solution and is worth ~0.3-0.6 BLEU in the original setup. It is applied
    to the *parameters*, not the predictions.
    """
    if not paths:
        raise ValueError("no checkpoints to average")
    avg: Optional[dict] = None
    for p in paths:
        sd = torch.load(p, map_location="cpu", weights_only=False)["model"]
        if avg is None:
            avg = {k: v.clone().float() for k, v in sd.items()}
        else:
            for k in avg:
                avg[k] += sd[k].float()
    for k in avg:
        avg[k] /= len(paths)
    model.load_state_dict({k: v.to(dtype=torch.float32) for k, v in avg.items()})
    return model


__all__ = ["Trainer", "TrainConfig", "average_checkpoints", "rss_mb"]
