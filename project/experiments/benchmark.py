"""Measure this machine's throughput on the paper's exact configurations.

Purpose: turn "we lack the hardware" into a quantitative statement. The
paper reports 0.4 s/step for base and 1.0 s/step for big on 8 NVIDIA P100
GPUs with ~25000 source + ~25000 target tokens per step. Measuring the same
quantity here yields an honest slowdown factor and a projected wall-clock
time for the full 100k / 300k step budgets, which is what Phase 2 Level 3
actually requires us to document.

We time forward + backward + optimizer step, which is what a training step
costs, and we do it at the paper's batch size in tokens (via gradient
accumulation, since a 25k-token batch does not fit in 3.9 GB of RAM at
d_model=512).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ..model.transformer import Transformer, TransformerConfig
from ..training.loss import LabelSmoothingLoss
from ..training.scheduler import NoamScheduler, build_optimizer
from ..training.trainer import rss_mb

# Paper Section 5.2 reference points.
PAPER = {
    "base": {"sec_per_step": 0.4, "steps": 100_000, "gpus": 8, "gpu": "P100"},
    "big": {"sec_per_step": 1.0, "steps": 300_000, "gpus": 8, "gpu": "P100"},
}
PAPER_TOKENS_PER_STEP = 25_000     # target tokens (Section 5.1)


def time_config(
    name: str,
    cfg: TransformerConfig,
    seq_len: int = 25,
    micro_batch: int = 8,
    n_steps: int = 3,
    tokens_per_step: int = PAPER_TOKENS_PER_STEP,
) -> dict:
    """Time full training steps at the paper's token batch size."""
    torch.manual_seed(0)
    model = Transformer(cfg)
    crit = LabelSmoothingLoss(cfg.tgt_vocab_size, cfg.pad_id, 0.1)
    opt = build_optimizer(model)
    sch = NoamScheduler(opt, cfg.d_model, 4000)
    model.train()

    tokens_per_micro = micro_batch * seq_len
    accum = max(1, round(tokens_per_step / tokens_per_micro))

    src = torch.randint(4, cfg.src_vocab_size, (micro_batch, seq_len))
    tgt_in = torch.randint(4, cfg.tgt_vocab_size, (micro_batch, seq_len))
    tgt_out = torch.randint(4, cfg.tgt_vocab_size, (micro_batch, seq_len))

    # One untimed step to pay any lazy-init / allocator warmup cost.
    opt.zero_grad(set_to_none=True)
    crit(model(src, tgt_in), tgt_out).backward()
    sch.step()

    t_micro0 = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    loss = crit(model(src, tgt_in), tgt_out)
    loss.backward()
    t_micro = time.perf_counter() - t_micro0

    t0 = time.perf_counter()
    for _ in range(n_steps):
        opt.zero_grad(set_to_none=True)
        for _ in range(min(accum, 4)):     # time a few micro-batches, then scale
            loss = crit(model(src, tgt_in), tgt_out)
            (loss / accum).backward()
        sch.step()
    measured = (time.perf_counter() - t0) / n_steps
    micros_timed = min(accum, 4)
    # Extrapolate to the full accumulation count.
    sec_per_step = measured * (accum / micros_timed)

    params = model.count_parameters()
    out = {
        "config": name,
        "d_model": cfg.d_model,
        "d_ff": cfg.d_ff,
        "num_heads": cfg.num_heads,
        "layers": cfg.num_encoder_layers,
        "parameters": params,
        "parameters_M": round(params / 1e6, 3),
        "micro_batch_sentences": micro_batch,
        "seq_len": seq_len,
        "tokens_per_micro_batch": tokens_per_micro,
        "accum_steps_for_paper_batch": accum,
        "sec_per_micro_batch": round(t_micro, 4),
        "sec_per_step_at_paper_batch": round(sec_per_step, 2),
        "target_tokens_per_sec": round(tokens_per_step / sec_per_step, 1),
        "rss_mb": round(rss_mb(), 1),
    }
    if name in PAPER:
        p = PAPER[name]
        out["paper_sec_per_step"] = p["sec_per_step"]
        out["paper_hardware"] = f"{p['gpus']}x {p['gpu']}"
        out["slowdown_vs_paper"] = round(sec_per_step / p["sec_per_step"], 1)
        out["paper_steps"] = p["steps"]
        days = sec_per_step * p["steps"] / 86400
        out["projected_days_for_paper_budget"] = round(days, 1)
        out["projected_years_for_paper_budget"] = round(days / 365.25, 2)
        out["paper_wallclock_hours"] = round(p["sec_per_step"] * p["steps"] / 3600, 1)
    return out


def memory_requirement(cfg: TransformerConfig, bytes_per_scalar: int = 4) -> dict:
    """Analytic training-memory floor for Adam, excluding activations.

    Adam keeps, per parameter: the parameter, its gradient, and two moment
    estimates (m and v) -> 4 tensors -> 16 bytes/parameter in fp32. This is a
    hard floor: no batch-size reduction or gradient accumulation can avoid it.
    """
    n = Transformer(cfg).count_parameters() if cfg.d_model <= 512 else None
    if n is None:
        # Count without materializing the model, to avoid the OOM we are
        # trying to describe.
        d, dff, h, N, V = cfg.d_model, cfg.d_ff, cfg.num_heads, cfg.num_encoder_layers, cfg.src_vocab_size
        attn = 4 * d * d
        ffn = 2 * d * dff + dff + d
        n = V * d + N * (attn + ffn + 4 * d) + N * (2 * attn + ffn + 6 * d)
    return {
        "parameters": n,
        "parameters_M": round(n / 1e6, 3),
        "bytes_per_param_adam_fp32": 16,
        "optimizer_state_gb": round(n * 16 / 1024 ** 3, 2),
        "note": ("params + grads + Adam m,v at fp32; excludes activations, "
                 "so this is a lower bound on peak RSS"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/results/benchmark.json")
    ap.add_argument("--vocab", type=int, default=37000)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=25)
    args = ap.parse_args()

    results = []
    for name, cfg in [
        ("base", TransformerConfig.base(src_vocab_size=args.vocab, tgt_vocab_size=args.vocab)),
        ("big", TransformerConfig.big(src_vocab_size=args.vocab, tgt_vocab_size=args.vocab)),
    ]:
        print(f"timing {name} ...", flush=True)
        try:
            r = time_config(name, cfg, seq_len=args.seq_len, micro_batch=args.micro_batch)
        except (RuntimeError, MemoryError) as e:
            # Recorded rather than swallowed: infeasibility IS a result.
            r = {"config": name, "status": "out_of_memory", "error": str(e)[:200],
                 **memory_requirement(cfg)}
        results.append(r)
        print(json.dumps(r, indent=2), flush=True)

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "environment": {
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
            "cuda": torch.cuda.is_available(),
        },
        "paper_reference": PAPER,
        "paper_tokens_per_step": PAPER_TOKENS_PER_STEP,
        "results": results,
    }, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
