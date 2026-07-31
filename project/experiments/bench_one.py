"""Time ONE config in an isolated process, so an OOM kill cannot destroy the
parent's results file. Prints a single JSON line to stdout."""
import argparse, json, sys
import torch
from ..model.transformer import TransformerConfig
from .benchmark import time_config, memory_requirement

ap = argparse.ArgumentParser()
ap.add_argument("--config", choices=["base", "big"], required=True)
ap.add_argument("--vocab", type=int, default=37000)
ap.add_argument("--repeats", type=int, default=5)
a = ap.parse_args()
cfg = (TransformerConfig.base if a.config == "base" else TransformerConfig.big)(
    src_vocab_size=a.vocab, tgt_vocab_size=a.vocab)
runs = [time_config(a.config, cfg, seq_len=25, micro_batch=8) for _ in range(a.repeats)]
sp = [r["sec_per_step_at_paper_batch"] for r in runs]
out = dict(runs[-1])
out["repeats"] = a.repeats
out["sec_per_step_runs"] = sp
out["sec_per_step_mean"] = sum(sp) / len(sp)
out["sec_per_step_min"] = min(sp)
out["sec_per_step_max"] = max(sp)
out["sec_per_step_spread_ratio"] = max(sp) / min(sp)
print("RESULT " + json.dumps(out))
