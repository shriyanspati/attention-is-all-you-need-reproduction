"""Experiment driver for Level 2 (scaled reproduction) and Phase 8 ablations.

Two presets exist for a deliberate reason:

  --preset paper-exact
      The paper's base architecture verbatim (N=6, d_model=512, d_ff=2048,
      h=8). Satisfies Level 2's "maintain same architecture" requirement.
      On one CPU core this costs ~5 s per 1024-token step, so only a few
      hundred steps are affordable: the run demonstrates that the exact
      architecture trains and its loss descends, but it cannot converge.

  --preset scaled
      A width/depth-reduced model (N=2, d_model=128, d_ff=512, h=8) that
      reaches a usable BLEU within minutes and therefore makes the Phase 8
      ablation grid affordable. Optimizer, schedule, loss, regularization,
      tokenization and batching are unchanged from the paper.

Everything that differs from the paper is a command-line flag with the
paper value as its default, so any deviation appears in the saved config.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from ..data.dataset import DataLoader, ParallelCorpus, TokenBatchSampler
from ..data.preprocessing import BPETokenizer, PAD_ID
from ..evaluation.bleu import corpus_bleu, sacrebleu_score
from ..model.transformer import Transformer, TransformerConfig
from ..training.trainer import TrainConfig, Trainer, average_checkpoints

PRESETS = {
    "paper-exact": dict(num_encoder_layers=6, num_decoder_layers=6, d_model=512,
                        d_ff=2048, num_heads=8),
    "scaled": dict(num_encoder_layers=2, num_decoder_layers=2, d_model=128,
                   d_ff=512, num_heads=8),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)   # CPU kernels here are deterministic


@torch.no_grad()
def translate_corpus(
    model: Transformer,
    corpus: ParallelCorpus,
    tok: BPETokenizer,
    batch_size: int = 32,
    beam_size: int = 1,
    max_extra: int = 50,
    alpha: float = 0.6,
    limit: int | None = None,
) -> list[str]:
    """Decode a corpus to detokenized strings.

    beam_size == 1 uses batched greedy decoding (fast, used for the BLEU
    curve during training). beam_size > 1 uses the paper's beam search, one
    sentence at a time (Section 6.1), used for final evaluation.
    """
    model.eval()
    examples = corpus.examples[:limit] if limit else corpus.examples
    hyps: list[str] = []

    if beam_size > 1:
        for ex in examples:
            src = torch.tensor(ex.src, dtype=torch.long).unsqueeze(0)
            out = model.beam_search(src, beam_size=beam_size,
                                    length_penalty_alpha=alpha, max_extra=max_extra)
            hyps.append(tok.decode(out[0].tolist()))
        return hyps

    # Sort by length for efficient batching, then restore original order.
    order = sorted(range(len(examples)), key=lambda i: len(examples[i].src))
    results: dict[int, str] = {}
    for s in range(0, len(order), batch_size):
        chunk = order[s : s + batch_size]
        L = max(len(examples[i].src) for i in chunk)
        src = torch.full((len(chunk), L), PAD_ID, dtype=torch.long)
        for r, i in enumerate(chunk):
            src[r, : len(examples[i].src)] = torch.tensor(examples[i].src)
        out = model.greedy_decode(src, max_extra=max_extra)
        for r, i in enumerate(chunk):
            results[i] = tok.decode(out[r].tolist())
    return [results[i] for i in range(len(examples))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default="scaled")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out-root", default="experiments/runs")
    ap.add_argument("--data", default="data/prepared")
    ap.add_argument("--raw", default="data/raw")
    # Paper hyperparameters (defaults = paper values)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--heads", type=int, default=None, help="override h (Table 3 row A)")
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--d-ff", type=int, default=None)
    ap.add_argument("--pos-enc", choices=["sinusoidal", "learned", "none"], default="sinusoidal")
    ap.add_argument("--norm-first", action="store_true", help="pre-norm (tensor2tensor) variant")
    ap.add_argument("--warmup", type=int, default=None,
                    help="default: 4%% of total steps, matching 4000/100000")
    ap.add_argument("--clip-norm", type=float, default=None)
    # Compute budget
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--bleu-sentences", type=int, default=200)
    ap.add_argument("--bleu-max-extra", type=int, default=15)
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--final-beam", type=int, default=4)
    ap.add_argument("--final-test", action="store_true", help="full test-set beam eval")
    ap.add_argument("--save-every", type=int, default=0,
                    help="periodic checkpoint interval; needed for checkpoint averaging")
    ap.add_argument("--beam-limit", type=int, default=None,
                    help="cap sentences for beam decoding (sequential, expensive)")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    set_seed(args.seed)
    tok = BPETokenizer.load(Path(args.data) / "bpe.json")
    raw = Path(args.raw)

    train = ParallelCorpus.from_files(raw / "train.en", raw / "train.de", tok,
                                      max_len=64, limit=args.train_limit)
    val = ParallelCorpus.from_files(raw / "val.en", raw / "val.de", tok, max_len=10_000)
    test = ParallelCorpus.from_files(raw / "test_2016_flickr.en",
                                     raw / "test_2016_flickr.de", tok, max_len=10_000)
    val_refs_full = (raw / "val.de").read_text(encoding="utf-8").splitlines()

    arch = dict(PRESETS[args.preset])
    if args.heads is not None:
        arch["num_heads"] = args.heads
    if args.d_model is not None:
        arch["d_model"] = args.d_model
    if args.d_ff is not None:
        arch["d_ff"] = args.d_ff

    mcfg = TransformerConfig(
        src_vocab_size=tok.vocab_size, tgt_vocab_size=tok.vocab_size,
        dropout=args.dropout, positional_encoding=args.pos_enc,
        norm_first=args.norm_first, tie_embeddings=True, max_len=128,
        pad_id=PAD_ID, bos_id=1, eos_id=2, **arch,
    )
    model = Transformer(mcfg)

    # Warmup scaled to preserve the paper's warmup/total ratio of 4%.
    warmup = args.warmup if args.warmup is not None else max(1, round(0.04 * args.steps))

    name = args.name or f"{args.preset}_s{args.seed}"
    out_dir = Path(args.out_root) / name
    tcfg = TrainConfig(
        total_steps=args.steps, warmup_steps=warmup, max_tokens=args.max_tokens,
        label_smoothing=args.label_smoothing, clip_norm=args.clip_norm,
        log_every=args.log_every, eval_every=args.eval_every, accum_steps=args.accum,
        save_every=args.save_every,
        seed=args.seed, out_dir=str(out_dir), notes=args.notes,
    )

    train_loader = DataLoader(train, TokenBatchSampler(train, max_tokens=args.max_tokens,
                                                       seed=args.seed))
    val_loader = DataLoader(val, TokenBatchSampler(val, max_tokens=args.max_tokens,
                                                   shuffle=False, seed=0))

    n_bleu = args.bleu_sentences

    def bleu_fn(m: Transformer, step: int) -> dict:
        hyps = translate_corpus(m, val, tok, beam_size=1,
                               max_extra=args.bleu_max_extra, limit=n_bleu)
        m.train()
        r = corpus_bleu(hyps, val_refs_full[:n_bleu])
        return {"bleu": r["bleu"], "bleu_bp": r["bp"], "bleu_ratio": r["ratio"]}

    print(f"=== {name} ===", flush=True)
    print(f"params {model.count_parameters() / 1e6:.3f}M | vocab {tok.vocab_size} | "
          f"train {len(train)} pairs | steps {args.steps} | warmup {warmup} | "
          f"max_tokens {args.max_tokens}", flush=True)

    trainer = Trainer(model, train_loader, val_loader, tcfg, bleu_fn=bleu_fn)
    t0 = time.time()
    history = trainer.train()
    train_seconds = time.time() - t0

    # ---- final evaluation ---------------------------------------------------
    summary = {
        "name": name, "preset": args.preset,
        "parameters": model.count_parameters(),
        "parameters_M": round(model.count_parameters() / 1e6, 3),
        "vocab_size": tok.vocab_size,
        "steps": args.steps, "warmup_steps": warmup,
        "max_tokens": args.max_tokens, "accum_steps": args.accum,
        "seed": args.seed, "train_seconds": round(train_seconds, 1),
        "train_hours": round(train_seconds / 3600, 3),
        "arch": arch, "dropout": args.dropout,
        "label_smoothing": args.label_smoothing,
        "positional_encoding": args.pos_enc, "norm_first": args.norm_first,
        "clip_norm": args.clip_norm, "notes": args.notes,
    }
    evals = [h for h in history if h.get("kind") == "eval"]
    if evals:
        summary["final_val_loss"] = evals[-1]["val_loss"]
        summary["final_val_ppl"] = evals[-1]["val_ppl"]
        summary["best_val_loss"] = min(e["val_loss"] for e in evals)
        summary["final_val_bleu_greedy_subset"] = evals[-1].get("bleu")
        summary["best_val_bleu_greedy_subset"] = max(
            (e.get("bleu", 0.0) for e in evals), default=None)
    logs = [h for h in history if "tokens_per_sec" in h]
    if logs:
        mid = logs[len(logs) // 4 :]
        summary["mean_tokens_per_sec"] = round(sum(l["tokens_per_sec"] for l in mid) / len(mid), 1)
        summary["mean_sec_per_step"] = round(
            args.max_tokens * args.accum / summary["mean_tokens_per_sec"], 4)
        summary["peak_rss_mb"] = round(max(l["rss_mb"] for l in logs), 1)
        summary["mean_grad_norm"] = round(sum(l["grad_norm"] for l in mid) / len(mid), 3)
        summary["max_grad_norm"] = round(max(l["grad_norm"] for l in logs), 3)

    if args.final_test:
        # Checkpoint averaging (Section 6.1) over available checkpoints.
        cks = sorted(Path(out_dir).glob("ckpt_*.pt"))
        if len(cks) >= 2:
            avg_model = Transformer(mcfg)
            average_checkpoints(cks[-5:], avg_model)
        else:
            avg_model = model   # averaging not exercised; recorded in summary

        test_refs = (raw / "test_2016_flickr.de").read_text(encoding="utf-8").splitlines()
        eval_specs = [("greedy", model, 1, None),
                      ("beam4", model, args.final_beam, args.beam_limit)]
        if avg_model is not model:
            # Paper procedure: beam-4 over the averaged checkpoint (Section 6.1).
            eval_specs.append(("beam4_avg", avg_model, args.final_beam, args.beam_limit))
        summary["checkpoints_averaged"] = len(cks[-5:]) if avg_model is not model else 0
        for tag, mdl, beam, lim in eval_specs:
            t1 = time.time()
            hyps = translate_corpus(mdl, test, tok, beam_size=beam, max_extra=50,
                                    alpha=0.6, limit=lim)
            r = corpus_bleu(hyps, test_refs[: len(hyps)])
            summary[f"test_bleu_{tag}_sentences"] = len(hyps)
            sb = sacrebleu_score(hyps, test_refs[: len(hyps)])
            summary[f"test_bleu_{tag}"] = round(r["bleu"], 3)
            summary[f"test_bleu_{tag}_sacrebleu"] = round(sb, 3) if sb else None
            summary[f"test_bleu_{tag}_bp"] = round(r["bp"], 4)
            summary[f"test_bleu_{tag}_ratio"] = round(r["ratio"], 4)
            summary[f"test_bleu_{tag}_precisions"] = [round(p, 5) for p in r["precisions"]]
            summary[f"test_decode_seconds_{tag}"] = round(time.time() - t1, 1)
            print(f"TEST {tag}: {r}", flush=True)
            (out_dir / f"hyps_test_{tag}.txt").write_text("\n".join(hyps), encoding="utf-8")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
