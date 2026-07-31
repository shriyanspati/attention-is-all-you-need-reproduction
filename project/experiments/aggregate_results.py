"""Aggregate all run summaries into the Phase 7 / Phase 8 comparison tables.

Writes experiments/results/results.json and results.md. Deliberately reports
the seed spread alongside every ablation delta: with a single seed at this
scale, a 1-BLEU difference is not evidence of anything, and a table that
omits the noise floor invites over-reading.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

RUNS = Path("experiments/runs")
OUT = Path("experiments/results")

PAPER = {
    "base": {"bleu_ende_newstest2014": 27.3, "params_M": 65, "steps": 100_000,
             "hardware": "8x NVIDIA P100", "train_time": "12 hours",
             "sec_per_step": 0.4, "dev_ppl": 4.92, "dev_bleu_newstest2013": 25.8},
    "big": {"bleu_ende_newstest2014": 28.4, "params_M": 213, "steps": 300_000,
            "hardware": "8x NVIDIA P100", "train_time": "3.5 days",
            "sec_per_step": 1.0, "dev_ppl": 4.33, "dev_bleu_newstest2013": 26.4},
}


def load_all() -> dict[str, dict]:
    out = {}
    if not RUNS.exists():
        return out
    for d in sorted(RUNS.iterdir()):
        s = d / "summary.json"
        if s.exists():
            out[d.name] = json.loads(s.read_text())
    return out


def fmt(v, nd=2, dash="--"):
    if v is None:
        return dash
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_all()
    bench_path = OUT / "benchmark.json"
    bench = json.loads(bench_path.read_text()) if bench_path.exists() else {}
    data_report_path = Path("data/prepared/data_report.json")
    data_report = json.loads(data_report_path.read_text()) if data_report_path.exists() else {}

    lines: list[str] = ["# Reproduction results", ""]

    # ---------------- headline comparison (Phase 7 table) -------------------
    main_run = runs.get("main_scaled", {})
    exact_run = runs.get("paper_exact_arch", {})
    lines += ["## Headline comparison", "",
              "| Metric | Paper (base) | This study (Level 2b, scaled) | "
              "This study (Level 2a, paper-exact arch) |",
              "|---|---|---|---|"]
    lines.append(f"| Dataset | WMT14 EN-DE (4.5M pairs) | Multi30k EN-DE (29k pairs) "
                 f"| Multi30k EN-DE (29k pairs) |")
    lines.append(f"| Test set | newstest2014 | test_2016_flickr | test_2016_flickr |")
    lines.append(f"| BLEU | 27.3 | {fmt(main_run.get('test_bleu_beam4'))} "
                 f"| {fmt(exact_run.get('best_val_bleu_greedy_subset'))} (val subset) |")
    lines.append(f"| Vocabulary | ~37,000 BPE (shared) | "
                 f"{fmt(main_run.get('vocab_size'), 0)} BPE (shared) | "
                 f"{fmt(exact_run.get('vocab_size'), 0)} BPE (shared) |")
    lines.append(f"| Parameters | 65M (reported) | {fmt(main_run.get('parameters_M'), 3)}M "
                 f"| {fmt(exact_run.get('parameters_M'), 3)}M |")
    lines.append(f"| Training steps | 100,000 | {fmt(main_run.get('steps'), 0)} "
                 f"| {fmt(exact_run.get('steps'), 0)} |")
    lines.append(f"| Batch size | ~25,000 src + 25,000 tgt tokens | "
                 f"{fmt(main_run.get('max_tokens'), 0)} tokens | "
                 f"{fmt(exact_run.get('max_tokens'), 0)} tokens |")
    lines.append(f"| Training time | 12 h | {fmt(main_run.get('train_hours'), 2)} h "
                 f"| {fmt(exact_run.get('train_hours'), 2)} h |")
    lines.append(f"| Hardware | 8x NVIDIA P100 | 1x Xeon vCPU (no GPU) "
                 f"| 1x Xeon vCPU (no GPU) |")
    lines.append(f"| s / step | 0.4 | {fmt(main_run.get('mean_sec_per_step'), 3)} "
                 f"| {fmt(exact_run.get('mean_sec_per_step'), 3)} |")
    lines.append(f"| Val perplexity (per BPE token) | 4.92 (newstest2013) | "
                 f"{fmt(main_run.get('final_val_ppl'))} | "
                 f"{fmt(exact_run.get('final_val_ppl'))} |")
    lines.append("")

    # ---------------- compute feasibility ----------------------------------
    if bench.get("results"):
        lines += ["## Measured compute gap (paper configurations on this hardware)", "",
                  "| Config | Params | s/step @ 25k tokens | Paper s/step | Slowdown | "
                  "Projected time for paper budget |", "|---|---|---|---|---|---|"]
        for r in bench["results"]:
            if r.get("status") == "out_of_memory":
                lines.append(f"| {r['config']} | {fmt(r.get('parameters_M'), 1)}M | "
                             f"OOM | 1.0 | n/a | infeasible: needs "
                             f"{fmt(r.get('optimizer_state_gb'))} GB for Adam state alone |")
            else:
                lines.append(
                    f"| {r['config']} | {fmt(r.get('parameters_M'), 1)}M | "
                    f"{fmt(r.get('sec_per_step_at_paper_batch'), 1)} | "
                    f"{fmt(r.get('paper_sec_per_step'), 1)} | "
                    f"{fmt(r.get('slowdown_vs_paper'), 0)}x | "
                    f"{fmt(r.get('projected_days_for_paper_budget'), 1)} days "
                    f"({fmt(r.get('projected_years_for_paper_budget'))} yr) |")
        lines.append("")

    # ---------------- ablations --------------------------------------------
    seed_runs = {k: v for k, v in runs.items() if k.startswith("abl_baseline_seed")}
    abl_runs = {k: v for k, v in runs.items()
                if k.startswith("abl_") and not k.startswith("abl_baseline_seed")}

    noise_bleu = noise_loss = None
    if len(seed_runs) >= 2:
        b = [v.get("best_val_bleu_greedy_subset") for v in seed_runs.values()
             if v.get("best_val_bleu_greedy_subset") is not None]
        l = [v.get("best_val_loss") for v in seed_runs.values()
             if v.get("best_val_loss") is not None]
        if len(b) >= 2:
            noise_bleu = {"mean": statistics.mean(b), "sd": statistics.stdev(b),
                          "min": min(b), "max": max(b), "n": len(b)}
        if len(l) >= 2:
            noise_loss = {"mean": statistics.mean(l), "sd": statistics.stdev(l),
                          "min": min(l), "max": max(l), "n": len(l)}

    if seed_runs:
        lines += ["## Seed variance (the noise floor)", "",
                  "| Run | seed | best val loss | best val BLEU |", "|---|---|---|---|"]
        for k, v in sorted(seed_runs.items()):
            lines.append(f"| {k} | {v.get('seed')} | {fmt(v.get('best_val_loss'), 4)} "
                         f"| {fmt(v.get('best_val_bleu_greedy_subset'))} |")
        if noise_bleu:
            lines.append(f"| **mean +/- sd** | -- | "
                         f"{fmt(noise_loss['mean'], 4)} +/- {fmt(noise_loss['sd'], 4)} | "
                         f"{fmt(noise_bleu['mean'])} +/- {fmt(noise_bleu['sd'])} |")
        lines.append("")
        if noise_bleu:
            lines.append(f"Seed-to-seed range: **{noise_bleu['max'] - noise_bleu['min']:.2f} BLEU** "
                         f"({noise_bleu['min']:.2f}-{noise_bleu['max']:.2f}). Any ablation delta "
                         f"smaller than this is not distinguishable from noise with n=1.")
            lines.append("")

    if abl_runs:
        # Ablations all use seed 1337, so the PAIRED comparison against the
        # seed-1337 baseline is the valid one. Deltas against the 3-seed mean
        # are also shown because the choice changes conclusions (h=16).
        b1337 = runs.get("abl_baseline_seed1337", {})
        loss_b = b1337.get("best_val_loss")
        bleu_b = b1337.get("best_val_bleu_greedy_subset")
        sd = noise_loss["sd"] if noise_loss else None
        rng_l = (noise_loss["max"] - noise_loss["min"]) if noise_loss else None
        lines += ["## Ablations", "",
                  "All arms n=1, seed 1337. `d loss` is the PAIRED delta against the "
                  "seed-1337 baseline (negative = better). No significance test is "
                  "performed: with one run per arm the within-arm variance is unknown, "
                  "so the 3-seed baseline range is shown only as a reference scale.", "",
                  "| Variant | params | val loss | d loss (paired) | > baseline seed range? | "
                  "val BLEU (60 sents) | d loss vs 3-seed mean |",
                  "|---|---|---|---|---|---|---|"]
        for k, v in sorted(abl_runs.items()):
            l = v.get("best_val_loss"); bl = v.get("best_val_bleu_greedy_subset")
            dl = (l - loss_b) if (l is not None and loss_b is not None) else None
            dm = (l - noise_loss["mean"]) if (l is not None and noise_loss) else None
            outside = "--"
            if dl is not None and rng_l is not None:
                outside = "yes" if abs(dl) > rng_l else "no"
            lines.append(f"| {k.replace('abl_', '')} | {fmt(v.get('parameters_M'), 3)}M | "
                         f"{fmt(l, 4)} | {fmt(dl, 4)} | {outside} | {fmt(bl)} | {fmt(dm, 4)} |")
        lines.append("")

    # ---------------- data pipeline ----------------------------------------
    if data_report:
        t = data_report.get("tokenizer", {})
        c = data_report.get("corpus", {})
        lines += ["## Data pipeline", "",
                  f"- Corpus: {c.get('train_pairs')} train / {c.get('val_pairs')} val / "
                  f"{c.get('test_pairs')} test sentence pairs",
                  f"- Joint word types: {c.get('word_types_joint')} "
                  f"(EN {c.get('word_types_en')}, DE {c.get('word_types_de')})",
                  f"- Shared BPE vocabulary: {t.get('vocab_size')} tokens from "
                  f"{t.get('merges_learned')} merges",
                  f"- Held-out UNK rate: val {t.get('val_unk_rate')}, "
                  f"test {t.get('test_unk_rate')}", ""]
        if data_report.get("vocab_scaling_study"):
            lines += ["| merges | vocab size | mean tokens/sentence | val UNK rate |",
                      "|---|---|---|---|"]
            for r in data_report["vocab_scaling_study"]:
                lines.append(f"| {r['merges_requested']} | {r['vocab_size']} | "
                             f"{r['mean_val_en_tokens_per_sentence']} | {r['val_unk_rate']} |")
            lines.append("")
        if data_report.get("dynamic_batching"):
            lines += ["| token budget | batches/epoch | mean sentences/batch | padding fraction |",
                      "|---|---|---|---|"]
            for r in data_report["dynamic_batching"]:
                lines.append(f"| {r['max_tokens']} | {r['batches_per_epoch']} | "
                             f"{r['mean_sentences_per_batch']} | {r['padding_fraction']} |")
            lines.append("")

    # ---------------- all runs ---------------------------------------------
    lines += ["## All runs", "",
              "| run | params | steps | s/step | tok/s | peak RSS (MiB) | "
              "final val loss | final val ppl | best val BLEU |",
              "|---|---|---|---|---|---|---|---|---|"]
    for k, v in sorted(runs.items()):
        lines.append(f"| {k} | {fmt(v.get('parameters_M'), 3)}M | {fmt(v.get('steps'), 0)} | "
                     f"{fmt(v.get('mean_sec_per_step'), 3)} | "
                     f"{fmt(v.get('mean_tokens_per_sec'), 0)} | "
                     f"{fmt(v.get('peak_rss_mb'), 0)} | {fmt(v.get('final_val_loss'), 4)} | "
                     f"{fmt(v.get('final_val_ppl'))} | "
                     f"{fmt(v.get('best_val_bleu_greedy_subset'))} |")
    lines.append("")

    (OUT / "results.md").write_text("\n".join(lines))
    (OUT / "results.json").write_text(json.dumps({
        "paper": PAPER, "runs": runs, "benchmark": bench,
        "noise_floor": {"bleu": noise_bleu, "val_loss": noise_loss},
        "data_report": data_report,
    }, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
