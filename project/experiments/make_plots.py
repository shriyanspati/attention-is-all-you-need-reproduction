"""Generate the Phase 6 figures from the JSONL training logs.

Figures produced:
  fig1_loss.png        training + validation loss vs steps
  fig2_bleu.png        validation BLEU vs steps
  fig3_lr_schedule.png equation (3), for the paper's base config and for
                       this study's scaled config, on log-log axes
  fig4_diagnostics.png gradient norm, throughput, memory
  fig5_ablations.png   ablation results with the measured seed-noise band
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..training.scheduler import noam_learning_rate

RUNS = Path("experiments/runs")
OUT = Path("experiments/results")
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})


def load(run: str) -> tuple[list[dict], dict]:
    p = RUNS / run / "log.jsonl"
    if not p.exists():
        return [], {}
    recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    s = RUNS / run / "summary.json"
    return recs, (json.loads(s.read_text()) if s.exists() else {})


def fig_loss(runs: list[str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for run in runs:
        recs, summ = load(run)
        if not recs:
            continue
        tr = [(r["step"], r["train_loss"]) for r in recs if "train_loss" in r]
        ev = [(r["step"], r["val_loss"]) for r in recs if r.get("kind") == "eval"]
        label = run
        if tr:
            axes[0].plot(*zip(*tr), lw=1.2, label=label)
        if ev:
            axes[1].plot(*zip(*ev), marker="o", ms=3, lw=1.2, label=label)
    axes[0].set_title("Training loss (label-smoothed, per token)")
    axes[1].set_title("Validation loss")
    for ax in axes:
        ax.set_xlabel("training step")
        ax.set_ylabel("loss")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_loss.png")
    plt.close(fig)


def fig_bleu(runs: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for run in runs:
        recs, _ = load(run)
        pts = [(r["step"], r["bleu"]) for r in recs
               if r.get("kind") == "eval" and r.get("bleu") is not None]
        if pts:
            ax.plot(*zip(*pts), marker="o", ms=3.5, lw=1.3, label=run)
    # NOTE: deliberately NO 27.3 reference line. The paper's 27.3 is WMT14
    # newstest2014 with beam-4 + checkpoint averaging; this axis is Multi30k
    # greedy on a validation subset. Drawing them on shared axes would imply a
    # comparability the study explicitly denies (see REPORT.md 6.5).
    ax.set_xlabel("training step")
    ax.set_ylabel("BLEU (greedy, validation subset)")
    ax.set_title("Validation BLEU vs steps (Multi30k, greedy, 150-sent subset)\n"
                 "not comparable to the paper's WMT14 beam-4 BLEU")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_bleu.png")
    plt.close(fig)


def fig_lr() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    steps = list(range(1, 100_001))
    lrs = [noam_learning_rate(s, 512, 4000) for s in steps]
    axes[0].plot(steps, lrs, lw=1.3, label="base: $d_{model}$=512, warmup=4000")
    lrs_big = [noam_learning_rate(s, 1024, 4000) for s in steps]
    axes[0].plot(steps, lrs_big, lw=1.3, ls="--",
                 label="big: $d_{model}$=1024, warmup=4000")
    peak = noam_learning_rate(4000, 512, 4000)
    axes[0].axvline(4000, c="gray", lw=0.8, ls=":")
    axes[0].annotate(f"peak {peak:.2e}\nat step 4000", (4000, peak),
                     textcoords="offset points", xytext=(28, -6), fontsize=7)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_title("Paper schedule, eq. (3)")

    # This study's scaled configuration.
    steps2 = list(range(1, 4001))
    axes[1].plot(steps2, [noam_learning_rate(s, 128, 160) for s in steps2], lw=1.3,
                 label="this study: $d_{model}$=128, warmup=160")
    axes[1].plot(steps2, [noam_learning_rate(s, 512, 4000) for s in steps2], lw=1.3,
                 ls="--", label="paper values over same span")
    axes[1].plot(steps2, [noam_learning_rate(s, 128, 1) for s in steps2], lw=1.0,
                 ls=":", c="crimson", label="warmup=1 ablation")
    axes[1].set_title("Scaled schedule (warmup held at 4% of budget)")
    for ax in axes:
        ax.set_xlabel("training step")
        ax.set_ylabel("learning rate")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_lr_schedule.png")
    plt.close(fig)


def fig_diagnostics(run: str) -> None:
    recs, _ = load(run)
    if not recs:
        return
    logs = [r for r in recs if "grad_norm" in r]
    if not logs:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    steps = [r["step"] for r in logs]
    axes[0].plot(steps, [r["grad_norm"] for r in logs], lw=1)
    axes[0].set_ylabel("global grad L2 norm"); axes[0].set_yscale("log")
    axes[0].set_title("Gradient norm")
    axes[1].plot(steps, [r["tokens_per_sec"] for r in logs], lw=1)
    axes[1].set_ylabel("target tokens / s"); axes[1].set_title("Throughput")
    axes[2].plot(steps, [r["rss_mb"] for r in logs], lw=1)
    axes[2].set_ylabel("resident memory (MiB)")
    axes[2].set_title("Process memory (no GPU available)")
    for ax in axes:
        ax.set_xlabel("training step")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_diagnostics.png")
    plt.close(fig)


def fig_ablations() -> None:
    summaries = {}
    for d in sorted(RUNS.glob("abl_*")):
        s = d / "summary.json"
        if s.exists():
            summaries[d.name] = json.loads(s.read_text())
    if not summaries:
        return

    seeds = [v for k, v in summaries.items() if k.startswith("abl_baseline_seed")]
    base_bleu = [v.get("best_val_bleu_greedy_subset") or 0 for v in seeds]
    base_loss = [v.get("best_val_loss") or 0 for v in seeds]
    import statistics as _st
    mean_b = sum(base_bleu) / len(base_bleu) if base_bleu else 0
    noise_loss_sd = _st.stdev(base_loss) if len(base_loss) >= 2 else None
    spread_b = (max(base_bleu) - min(base_bleu)) if base_bleu else 0

    groups = [(k, v) for k, v in summaries.items() if not k.startswith("abl_baseline_seed")]
    labels = ["baseline\n(mean of seeds)"] + [k.replace("abl_", "") for k, _ in groups]
    bleus = [mean_b] + [v.get("best_val_bleu_greedy_subset") or 0 for _, v in groups]
    losses = [sum(base_loss) / len(base_loss) if base_loss else 0] + \
             [v.get("best_val_loss") or 0 for _, v in groups]

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6), sharex=True)
    # Primary metric first: validation loss (low variance). BLEU on a
    # 60-sentence subset has a measured seed spread of 2.8 and is secondary.
    axes[0].bar(range(len(losses)), losses, color="indianred")
    lo, hi = min(losses), max(losses)
    axes[0].set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))   # zoom: bars from 0 hide the effect
    axes[0].set_ylabel("best val loss (lower=better)")
    if noise_loss_sd:
        axes[0].axhspan(losses[0] - noise_loss_sd, losses[0] + noise_loss_sd,
                        color="gray", alpha=0.3, label=f"baseline +/-1 sd of {len(base_loss)} seeds")
        axes[0].legend(fontsize=7)
    axes[1].bar(range(len(bleus)), bleus, color="steelblue")
    if spread_b:
        axes[1].axhspan(min(base_bleu), max(base_bleu), color="gray", alpha=0.25,
                        label=f"baseline seed range ({spread_b:.2f} BLEU, n={len(base_bleu)})")
        axes[1].legend(fontsize=7)
    axes[0].set_ylabel("best val BLEU (greedy)")
    steps_used = sorted({v.get("steps") for v in summaries.values() if v.get("steps")})
    step_txt = f"{steps_used[0]} steps" if len(steps_used) == 1 else f"{min(steps_used)}-{max(steps_used)} steps"
    axes[0].set_title(f"Ablations, scaled architecture, {step_txt}, Multi30k EN-DE (n=1 per arm)")
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_ablations.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    present = [d.name for d in sorted(RUNS.iterdir()) if d.is_dir()] if RUNS.exists() else []
    main_runs = [r for r in present if r in ("main_scaled", "paper_exact_arch")]
    fig_loss(main_runs or present[:2])
    fig_bleu(main_runs or present[:2])
    fig_lr()
    if "main_scaled" in present:
        fig_diagnostics("main_scaled")
    fig_ablations()
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
