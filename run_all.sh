#!/bin/bash
# Sequential experiment sequence, sized to complete in ~45 min of wall clock on
# one CPU core. Runs strictly sequentially: concurrent runs would divide the
# single core and distort the throughput measurements Phase 6 requires.
set -u
cd "$(dirname "$0")"
mkdir -p experiments/results
LOG=experiments/results/run_all.log
: > "$LOG"
say() { echo "=== $(date -u +%H:%M:%S) $* ===" >> "$LOG"; }
R="python3 -m project.experiments.run_experiment"

# --- Level 2b: main scaled reproduction --------------------------------------
say "MAIN scaled 1600 steps"
$R --preset scaled --steps 1600 --max-tokens 1024 --eval-every 400 --log-every 50 \
   --bleu-sentences 200 --seed 1337 --final-test --beam-limit 200 --name main_scaled \
   --notes "Level 2b main run: scaled architecture, full pipeline" >> "$LOG" 2>&1

# --- Level 2a: paper-exact architecture, truncated budget -------------------
say "PAPER-EXACT architecture 80 steps"
$R --preset paper-exact --steps 80 --max-tokens 1024 --eval-every 40 --log-every 10 \
   --bleu-sentences 60 --bleu-max-extra 12 --seed 1337 --name paper_exact_arch \
   --notes "Level 2a: paper base architecture verbatim, severely truncated budget" \
   >> "$LOG" 2>&1

# --- Phase 8: seed variance (the noise floor) -------------------------------
for S in 1337 7 42; do
  say "ABLATION baseline seed=$S"
  $R --preset scaled --steps 400 --max-tokens 1024 --eval-every 400 --log-every 200 \
     --bleu-sentences 150 --seed "$S" --name "abl_baseline_seed$S" \
     --notes "seed variance baseline" >> "$LOG" 2>&1
done

# --- Phase 8: Table 3 row (A), h varied at constant compute ------------------
for H in 1 4 16; do
  say "ABLATION heads=$H"
  $R --preset scaled --steps 400 --max-tokens 1024 --eval-every 400 --log-every 200 \
     --bleu-sentences 150 --seed 1337 --heads "$H" --name "abl_heads$H" \
     --notes "Table 3 row A: h=$H with d_k=d_v=d_model/h" >> "$LOG" 2>&1
done

# --- Phase 8: positional encoding (row E, and removal) ----------------------
for PE in none learned; do
  say "ABLATION pos-enc=$PE"
  $R --preset scaled --steps 400 --max-tokens 1024 --eval-every 400 --log-every 200 \
     --bleu-sentences 150 --seed 1337 --pos-enc "$PE" --name "abl_posenc_$PE" \
     --notes "positional encoding = $PE" >> "$LOG" 2>&1
done

# --- Phase 8: Table 3 row (D), dropout --------------------------------------
say "ABLATION dropout=0.0"
$R --preset scaled --steps 400 --max-tokens 1024 --eval-every 400 --log-every 200 \
   --bleu-sentences 150 --seed 1337 --dropout 0.0 --name "abl_dropout0.0" \
   --notes "Table 3 row D: P_drop=0.0" >> "$LOG" 2>&1

# --- Extra: paper post-norm vs tensor2tensor pre-norm -----------------------
say "ABLATION pre-norm"
$R --preset scaled --steps 400 --max-tokens 1024 --eval-every 400 --log-every 200 \
   --bleu-sentences 150 --seed 1337 --norm-first --name abl_prenorm \
   --notes "pre-norm: tensor2tensor layout, not the paper text" >> "$LOG" 2>&1

# --- Extra: is the warmup schedule load-bearing for post-norm? --------------
say "ABLATION no warmup"
$R --preset scaled --steps 400 --max-tokens 1024 --eval-every 400 --log-every 200 \
   --bleu-sentences 150 --seed 1337 --warmup 1 --name abl_nowarmup \
   --notes "warmup_steps=1: tests whether warmup is load-bearing" >> "$LOG" 2>&1

say "ALL RUNS COMPLETE"
