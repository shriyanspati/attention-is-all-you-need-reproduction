# Reproducing *Attention Is All You Need* (Vaswani et al., NeurIPS 2017)

A from-scratch reproduction study of the Transformer. Every component of the
architecture, data pipeline, training procedure and evaluation is
reimplemented in PyTorch and verified against the paper's equations, rather
than adapted from an existing Transformer implementation.

**Read [`REPORT.md`](REPORT.md) for the full study.** This file covers the
repository layout and how to re-run everything.

---

## What this study concludes, in one paragraph

The architecture reproduces exactly: a 107-test suite checks every equation, mask,
and shape against naive loop transcriptions, and the closed-form parameter
count matches the implementation to the parameter. (Caveat: ~99 of those tests
compare our code against our own transcription, so they catch coding errors,
not misreadings.) The *results* do not reproduce, and cannot:
the available hardware is one CPU core with 3.9 GB of RAM, measured at ~400x slower than the paper's 8 P100 GPUs (386-419x over 5 timings), which projects the base model's 100k-step budget to ~185 days. The big model cannot be trained at all -- Adam's optimizer
state alone requires 3.19 GB. WMT14 is unreachable from this network, so
Multi30k EN-DE (29k pairs) is substituted. We therefore report a scientifically
valid *scaled* reproduction and a quantified account of the gap, and we flag
three discrepancies found in the paper itself (see REPORT.md, Section 7).

---

## Layout

```
project/
├── model/
│   ├── attention.py            scaled dot-product + multi-head attention (eq. 1, 3.2.2)
│   ├── encoder.py              encoder stack; post-norm (paper) and pre-norm (t2t)
│   ├── decoder.py              decoder stack: masked self-attn, cross-attn, FFN
│   ├── feed_forward.py         position-wise FFN (eq. 2)
│   ├── embeddings.py           shared weight matrix + sqrt(d_model) scaling (3.4)
│   ├── positional_encoding.py  sinusoidal / learned / none (3.5)
│   ├── masking.py              padding and causal mask construction
│   └── transformer.py          full model, weight tying, greedy + beam search
├── data/
│   ├── preprocessing.py        byte-pair encoding from scratch (Sennrich et al.)
│   └── dataset.py              right-shift, token-count dynamic batching
├── training/
│   ├── scheduler.py            Noam schedule (eq. 3) + Adam(0.9, 0.98, 1e-9)
│   ├── loss.py                 label smoothing (5.4) + unsmoothed CE for PPL
│   └── trainer.py              training loop, monitoring, checkpoint averaging
├── evaluation/
│   └── bleu.py                 corpus BLEU from scratch + sacrebleu cross-check
├── inference/
│   └── decoding.py             beam search, length penalty, checkpoint averaging
├── tests/                      107 verification tests (Level 1)
└── experiments/
    ├── benchmark.py            throughput/memory vs the paper's hardware
    ├── prepare_data.py         BPE training, vocab scaling study, encoding
    ├── run_experiment.py       training driver (presets + ablation flags)
    ├── make_plots.py           Phase 6 figures
    └── aggregate_results.py    Phase 7/8 result tables
```

## Reproducing

```bash
pip install torch sacrebleu numpy matplotlib

# 1. Data: downloads Multi30k, trains shared BPE, runs the vocab scaling study
python3 -m project.experiments.prepare_data --merges 4000 --scaling-study

# 2. Level 1: architecture verification (107 tests, ~14 s)
python3 -m unittest discover -s project/tests -t . -p "test_*.py" -v

# 3. Quantify the compute gap against the paper's configurations
python3 -m project.experiments.benchmark

# 4. Level 2b: main scaled reproduction
python3 -m project.experiments.run_experiment --preset scaled --steps 1600 \
    --max-tokens 1024 --seed 1337 --final-test --name main_scaled

# 5. Level 2a: the paper's exact base architecture (truncated budget)
python3 -m project.experiments.run_experiment --preset paper-exact --steps 80 \
    --max-tokens 1024 --name paper_exact_arch

# 6. Everything, including the Phase 8 ablation grid
./run_all.sh

# 7. Figures and tables
python3 -m project.experiments.make_plots
python3 -m project.experiments.aggregate_results
```

Long runs are resumable: `Trainer` writes `last.pt` at every evaluation and
restores model, both Adam moments, and the scheduler step. (Caveat: the data
stream restarts at the epoch implied by the step count, so batch ordering
after a resume is not byte-identical to an uninterrupted run.)

## Verification (fresh run)

```
$ python3 -m unittest discover -s project/tests -t . -p "test_*.py"
Ran 107 tests in 22.528s

OK
```

Full transcript in [`ci_test_log.txt`](ci_test_log.txt), regenerated on every
release. Test distribution:

| File | Tests | Covers |
|---|---|---|
| `test_training_and_eval.py` | 42 | LR schedule (eq. 3), label smoothing, BLEU, BPE, batching, end-to-end overfit |
| `test_architecture.py` | 21 | Stack composition, shapes, closed-form parameter counts, tying, decoding |
| `test_positional_encoding.py` | 13 | Sinusoid formula, sin/cos interleaving, relative-offset linearity, wavelengths |
| `test_attention.py` | 12 | Eq. (1) vs naive loops, `1/sqrt(d_k)` scaling, fused-vs-per-head, permutation equivariance |
| `test_embeddings_and_decoding.py` | 10 | Three-way tying object identity, `sqrt(d_model)`, length penalty, checkpoint averaging |
| `test_masking.py` | 9 | Causal/padding masks, autograd-Jacobian causality, batch-composition invariance |
| **Total** | **107** | |

Caveat repeated from the report: roughly 99 of these compare our implementation
against our own naive transcription of the paper's equations. They are strong
against coding errors and cannot catch a shared misreading of the paper. Only
the sacrebleu cross-check and the Table 3 parameter-count comparisons use an
external oracle.


## Reproducibility

| Item | Value |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.13.0+cu130 (CPU execution; no CUDA device present) |
| numpy / matplotlib / sacrebleu | 2.4.4 / 3.10.8 / 2.6.0 |
| Hardware | 1x Intel Xeon vCPU @ 2.10 GHz, 3.9 GiB RAM, no GPU |
| `torch.get_num_threads()` | 1 |
| Seeds | 1337 (default), plus 7 and 42 for the variance estimate |
| Determinism | `random`, `numpy`, and `torch` seeded per run; CPU kernels used here are deterministic. BPE training is seed-independent by construction (ties broken lexicographically). |

Every run writes `config.json` (full model + training + environment config,
parameter breakdown, peak LR), `log.jsonl` (per-step metrics), `summary.json`
(final metrics), and checkpoints to `experiments/runs/<name>/`.

## Deviations from the paper at a glance

Full discussion in REPORT.md Section 7. Nothing below is silent: each is a
command-line flag whose default is the paper's value, recorded in the saved
config of every run.

| # | Paper | Here | Why |
|---|---|---|---|
| 1 | WMT14 EN-DE, 4.5M pairs | Multi30k EN-DE, 29k pairs | statmt.org and HuggingFace blocked by network policy |
| 2 | ~37,000 shared BPE vocabulary | 4,104 (4,000 merges) | corpus has only 28,071 joint word types; 37k merges is not defined on it |
| 3 | 8x P100, 100k/300k steps | 1 CPU core, 5,000 steps (main) / 400 (ablations) / 25 (paper-exact arch) | measured ~400x slower; ~185 days projected for the base budget |
| 4 | ~25,000 src + 25,000 tgt tokens/batch | 1,024 tokens/batch | 25k-token batches need ~163 s/step here |
| 5 | warmup_steps = 4000 | 4% of the step budget | preserves the schedule's *shape*; 4000 would never leave the ramp |
| 6 | d_model 512, N=6 (base) | d_model 128, N=2 for the ablation grid | one paper-exact run is included separately (Level 2a) |
| 7 | newstest2014, case-sensitive BLEU | test_2016_flickr | follows from deviation 1 |
| 8 | Averaged last 5 checkpoints | averaged last 5 (exercised; +0.57 BLEU) | matches the paper's procedure |

## Findings about the paper itself

1. **Parameter counts are mutually inconsistent.** Our closed-form count
   (verified equal to the implementation) gives 63,045,632 for base against a
   reported 65M, and 214,171,648 for big against a reported 213M. Under the
   three-way weight tying of Section 3.4, 65M implies a ~40.8k vocabulary while
   213M implies ~35.9k -- these differ by ~4,960 tokens, so a single shared
   vocabulary cannot explain both.
2. **The stated wavelength range is imprecise.** Section 3.5 says wavelengths
   form "a geometric progression from 2*pi to 10000 * 2*pi". The progression is
   exactly geometric and starts exactly at 2*pi, but ends at
   2*pi * 10000^{510/512} = 9646.6 * 2*pi -- 3.5% below the stated endpoint.
3. **The paper text and the authors' reference code disagree on LayerNorm
   placement.** Section 3.1 specifies `LayerNorm(x + Sublayer(x))` (post-norm);
   tensor2tensor implements pre-norm. We default to the paper text and ablate
   both.

## Bugs the verification suite caught in our own code

Recorded because they are the reason the suite exists, and both were silent:

* **BPE vocabulary was not closed under its own merge operations.** Building
  the vocabulary from the final segmentation of training words left 32 of 52
  merge results out of it, so unseen words fell back to `<unk>` -- destroying
  the open-vocabulary property that motivates BPE. Held-out UNK rate went from
  nonzero to exactly 0.0 after the fix.
* **Off-by-one in decoder-input padding** added a spurious all-PAD column to
  every batch. Masked, so numerically harmless, but it wasted compute and
  indicated a misunderstanding of the right-shift.

## License / provenance

Multi30k (Elliott et al., 2016) is used under its original terms; only the
publicly mirrored `task1/raw` splits are downloaded. No pretrained weights are
used anywhere -- every number reported here comes from parameters initialized
and trained inside this container.
