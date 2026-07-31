# Reproduction results

## Headline comparison

| Metric | Paper (base) | This study (Level 2b, scaled) | This study (Level 2a, paper-exact arch) |
|---|---|---|---|
| Dataset | WMT14 EN-DE (4.5M pairs) | Multi30k EN-DE (29k pairs) | Multi30k EN-DE (29k pairs) |
| Test set | newstest2014 | test_2016_flickr | test_2016_flickr |
| BLEU | 27.3 | 19.17 | 0.00 (val subset) |
| Vocabulary | ~37,000 BPE (shared) | 4104 BPE (shared) | 4104 BPE (shared) |
| Parameters | 65M (reported) | 1.448M | 46.203M |
| Training steps | 100,000 | 5000 | 25 |
| Batch size | ~25,000 src + 25,000 tgt tokens | 1024 tokens | 1024 tokens |
| Training time | 12 h | 0.00 h | 0.03 h |
| Hardware | 8x NVIDIA P100 | 1x Xeon vCPU (no GPU) | 1x Xeon vCPU (no GPU) |
| s / step | 0.4 | 0.256 | 3.327 |
| Val perplexity (per BPE token) | 4.92 (newstest2013) | 13.98 | 594.88 |

## Measured compute gap (paper configurations on this hardware)

| Config | Params | s/step @ 25k tokens | Paper s/step | Slowdown | Projected time for paper budget |
|---|---|---|---|---|---|
| base | 63.0M | 154.8 | 0.4 | 387x | 179.2 days (0.49 yr) |
| big | 214.2M | OOM | 1.0 | n/a | infeasible: needs 3.19 GB for Adam state alone |

## Seed variance (the noise floor)

| Run | seed | best val loss | best val BLEU |
|---|---|---|---|
| abl_baseline_seed1337 | 1337 | 4.8349 | 6.29 |
| abl_baseline_seed42 | 42 | 4.8930 | 3.49 |
| abl_baseline_seed7 | 7 | 4.8782 | 5.46 |
| **mean +/- sd** | -- | 4.8687 +/- 0.0302 | 5.08 +/- 1.44 |

Seed-to-seed range: **2.80 BLEU** (3.49-6.29). Any ablation delta smaller than this is not distinguishable from noise with n=1.

## Ablations

All arms n=1, seed 1337. `d loss` is the PAIRED delta against the seed-1337 baseline (negative = better). No significance test is performed: with one run per arm the within-arm variance is unknown, so the 3-seed baseline range is shown only as a reference scale.

| Variant | params | val loss | d loss (paired) | > baseline seed range? | val BLEU (60 sents) | d loss vs 3-seed mean |
|---|---|---|---|---|---|---|
| dropout0.0 | 1.448M | 4.6989 | -0.1360 | yes | 5.97 | -0.1698 |
| heads1 | 1.448M | 4.9515 | 0.1166 | yes | 1.94 | 0.0828 |
| heads16 | 1.448M | 4.8338 | -0.0011 | no | 3.03 | -0.0350 |
| heads32 | 1.448M | 4.8317 | -0.0032 | no | 2.08 | -0.0370 |
| heads4 | 1.448M | 4.8725 | 0.0376 | no | 2.31 | 0.0038 |
| nowarmup | 1.448M | 6.6017 | 1.7668 | yes | 0.00 | 1.7330 |
| nowarmup_prenorm | 1.448M | 5.5707 | 0.7358 | yes | 0.81 | 0.7020 |
| posenc_learned | 1.481M | 4.6776 | -0.1573 | yes | 7.25 | -0.1911 |
| posenc_none | 1.448M | 4.6663 | -0.1686 | yes | 6.18 | -0.2024 |
| prenorm | 1.448M | 4.3400 | -0.4949 | yes | 8.93 | -0.5287 |

## Data pipeline

- Corpus: 29000 train / 1014 val / 1000 test sentence pairs
- Joint word types: 28071 (EN 10825, DE 18483)
- Shared BPE vocabulary: 4104 tokens from 4000 merges
- Held-out UNK rate: val 0.0, test 0.0

| merges | vocab size | mean tokens/sentence | val UNK rate |
|---|---|---|---|
| 1000 | 1104 | 19.46 | 0.0 |
| 2000 | 2104 | 16.79 | 0.0 |
| 4000 | 4104 | 15.17 | 0.0 |
| 8000 | 8104 | 14.08 | 0.0 |
| 16000 | 16104 | 13.55 | 0.0 |

| token budget | batches/epoch | mean sentences/batch | padding fraction |
|---|---|---|---|
| 1024 | 563 | 51.3 | 0.0062 |
| 4096 | 164 | 176.8 | 0.0276 |
| 25000 | 36 | 805.6 | 0.1302 |

## All runs

| run | params | steps | s/step | tok/s | peak RSS (MiB) | final val loss | final val ppl | best val BLEU |
|---|---|---|---|---|---|---|---|---|
| abl_baseline_seed1337 | 1.448M | 400 | 0.341 | 3007 | 917 | 4.8349 | 68.75 | 6.29 |
| abl_baseline_seed42 | 1.448M | 400 | 0.272 | 3760 | 903 | 4.8930 | 73.62 | 3.49 |
| abl_baseline_seed7 | 1.448M | 400 | 0.347 | 2955 | 926 | 4.8782 | 72.26 | 5.46 |
| abl_dropout0.0 | 1.448M | 400 | 0.229 | 4472 | 912 | 4.6989 | 59.34 | 5.97 |
| abl_heads1 | 1.448M | 400 | 0.264 | 3884 | 925 | 4.9515 | 78.40 | 1.94 |
| abl_heads16 | 1.448M | 400 | 0.272 | 3764 | 920 | 4.8338 | 68.51 | 3.03 |
| abl_heads32 | 1.448M | 400 | 0.460 | 2228 | 938 | 4.8317 | 68.12 | 2.08 |
| abl_heads4 | 1.448M | 400 | 0.242 | 4236 | 927 | 4.8725 | 71.07 | 2.31 |
| abl_nowarmup | 1.448M | 400 | 0.274 | 3732 | 904 | 6.6017 | 526.47 | 0.00 |
| abl_nowarmup_prenorm | 1.448M | 400 | 0.304 | 3369 | 910 | 5.5707 | 160.87 | 0.81 |
| abl_posenc_learned | 1.481M | 400 | 0.239 | 4280 | 907 | 4.6776 | 56.90 | 7.25 |
| abl_posenc_none | 1.448M | 400 | 0.243 | 4213 | 950 | 4.6663 | 56.30 | 6.18 |
| abl_prenorm | 1.448M | 400 | 0.261 | 3922 | 912 | 4.3400 | 38.28 | 8.93 |
| main_scaled | 1.448M | 5000 | 0.256 | 4002 | 927 | 3.4931 | 13.98 | 18.29 |
| obs_warmup64 | 1.448M | 2400 | 0.274 | 3739 | 923 | 4.6483 | 54.62 | 2.08 |
| paper_exact_arch | 46.203M | 25 | 3.327 | 308 | 2258 | 6.6999 | 594.88 | 0.00 |
