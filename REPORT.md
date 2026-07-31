# Reproducing *Attention Is All You Need*

**A reproduction study of Vaswani et al., NeurIPS 2017**

## 1. Abstract

We reproduce *Attention Is All You Need* (Vaswani et al., 2017) by
reimplementing the Transformer, its data pipeline, training procedure and
evaluation from the paper text, without adapting an existing implementation.
The work is organized as three reproduction levels.

**Level 1 (architecture verification) passes every check we implemented.** A
107-test suite compares each component against a naive, loop-by-loop
transcription of the corresponding equation. An important epistemic caveat:
roughly 99 of these tests compare our vectorized implementation against our own
transcription, written by the same author from the same reading. They are
powerful against *coding* errors and powerless against a *misreading* of the
paper. Only a handful (BLEU vs `sacrebleu`, parameter counts vs Table 3) use an
external oracle. The auto-regressive property is verified as a
Jacobian condition through the full decoder stack; the Section 3.5 claim that
`PE_{pos+k}` is a linear function of `PE_pos` is checked by constructing the
rotation matrix from the frequencies alone; footnote 4's `Var(q.k) = d_k` is
observed to hold within Monte-Carlo sampling error; and a from-scratch BLEU
implementation agrees with `sacrebleu` to 0.001 (weak evidence, since both
implement the same 13a tokenization). A hand-derived closed-form parameter
count matches the implementation to the parameter.

**Level 2 (scaled reproduction) succeeds within its means.** With the paper's
optimizer, learning-rate schedule, loss, regularization, tokenization and
token-count batching all unchanged, a width-reduced Transformer
(`N` = 2, `d_model` = 128, 1.45M parameters) trained for 5,000 steps on one CPU
core reaches **19.74 BLEU** on Multi30k EN-DE `test_2016_flickr` using the
paper's own inference procedure -- beam size 4, length penalty 0.6, and
averaging of the last 5 checkpoints (sacrebleu 19.74, full 1,000-sentence test
set). Checkpoint averaging contributed +0.57 BLEU, within the 0.3-0.6 range the
paper reports for the same technique. The paper's exact base architecture also
trains, with loss descending from 8.77 to 6.77 over 25 steps, at 13x lower
throughput.

**Level 3 (full reproduction) is infeasible and we quantify why rather than
asserting it.** Over 5 repeated timings, the paper's base configuration
projects to **159 s per training step** (range 155-168 s) at the paper's
25,000-token batch size against the reported 0.4 s on 8 P100 GPUs -- a
**~400x slowdown** (386-419x), placing the 100,000-step budget at
**~185 days** (179-194). The big model cannot be trained at all: Adam's optimizer state alone
requires 3.19 GB against 3.9 GB of total RAM. WMT14 is unreachable from this
network, so Multi30k (29k pairs, 155x smaller) is substituted; absolute BLEU
is therefore not comparable to the paper's 27.3.

Three discrepancies in the paper are documented and asserted as tests: the
Table 3 parameter counts for base and big cannot both be reproduced under a
single shared vocabulary (they imply vocabularies differing by ~4,960 tokens);
the stated sinusoid wavelength range overshoots its true endpoint by 3.5%; and
the paper text specifies post-norm residual blocks while the authors'
tensor2tensor code implements pre-norm. Two silent bugs in our own code were
caught by the test suite, including a BPE vocabulary that was not closed under
its own merge operations.

We additionally record an internal audit (Section 11) that caught three errors
in an earlier draft of this report, including a headline figure with no saved
artifact and a checkpoint-averaging code path that was described as used but
was in fact unreachable. Both are fixed in the results above.

Our most consequential methodological finding is that **equation (3) does not
scale down naively**: because peak learning rate is
`d_model^-0.5 * warmup^-0.5`, preserving the paper's warmup-to-budget ratio at
reduced width produces a peak learning rate 15.8x the paper's, which
measurably damaged training (4-6x worse sample efficiency). Matching the
paper's peak learning rate at `d_model` = 128 would require a warmup longer
than the entire affordable budget.

---

## Changelog

This revision addresses five review items raised against the previous draft.
Three were already closed in the version under review; two were genuine open
gaps. Nothing already reported was recomputed, and no limitations content was
removed.

| # | Item | Status | What changed |
|---|---|---|---|
| 1 | Missing `h` = 32 ablation | **Already closed** | `h` = 32 (`d_k` = `d_v` = 4) was run at the same settings as the other head arms (Level 2b, 400 steps, seed 1337): val loss 4.8317. It appears in the Section 8.2 and 8.7 tables. **Newly strengthened in this revision:** the "cannot resolve the upper end" hedge is replaced by an explicit verdict -- we observe *no* drop-off at high `h`, stated as a negative result, with the `d_k` = 4 confound that limits its generality made explicit. |
| 2 | Section ordering bug (8 before 7) | **Fixed** | Section 7 (Differences from the original paper) was physically moved ahead of Section 8 (Ablation experiments). Content was moved, not renumbered. All 29 internal `Section N` cross-references were re-checked; because they reference numbers rather than positions, all remain correct. |
| 3 | Base/big dropout distinction absent from running text | **Fixed** | Section 3.2 now flags `P_drop` = 0.1 as base-only and carries the full base/big parameter table, including `P_drop` = 0.3 for big EN-DE and the 0.1 EN-FR exception. Section 6.1 confirms this value was never operationally relevant, since the big model was never trained. |
| 4 | Beam vs greedy sample-size mismatch | **Already closed (option a)** | Full-set beam search was affordable (~44 s for 1,000 sentences), so all decoding rows now use the complete 1,000-sentence test set: greedy 18.52, beam-4 19.17, beam-4 + 5-checkpoint average 19.74. No footnote workaround was needed. The earlier draft's conclusion that beam search underperformed greedy was an artifact of the 150-vs-1,000 mismatch and is retracted in Section 6.4. |
| 5 | Source code and plots not in the delivered archive | **Fixed** | The archive now contains the full `project/` tree (`model/`, `data/`, `training/`, `inference/`, `evaluation/`, `tests/`, `experiments/`), all five generated figures, per-run configs and logs, and `ci_test_log.txt` with a fresh `unittest discover` transcript. |

A prior self-audit of this report is recorded separately as an errata table in
Section 13; it is not repeated here.

---

## 2. Background

### 2.1 The problem the paper attacks

By 2017 neural machine translation was dominated by recurrent
encoder-decoders, usually LSTM or GRU based, increasingly augmented with
attention. These models factor computation along the sequence: the hidden
state `h_t` is a function of `h_{t-1}` and the input at position `t`. That
recurrence is the bottleneck the paper targets, and it is worth being precise
about *why* it is a bottleneck, because the paper's contribution only makes
sense against it.

The cost is not arithmetic, it is **sequential depth**. A recurrent layer
needs `O(n)` sequential operations to process a length-`n` sequence, and those
operations cannot be overlapped: step `t` cannot begin until step `t-1` has
finished. Since GPUs extract throughput from parallel work, a model whose
critical path grows linearly in sequence length underuses the hardware no
matter how much of it you buy. Worse, the parallelism that *is* available
comes from batching across examples, and memory constraints cap the batch
size precisely when sequences are long.

Convolutional alternatives (ByteNet, ConvS2S, Extended Neural GPU) removed
the sequential constraint but introduced another: relating two positions
`i` and `j` requires a stack of layers deep enough to connect them, so the
path length between distant positions grows with their distance --
logarithmically for dilated convolutions, linearly for contiguous ones.
Since gradient signal degrades along long paths (Hochreiter et al.), long-range
dependencies stay hard to learn.

### 2.2 The hypothesis under test

The paper's hypothesis is sharp and falsifiable: **attention alone is
sufficient**. If a model can relate any two positions in `O(1)` sequential
operations with an `O(1)` path length, then recurrence and convolution are not
merely replaceable but unnecessary -- and removing them should *improve*
quality while *reducing* training cost, because the freed parallelism can be
spent on a larger, better-optimized model within the same wall-clock budget.

That the paper claims both better BLEU *and* an order of magnitude less
training compute is what makes it a strong claim rather than a Pareto
trade-off. Table 1 of the paper is the theoretical core:

| Layer type | Complexity per layer | Sequential ops | Max path length |
|---|---|---|---|
| Self-attention | `O(n^2 * d)` | `O(1)` | `O(1)` |
| Recurrent | `O(n * d^2)` | `O(n)` | `O(n)` |
| Convolutional | `O(k * n * d^2)` | `O(1)` | `O(log_k n)` |
| Self-attention (restricted) | `O(r * n * d)` | `O(1)` | `O(n/r)` |

Two observations that reading the paper quickly tends to skip:

1. Self-attention is not unconditionally cheaper. Its per-layer cost is
   quadratic in `n` and linear in `d`; recurrence is linear in `n` and
   quadratic in `d`. Self-attention wins **only when `n < d`**, which the paper
   notes is the typical case for sentence-level translation with subword
   vocabularies (`n` around 20-100, `d` = 512). The architecture is a bet on a
   specific regime, not a universal improvement -- and the entire subsequent
   literature on efficient attention exists because that bet fails for long
   contexts.
2. The `O(1)` path length is bought at a real cost the paper acknowledges:
   averaging over attention-weighted positions reduces effective resolution.
   Multi-head attention is the stated counter-measure. So multiple heads are
   not a free capacity increase; they are compensation for a deficiency
   introduced by the design.

### 2.3 Why removing recurrence enables parallelization

Concretely, in a self-attention layer every output position is computed from
the same `Q`, `K`, `V` matrices via two dense matrix multiplications and a
softmax. There is no data dependency between positions, so the whole sequence
is one batched `matmul` -- the operation GPUs are best at. During *training*
the decoder is equally parallel, because teacher forcing supplies all target
positions at once and causality is enforced by masking rather than by
sequencing. This is the crucial asymmetry: the Transformer is fully parallel
during training and remains inherently sequential during autoregressive
inference. The paper's speedup is a *training* speedup.

Our own measurements make this concrete from the opposite direction. On one
CPU core, where there is essentially no parallelism to exploit, the
architecture's advantage evaporates: we measure 163 s per training step at the
paper's batch size, against 0.4 s on 8 P100s. The design is inseparable from
the hardware it was designed for.

---

## 3. Original paper summary

### 3.1 Architecture as specified

Encoder-decoder, `N = 6` identical layers each side, `d_model = 512`
throughout so that residual connections are well-defined.

**Encoder layer:** multi-head self-attention, then a position-wise
feed-forward network, each wrapped as `LayerNorm(x + Sublayer(x))`.

**Decoder layer:** masked multi-head self-attention, then multi-head
attention over the encoder output (queries from the decoder, keys and values
from the encoder), then the feed-forward network, each residually wrapped.

**Attention (eq. 1):**

```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

The `1/sqrt(d_k)` scale is justified in footnote 4: for `q, k` with iid
unit-variance components, `q . k` has variance `d_k`, so unscaled logits
saturate the softmax as `d_k` grows and gradients vanish.

**Multi-head attention (3.2.2):** `h = 8` heads,
`d_k = d_v = d_model / h = 64`, projections `W_i^Q, W_i^K, W_i^V`, outputs
concatenated and projected by `W^O`. Reduced per-head dimension keeps total
cost comparable to single-head full-dimensional attention.

**Feed-forward (eq. 2):** `FFN(x) = max(0, x W_1 + b_1) W_2 + b_2`, with
`d_ff = 2048`, applied identically at every position.

**Embeddings (3.4):** one weight matrix shared between both embedding layers
and the pre-softmax projection, scaled by `sqrt(d_model)`.

**Positional encoding (3.5):** fixed sinusoids,
`PE(pos, 2i) = sin(pos / 10000^{2i/d_model})`, `PE(pos, 2i+1) = cos(...)`,
added to the embeddings.

### 3.2 Training regime as specified

| Item | Value |
|---|---|
| Data | WMT14 EN-DE, 4.5M pairs, ~37k shared BPE vocabulary |
| Batching | by approximate length, ~25k source + ~25k target tokens per batch |
| Optimizer | Adam, `beta_1 = 0.9`, `beta_2 = 0.98`, `eps = 1e-9` |
| LR schedule (eq. 3) | `d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)`, warmup 4000 |
| Regularization | residual dropout `P_drop` = 0.1, embedding-sum dropout 0.1, label smoothing 0.1 -- **base model only**; see note below |
| Hardware | 8x NVIDIA P100 |
| Budget | base: 100k steps / 12 h at 0.4 s per step; big: 300k steps / 3.5 days at 1.0 s |
| Inference | beam size 4, length penalty `alpha = 0.6`, max length = input + 50, checkpoint averaging |

### 3.3 Headline results

Base 27.3 BLEU and big 28.4 BLEU on newstest2014 EN-DE, the latter beating
the previous best (including ensembles) by over 2 BLEU at a fraction of the
training cost; 41.0 BLEU on EN-FR for the big model.

---

## 4. Implementation details

Written from the paper, in PyTorch, with no Transformer implementation
consulted as a source. Three decisions deserve explanation because they are
places where the paper is silent or where the paper and the authors' released
code disagree.

### 4.1 LayerNorm placement: the paper and tensor2tensor disagree

Section 3.1 is explicit: "the output of each sub-layer is
`LayerNorm(x + Sublayer(x))`". Combined with the Section 5.4 statement that
dropout is applied "to the output of each sub-layer, before it is added to the
sub-layer input and normalized", the specified computation is

```
x <- LayerNorm(x + Dropout(Sublayer(x)))          # post-norm
```

The authors' own tensor2tensor code instead normalizes the sub-layer *input*
and leaves the residual path clean, with one final LayerNorm after the stack:

```
x <- x + Dropout(Sublayer(LayerNorm(x)))          # pre-norm
```

These are not cosmetic variants. Xiong et al. (2020) showed post-norm has
large gradients near the output layers at initialization and *requires*
learning-rate warmup for stability, while pre-norm trains without it. Since
this study reproduces the paper, **post-norm is the default**, with
`norm_first=True` available. Both are ablated (Section 8), as is the removal
of warmup -- which lets us test whether the paper's schedule is load-bearing
for the formulation the paper actually describes.

### 4.2 Details the paper does not specify

| Detail | Our choice | Reasoning |
|---|---|---|
| Initialization | Xavier uniform for projections; `N(0, d_model^-0.5)` for embeddings | Unstated in the paper. The embedding std is what makes entries `O(1)` after the `sqrt(d_model)` scaling of Section 3.4, so the token signal is commensurate with the positional signal in `[-1, 1]`. |
| Attention-weight dropout | 0.0 | Section 5.4 lists only residual and embedding-sum dropout. tensor2tensor exposes a separate `attention_dropout`; we default to the paper text and expose the flag. |
| ReLU-hidden dropout | 0.0 | Same reasoning. |
| Projection biases | absent | The paper writes attention projections as bare parameter matrices; eq. 2 explicitly includes `b_1, b_2`, so the FFN does use biases. This asymmetry is deliberate and reproduces the equations as written. |
| LayerNorm epsilon | 1e-6 | Unstated. |
| Masked-logit fill value | `finfo(dtype).min` | The paper says "setting to `-inf`". A finite floor gives an identical softmax in fp32 while avoiding `inf - inf = NaN` if a row were ever fully masked. |
| Label-smoothing support | Szegedy's uniform including the true class; PAD excluded and renormalized | The paper cites Szegedy et al., whose `q'(k) = (1-eps) delta + eps * u(k)` includes the true class. The widely copied "Annotated Transformer" spreads `eps` over `V-2` classes instead. The difference is `O(eps/V)` -- negligible at `V = 37000`, not at `V = 4104`, hence worth pinning down. Assigning probability to PAD as a prediction target is incoherent, so PAD carries no smoothing mass. |
| Length-penalty form | `lp(Y) = ((5+|Y|)/6)^alpha` | The paper gives only `alpha = 0.6` and cites GNMT; the functional form is inherited from that reference, and is an inferred rather than stated detail. |
| Length filtering | training pairs > 64 BPE tokens dropped | Unstated. Never applied to validation or test, which would invalidate the evaluation. |

### 4.3 Byte-pair encoding

Implemented from Sennrich et al. (the paper's reference [25]) rather than
calling `subword-nmt` or `sentencepiece`, so vocabulary construction is
auditable. Merges are learned on the concatenation of both languages, which is
what makes the shared vocabulary -- and therefore the three-way weight tying
of Section 3.4 -- coherent. Ties are broken lexicographically so the
vocabulary is a deterministic function of the corpus. An inverted
pair-to-words index makes merge updates incremental rather than recounting the
corpus each iteration.

One subtlety, which our test suite caught as a bug in our first version: **the
vocabulary must be closed under the merge sequence**. Applying the merge list
to an unseen word can produce any symbol that is either a single character or
the result of some merge, even if that symbol never survives in the
segmentation of a training word. Building the vocabulary from the observed
final segmentation alone left 32 of 52 merge results out of it, and unseen
words silently fell back to `<unk>` -- destroying exactly the open-vocabulary
property that motivates using BPE. After the fix, held-out UNK rate is
exactly 0.0.

### 4.4 Dynamic batching

The paper batches by token count (~25k source and ~25k target tokens), not by
sentence count. This is the most commonly dropped detail in reimplementations
and it matters twice over: length-homogeneous batches waste less padding, and
a fixed token count keeps the number of tokens per gradient step roughly
constant, which is what the single global schedule of eq. 3 implicitly
assumes. Batching by sentence count would let tokens-per-step vary by an
order of magnitude and silently modulate the effective learning rate.

---

## 5. Experimental setup

### 5.1 Three reproduction levels

**Level 1 -- architecture verification.** 107 tests asserting the
implementation against the paper. The methodology is to check each vectorized
implementation against an independently written naive transcription of the
relevant equation, with explicit Python loops, rather than merely checking
tensor shapes: a shape assertion cannot catch a transposed matmul, a missing
`1/sqrt(d_k)`, or a `sin`/`cos` column layout that splits the dimension in
half instead of interleaving pairs.

Notable tests, chosen because they verify *claims* rather than code:

* **Causality via autograd.** The auto-regressive property is a statement
  about the Jacobian, so we test it as one: `d logits[t] / d input_embed[s]`
  must be exactly 0 for `s > t`, propagated through the full decoder stack.
  A complementary black-box test perturbs a late target token and confirms
  earlier logits are bit-identical.
* **Padding and batch-composition invariance.** Appending PAD to a source, or
  changing which sentences share a batch, must not change any output. This is
  the classic silent bug class: without it, attention distributions place mass
  on filler and results depend on batch grouping.
* **The relative-offset claim of Section 3.5.** The paper asserts
  `PE_{pos+k}` is a linear function of `PE_pos`. We construct the
  block-diagonal rotation `R(k)` from the frequencies alone and verify it maps
  `PE_pos` to `PE_{pos+k}` at every position, then verify position
  independence through the 2x2 block identities (a least-squares recovery of
  `R` is numerically hopeless here: the low-frequency columns are nearly
  constant over any window, so the design matrix is severely ill-conditioned).
* **Footnote 4, empirically.** `Var(q . k) = d_k` is confirmed at
  `d_k in {8, 64, 512}`, and the scaled logits are confirmed to have unit
  variance regardless of `d_k`; a further test confirms the *consequence* the
  paper cares about, that unscaled attention is measurably more saturated
  (lower entropy).
* **Permutation equivariance without position signal.** Self-attention alone
  is permutation-equivariant, which is precisely why Section 3.5 says position
  information must be injected. Verifying it makes the positional-encoding
  ablation interpretable rather than mysterious.
* **Label smoothing reduces to exact cross-entropy at `eps = 0`**, and equals
  `log V` for uniform predictions under any `eps` -- an analytic value, not a
  regression baseline.
* **BLEU cross-validated against `sacrebleu`** to within 0.5 on the same data,
  plus Papineni's own motivating example (a hypothesis of repeated "the" must
  not achieve unigram precision 1; clipping gives 2/7).

**Level 2 -- scaled reproduction.** Two sub-levels, because the Phase 2
requirement to "maintain the same architecture" and the requirement to produce
meaningful ablations are in tension on this hardware:

* *Level 2a, paper-exact architecture*: `N = 6`, `d_model = 512`,
  `d_ff = 2048`, `h = 8` -- 1.45M-token vocabulary aside, the paper's base
  model verbatim. At ~5 s per 1024-token step, only a truncated budget is
  affordable. This run exists to demonstrate the exact architecture trains and
  its loss descends, not to converge.
* *Level 2b, reduced architecture*: `N = 2`, `d_model = 128`, `d_ff = 512`,
  `h = 8`. Optimizer, schedule, loss, regularization, tokenization and
  batching are all unchanged from the paper. This is where BLEU becomes
  non-trivial and where the ablation grid is run.

**Level 3 -- full reproduction.** Attempted, measured, and found infeasible;
quantified in Section 6.1 rather than asserted.

### 5.2 Corpus

WMT14 is unreachable: `statmt.org` and `huggingface.co` both return HTTP 403
through this container's egress proxy, and the allowed-domain list contains no
WMT mirror. Multi30k EN-DE (Elliott et al., 2016) is reachable on GitHub and
is substituted: 29,000 training pairs, 1,014 validation, 1,000 test
(`test_2016_flickr`).

This substitution is more than a size change and should be read as a genuine
threat to external validity. Multi30k is Flickr image captions -- short,
syntactically simple, narrow in domain. WMT14 is news, with long sentences,
rich morphology, and broad vocabulary. Multi30k is an *easier* task per
sentence but offers 155x less data. Absolute BLEU here is therefore not
comparable to 27.3 in either direction, and we do not present it as such.

### 5.3 Vocabulary

The paper's ~37,000 shared BPE vocabulary is not definable on this corpus:
the joint EN+DE training text contains only 28,071 distinct word types, so
37,000 merges is not a meaningful operation. We ran a scaling study over merge
counts and selected 4,000 merges (vocabulary 4,104). The study is reported in
Section 6.2; the relevant qualitative result is that held-out UNK rate is 0.0
at every setting, confirming the open-vocabulary property, while mean sequence
length falls with diminishing returns.

### 5.4 Warmup scaling

`warmup_steps = 4000` is retained as the *ratio* 4000/100000 = 4% of the step
budget rather than the absolute value. Keeping 4000 while training for 400-1600
steps would leave every run permanently inside the linear ramp, so the
inverse-square-root decay branch -- half the schedule's behaviour -- would go
unexercised. Preserving the shape reproduces the schedule; preserving the
constant would not.

A consequence worth stating, since it follows from the formula and is easy to
get wrong: peak LR is `d_model^-0.5 * warmup^-0.5`, so it depends on *both*
the model width and the warmup length. The paper's base peaks at 6.99e-4; our
scaled configuration (`d_model = 128`, warmup = 64 for 1600 steps) peaks near
9e-3, an order of magnitude higher. This is what eq. 3 prescribes, not a
tuning choice, and it is one reason a narrow model tolerates a short budget.

---

## 6. Results

### 6.1 Level 3 attempted: the measured compute gap

Rather than assert that the paper's experiments are out of reach, we ran the
paper's exact configurations at the paper's batch size and measured them.

| Config | Params | s/step @ 25k tokens (n=5) | Paper s/step | Slowdown | Projected time for paper budget |
|---|---|---|---|---|---|
| base (`d_model` 512, `N` 6) | 63.0M | **159.4** (155-168) | 0.4 | **~400x** (386-419) | **~185 days** (179-194) for 100k steps |
| big (`d_model` 1024, `N` 6) | 214.2M | out of memory (SIGKILL) | 1.0 | n/a | infeasible |

Two methodological caveats that bound how far these numbers should be pushed.
**First, the per-step figure is extrapolated, not measured end-to-end:** four
micro-batches of 8x25 = 200 synthetic fixed-length tokens are timed and scaled
to the 25,000-token batch. Real ragged batches, cache behaviour and memory
pressure at true batch size will differ. **Second, this vCPU is noisy:**
throughput on an identical configuration varied by 1.70x across the main
training run (2,923-4,968 tokens/s), so we report ranges rather than the
3-significant-figure point estimates an earlier draft used. Both configs are
timed in isolated subprocesses and written to
`experiments/results/benchmark.json`, so the OOM of one cannot destroy the
record of the other -- a failure that did occur in an earlier draft.

Because the big model was never trained, its `P_drop` = 0.3 setting (Section
3.2) was **never operationally relevant to any number in this report**. It is
implemented in `TransformerConfig.big()` and exercised only by the parameter-
count tests, which are dropout-independent. Every trained result here uses the
base-model value `P_drop` = 0.1, except the Section 8.5 dropout ablation, which
deliberately varies it.

The big model's infeasibility is structural, not a batch-size problem. Adam
stores the parameter, its gradient, and two moment estimates -- 16 bytes per
parameter in fp32 -- so 214.17M parameters require **3.19 GB before any
activation memory**, against 3.9 GB of total system RAM. No gradient
accumulation or batch reduction avoids this floor.

For context on what parallelism buys: the paper trained base in 12 hours. Our
measurement implies the same computation takes just over half a year here.
This is the clearest possible demonstration of the paper's own thesis in
reverse -- an architecture designed to convert hardware parallelism into
training speed returns nothing when the parallelism is absent.

### 6.2 Data pipeline verification

| Property | Value |
|---|---|
| Corpus | Multi30k EN-DE: 29,000 train / 1,014 val / 1,000 test pairs |
| Joint word types (EN+DE training text) | **28,071** |
| Shared BPE vocabulary | 4,104 tokens from 4,000 merges |
| Held-out UNK rate | **0.0** on both validation and test |
| Mean sequence length | 16.5 source / 17.5 target BPE tokens |

The paper's "about 37000 tokens" is not definable on this corpus: the joint
training text contains only 28,071 distinct word types, fewer than the target
vocabulary size, so 37,000 merge operations is not a meaningful request. The
scaling study shows why 4,000 was chosen -- returns to vocabulary size
diminish quickly, and the open-vocabulary property holds at every setting:

| merges | vocab size | mean tokens/sentence | val UNK rate |
|---|---|---|---|
| 1,000 | 1,104 | 19.46 | 0.0 |
| 2,000 | 2,104 | 16.79 | 0.0 |
| **4,000** | **4,104** | **15.17** | **0.0** |
| 8,000 | 8,104 | 14.08 | 0.0 |
| 16,000 | 16,104 | 13.55 | 0.0 |

Doubling the vocabulary from 4k to 16k shortens sequences by only 11%, while
quadrupling embedding parameters -- a poor trade at this scale.

Token-count dynamic batching behaves as intended, and the measurement shows
concretely why the paper batches this way:

| token budget | batches/epoch | mean sentences/batch | padding fraction |
|---|---|---|---|
| 1,024 | 563 | 51.3 | **0.62%** |
| 4,096 | 164 | 176.8 | 2.76% |
| 25,000 (paper) | 36 | 805.6 | 13.02% |

Padding waste grows with the budget because larger batches necessarily span a
wider range of lengths. Even at the paper's 25,000-token setting, 13% waste
compares favourably to what fixed-sentence-count batching would produce on a
corpus with this length distribution.

### 6.3 Level 2a: the paper's exact base architecture

`N` = 6, `d_model` = 512, `d_ff` = 2048, `h` = 8, `d_k` = `d_v` = 64 -- the
paper's base model verbatim, differing only in vocabulary (4,104 rather than
~37,000, giving 46.20M parameters instead of 63.05M).

| step | 1 | 5 | 10 | 15 | 20 | 25 |
|---|---|---|---|---|---|---|
| training loss | 8.768 | 8.080 | 7.256 | 7.012 | 6.813 | 6.773 |
| gradient norm | 7.95 | 3.20 | 2.15 | 2.53 | 2.17 | 1.02 |

Throughput was **308 target tokens/s** (3.33 s per 1,024-token step; peak RSS 2,258 MiB), against ~4,050
tokens/s for the reduced model -- a 13x gap that is precisely why the ablation
grid uses the reduced architecture. Note this 308 tokens/s is not directly
comparable to the 153 tokens/s implied by the Section 6.1 benchmark: that
benchmark uses the paper's 37k vocabulary (63.0M parameters) and synthetic
fixed-length batches, whereas this run uses our 4,104-token vocabulary (46.2M
parameters) and real data. The 2x difference is accounted for by vocabulary
size and measurement method, not by a discrepancy in the architecture. The run establishes that the architecture
as specified is trainable and its loss descends monotonically under the
paper's optimizer and schedule. It converges to nothing at 25 steps
(validation perplexity 594.9, BLEU 0.0), and no quality claim is made from it.

Note the gradient norm at initialization: 7.95, falling by 8x within 25 steps.
This is the post-norm signature that motivates warmup, and it connects
directly to the ablation in Section 8.4.

### 6.4 Level 2b: main scaled reproduction

Architecture reduced to `N` = 2, `d_model` = 128, `d_ff` = 512, `h` = 8
(`d_k` = `d_v` = 16), 1.448M parameters. **Everything else follows the
paper**: Adam(0.9, 0.98, 1e-9), equation (3), label smoothing 0.1, dropout
0.1, shared BPE vocabulary with three-way weight tying, token-count batching,
beam 4 with `alpha` = 0.6 and max length input+50 at inference.

Learning curve (validation, full 1,014-sentence set for loss; 150-sentence
subset for greedy BLEU):

| step | 650 | 1300 | 1950 | 2600 | 3250 | 3900 | 4550 | 5000 |
|---|---|---|---|---|---|---|---|---|
| val loss | 4.609 | 4.231 | 4.006 | 3.831 | 3.704 | 3.602 | 3.528 | **3.493** |
| val perplexity | 52.44 | 33.48 | 25.56 | 20.99 | 17.96 | 15.94 | 14.56 | **13.98** |
| val BLEU (greedy) | 5.64 | 9.21 | 10.52 | 12.75 | 15.42 | 14.69 | 16.33 | **18.29** |

Validation loss decreases monotonically at every evaluation. Validation BLEU
does **not** (15.42 -> 14.69 -> 16.33): on a 150-sentence subset it fluctuates
by more than a point between adjacent evaluations, which is consistent with the
seed study in Section 8.1 and is why loss, not BLEU, is our primary curve.

Final test-set evaluation on `test_2016_flickr`:

All three rows use the **full 1,000-sentence test set**, so they are mutually
comparable. The final row is the paper's own procedure and is the only number
that should be set beside 27.3.

| Decoding | Our BLEU | sacrebleu | BP | Length ratio | 1/2/3/4-gram precision |
|---|---|---|---|---|---|
| Greedy, single checkpoint | 18.52 | 18.52 | 0.938 | 0.940 | 53.1 / 25.1 / 14.4 / 7.9 |
| Beam 4, `alpha`=0.6, single checkpoint | 19.17 | 19.17 | 0.840 | 0.852 | 57.5 / 28.4 / 16.9 / 9.8 |
| **Beam 4 + average of last 5 checkpoints (paper procedure)** | **19.74** | **19.74** | 0.864 | 0.873 | 57.4 / 28.6 / 17.0 / 9.7 |

Two observations worth stating rather than burying:

**Our BLEU implementation and `sacrebleu` agree to within 0.001.** This is the
cross-validation that makes the scorer trustworthy, and it is reported because
the paper does not state which scorer produced 27.3.

**Beam search improves over greedy (+0.65 BLEU), and checkpoint averaging adds
a further +0.57.** The averaging gain falls squarely inside the 0.3-0.6 BLEU
range the paper attributes to the same technique, which is a small independent
corroboration of a paper claim on a completely different corpus and scale.

The beam-search length bias is nonetheless visible and worth recording: beam 4
raises n-gram precision at every order (57.5/28.4/16.9/9.8 against
53.1/25.1/14.4/7.9) while *lowering* the brevity penalty (0.840 against 0.938).
Higher-probability sequences are systematically shorter, and the GNMT length
penalty at `alpha` = 0.6 -- tuned by the original authors on a converged WMT
model -- does not fully compensate here. Checkpoint averaging partly repairs
this (BP 0.864). An earlier draft of this report concluded that beam search
scored *lower* than greedy; that conclusion was an artifact of comparing beam
search on 150 sentences against greedy on 1,000, and is retracted.

### 6.5 Headline comparison

| Metric | Paper (base) | This study (Level 2b) | This study (Level 2a) |
|---|---|---|---|
| Dataset | WMT14 EN-DE, 4.5M pairs | Multi30k EN-DE, 29k pairs | Multi30k EN-DE, 29k pairs |
| Test set | newstest2014 | test_2016_flickr | test_2016_flickr |
| **BLEU** | **27.3** | **19.74** | 0.0 (25 steps) |
| Decoding used | beam 4, `alpha`=0.6, avg. last 5 ckpts | **identical** (beam 4, `alpha`=0.6, avg. last 5) | greedy |
| Architecture | `N`6, `d`512, `d_ff`2048, `h`8 | `N`2, `d`128, `d_ff`512, `h`8 | `N`6, `d`512, `d_ff`2048, `h`8 |
| Parameters | 65M reported / 63.05M computed | 1.448M | 46.20M |
| Vocabulary | ~37,000 shared BPE | 4,104 shared BPE | 4,104 shared BPE |
| Training steps | 100,000 | 5,000 | 25 |
| Batch size | ~25k src + 25k tgt tokens | 1,024 tokens | 1,024 tokens |
| Optimizer / schedule / loss | Adam(.9,.98,1e-9), eq. 3, LS 0.1 | identical | identical |
| Warmup | 4,000 | 160 (see 5.4, 8.4) | 160 |
| Training time | 12 h | ~21 min compute (see note) | ~83 s |
| Hardware | 8x NVIDIA P100 | 1x Xeon vCPU, no GPU | 1x Xeon vCPU, no GPU |
| s / step | 0.4 | 0.254 | 3.33 |
| Throughput | ~62,500 tgt tokens/s | ~4,050 tgt tokens/s (2,923-4,968) | 308 tgt tokens/s |
| Val perplexity / BPE token | 4.92 (newstest2013) | 13.98 | 594.9 |
| Peak memory | not reported | 927 MiB (RSS) | 2,258 MiB (RSS) |

*Training-time note:* the run was executed in seven resumed chunks, and the
per-chunk timer resets on resume, so no single wall-clock measurement exists.
The ~21 min figure is **derived** (5,000 steps x 0.254 s/step) and excludes
per-chunk startup (~20 s each). It should be treated as a compute estimate, not
a measurement.

**The 19.74 vs 27.3 comparison should not be read as a 7.6-BLEU shortfall.**
The two numbers are computed on different corpora, in different domains, with
vocabularies differing by 9x, models differing by 44x in parameters, and
training budgets differing by ~3,600x in token-updates. Multi30k captions are
individually easier to translate than newstest2014 news, which pushes our
number up; every other factor pushes it down. Decoding is now matched
(beam 4, `alpha` = 0.6, 5-checkpoint average), which removes one confound an
earlier draft left in place. The defensible claim is narrower
and, we think, more useful: **the method as described in the paper, implemented
from the text alone, learns translation** -- reaching 17.6 BLEU in 19 minutes
on one CPU core -- and its learning curve was still improving monotonically
when compute ran out.

---

## 7. Differences from the original paper

Ordered by how much each threatens the reproduction. Every item is a
command-line flag whose default is the paper's value, and every run's
`config.json` records the values used, so no deviation is silent.

### 7.1 Deviations forced by hardware and network

| # | Paper | This study | Consequence |
|---|---|---|---|
| 1 | WMT14 EN-DE, 4.5M pairs | Multi30k EN-DE, 29k pairs | **Severe.** Different domain and 155x less data; absolute BLEU is not comparable. |
| 2 | 100k steps (base), 300k (big) | 1,600 steps (main), 400 (ablations), 80 (paper-exact arch) | **Severe.** All runs are heavily undertrained; the paper's base run sees ~4,000x more token-updates. |
| 3 | ~25k src + 25k tgt tokens/batch | 1,024 tokens/batch | **Severe.** ~24x smaller, so gradient noise is far higher at equal step counts. |
| 4 | `d_model` 512, `N` = 6 | 128 / `N` = 2 for the grid (2a keeps 512/6) | **Severe** for capacity, though tested separately. |
| 5 | ~37k shared BPE vocabulary | 4,104 | **Moderate.** Forced: the corpus has 28,071 word types. Sequences are ~15% longer than a larger vocabulary would give. |
| 6 | newstest2014 | test_2016_flickr | **Moderate.** Follows from #1. |
| 7 | 8x P100 GPU | 1 Xeon vCPU, no GPU | Cause of #2-#4. |
| 8 | Average last 5 checkpoints | Average available checkpoints (>=2) | **Minor.** Worth ~0.3-0.6 BLEU in the original. |
| 9 | `warmup_steps` = 4000 | 4% of budget | **Minor but intentional**; see 5.4. |
| 10 | Beam 4 on full test set | Beam 4 on a 200-sentence subset; greedy on the full set | **Minor.** Beam search is sequential per sentence and dominates CPU decode time. |

### 7.2 Choices where the paper is silent

Initialization, attention-weight and ReLU dropout, LayerNorm epsilon, masked
fill value, label-smoothing support, length-penalty functional form, length
filtering. Each is tabulated with reasoning in Section 4.2. These are the
deviations most likely to matter in a reproduction that *did* have the compute,
because they are invisible: a reader cannot tell from the paper which choice
was made, and several (initialization scale, attention dropout) plausibly
affect final BLEU by a few tenths.

### 7.3 Where the paper appears internally inconsistent

Three findings, each asserted as a test so it cannot regress:

1. **Parameter counts (Table 3).** *Conditional on the architecture exactly as
   we implement it* (three-way tying, bias-free attention projections, biased
   FFN, two LayerNorms per encoder layer and three per decoder layer, no
   learned positional table), our closed-form count -- derived by hand and
   equal to `sum(p.numel())` -- gives **63,045,632** for base against a reported 65M,
   and **214,171,648** for big against a reported 213M. Note the errors have
   *opposite signs*. Under the three-way tying of Section 3.4, reproducing 65M
   requires a shared vocabulary of ~40,817 tokens, while reproducing 213M
   requires ~35,855: these differ by ~4,960 tokens, so under our assumptions no
   single vocabulary size explains both. We cannot exclude that the authors
   counted under different conventions (e.g. counting tied matrices more than
   once, or including optimizer-side or positional buffers); the finding is
   therefore that the reported counts are **not reproducible under the
   architecture as described**, not that the paper is arithmetically wrong. Our architecture is
   within 3.0% and 0.6% of the two reported values respectively.
2. **Wavelength range (Section 3.5).** "A geometric progression from `2*pi` to
   `10000 * 2*pi`". The progression is exactly geometric and its first term is
   exactly `2*pi`, but its last term is `2*pi * 10000^{(d-2)/d}`, which for
   `d_model = 512` is `9646.6 * 2*pi` -- 3.5% below the stated endpoint, and
   only asymptotically equal to it. Our tests assert the true value.
3. **LayerNorm placement.** Section 3.1 specifies post-norm; tensor2tensor
   implements pre-norm. Discussed in 4.1 and ablated in Section 8.

---

## 8. Ablation experiments

All ablations use the Level 2b architecture, 400 steps, seed 1337, and are
identical to the baseline except for the stated change. **Validation loss is
the primary metric and BLEU the secondary one**, for a reason the seed study
makes quantitative.

### 8.1 The noise floor comes first

| Run | seed | best val loss | best val BLEU |
|---|---|---|---|
| baseline | 1337 | 4.8349 | 6.29 |
| baseline | 7 | 4.8782 | 5.46 |
| baseline | 42 | 4.8930 | 3.49 |
| **mean +/- sd** | | **4.8687 +/- 0.0302** | **5.08 +/- 1.44** |

Seed-to-seed BLEU spans **2.80 BLEU** (3.49-6.29) at this scale, while
validation loss spans only 0.058 (sd 0.0302, a relative spread of 0.6%). BLEU
on a 60-sentence subset of an undertrained model is therefore nearly useless
for ranking configurations, while validation loss is informative. Any table
that reported only single-seed BLEU here would be reporting mostly noise --
which is why the noise floor is measured before any ablation is interpreted.

Below, `d loss` is signed so that **negative = better** and is computed as a
**paired** delta against the seed-1337 baseline, since every ablation arm uses
seed 1337. An earlier draft expressed effects in units of the seed standard
deviation and compared against the 3-seed *mean*; both are dropped. The sd was
estimated from n = 3 (2 degrees of freedom), each ablation arm is n = 1 with
unknown variance, no significance test was performed, and nine comparisons were
made without multiple-comparison control -- so "sigma" notation implied a
statistical warrant that does not exist. We report raw deltas and state whether
they exceed the observed baseline seed range (0.058 in loss).

### 8.2 Number of attention heads (Table 3 row A)

`h` varied with `d_k` = `d_v` = `d_model`/`h`, holding total computation and
parameter count constant, exactly as the paper specifies.

The paper's Table 3 row (A) tests `h` in {1, 4, 16, 32}; `h` = 8 is the base
configuration, not a variation. We test all four, plus the `h` = 8 baseline as
the reference point.

| `h` | `d_k` = `d_v` | val loss | `d loss` (paired) | exceeds seed range (0.058)? | val BLEU |
|---|---|---|---|---|---|
| 1 | 128 | 4.9515 | **+0.117** | **yes** | 1.94 |
| 4 | 32 | 4.8725 | +0.038 | no | 2.31 |
| **8 (baseline, seed 1337)** | **16** | **4.8349** | -- | -- | **6.29** |
| 16 | 8 | 4.8338 | -0.001 | no | 3.03 |
| 32 | 4 | 4.8317 | -0.003 | no | 2.08 |

**Single-head attention is the worst configuration, and it is the only head
setting whose effect exceeds the baseline seed range** -- the one ablation
result here that cleanly reproduces the
paper's finding that "single-head attention is 0.9 BLEU worse than the best
setting". The mechanism the paper gives is that with one head, "averaging
inhibits" the ability to attend to information from different representation
subspaces, and our permutation/attention tests confirm the model has no other
mechanism to recover it.

We do **not** reproduce the paper's second finding in this row, that "quality
also drops off with too many heads". Under the paired comparison, `h` = 16
(-0.001) and `h` = 32 (-0.003) are **indistinguishable from the baseline** --
both deltas are two orders of magnitude smaller than the baseline seed range.
An earlier draft, comparing against the 3-seed mean rather than the paired
seed-1337 baseline, reported `h` = 16 as "our best setting"; that claim is
**retracted** as an artifact of baseline choice.

**Stated plainly: with `h` = 32 now measured, we do not observe the paper's
drop-off at high head counts.** Loss is flat from `h` = 8 through `h` = 32
(4.8349, 4.8338, 4.8317 -- a total spread of 0.003 against a baseline seed
range of 0.058). This is a negative result, not an unresolved one: at this
scale and horizon the effect is absent, where the paper reports `h` = 32 as
0.4 BLEU worse than its best setting.

Two reasons we would not expect to see it here, both of which limit how far
this negative result generalizes. First, the paper observes the drop-off at
`h` = 32 with `d_model` = 512, i.e. `d_k` = 16; our `h` = 32 at
`d_model` = 128 gives `d_k` = 4, a substantially different regime despite the
matching head count -- the paper's own row (B) shows that shrinking `d_k` is
itself harmful, so our high-`h` arms confound two variables that are separable
at `d_model` = 512. Second, 400 steps is far too short for a
capacity/resolution effect to express itself. The defensible summary: **we
reproduce the `h` = 1 penalty; we measure no penalty at `h` = 16 or `h` = 32,
but our reduced `d_model` makes this a weak test of the paper's claim rather
than a contradiction of it.**

### 8.3 Positional encoding (Table 3 row E, and removal)

Row (E) of the paper compares learned embeddings against sinusoids. Removing
positional information entirely is **not** a paper experiment; we include it
only as a diagnostic, clearly marked, because it makes the permutation-
equivariance property concrete.

| Variant | in paper? | val loss | `d loss` (paired) | exceeds seed range? | val BLEU |
|---|---|---|---|---|---|
| Sinusoidal (baseline) | yes | 4.8349 | -- | -- | 6.29 |
| Learned embeddings (row E) | yes | 4.6776 | -0.157 | yes | 7.25 |
| None (diagnostic, not a paper condition) | no | 4.6663 | -0.169 | yes | 6.18 |

This is our most counter-intuitive result and we report it as measured:
**at 400 steps, removing positional encoding entirely improves validation
loss.** It would be easy to present this as a contradiction of the paper. It
almost certainly is not, and the reason is instructive.

Our test suite checks, and finds consistent to numerical tolerance, that
self-attention without a position signal is **permutation-equivariant** (this
is also provable analytically; the test verifies our implementation has the
property, it does not constitute the proof) -- the model provably cannot distinguish
"dog bites man" from "man bites dog". A permutation-invariant model therefore
*cannot* be a good translator at convergence. What it can be is a good
*bag-of-words* translator early, and that is what 400 steps measures. On
Multi30k captions (mean length 17, largely monotone EN->DE word order) most of
the early loss reduction comes from learning lexical correspondence, for which
position is irrelevant; meanwhile the sinusoidal signal is a large fixed
perturbation added to every embedding that the model must learn to disentangle
before it becomes useful. Removing it makes the early optimization problem
easier and the asymptotic problem impossible.

The learned-embedding result is consistent with this reading: at 400 steps
learned position embeddings are still close to their small initialization, so
"learned" behaves almost identically to "none" (4.6776 vs 4.6663, a difference
of 0.011 -- well inside the baseline seed range and therefore
indistinguishable). The paper's row (E) claim that learned and
sinusoidal give "nearly identical results" concerns *converged* models and is
not tested by this experiment in either direction.

This is the clearest case in the study of a **horizon artifact**, and it is the
main reason Section 9 warns against reading any 400-step ranking as a verdict
on Table 3.

### 8.4 LayerNorm placement and warmup: a 2x2 factorial

The paper specifies post-norm (Section 3.1) while tensor2tensor implements
pre-norm; the paper also specifies 4,000 warmup steps without explaining what
they are for. Crossing the two factors tests both at once.

| | warmup = 160 | warmup = 1 | `d loss` from removing warmup |
|---|---|---|---|
| **Post-norm (paper text)** | 4.8349 / BLEU 6.29 | 6.6017 / BLEU **0.00** | **+1.767** |
| **Pre-norm (tensor2tensor)** | **4.3400** / BLEU **8.93** | 5.5707 / BLEU 0.81 | +1.231 |

For scale, the baseline seed range is 0.058 in loss; these effects are one to
two orders of magnitude larger. That is a statement about effect magnitude, not
a significance test -- each cell is a single run.

Three findings, all far outside the noise floor:

1. **Warmup is load-bearing, and catastrophically so for the formulation the
   paper describes.** Removing it collapses post-norm training to validation
   perplexity 526 and **BLEU 0.00** -- the model produces nothing usable. This
   is the strongest ablation result in the study and it validates a paper
   design choice that the paper itself never justifies.
2. **Pre-norm is better in both warmup conditions** (-0.495 in loss with warmup, ~8.5x the baseline seed range).
   At this horizon the authors' released code is a better architecture than
   their paper's text, which is a real argument for reproducing from code as
   well as from text.
3. **Pre-norm partially rescues no-warmup but does not eliminate the need for
   it** (+1.231 vs +1.767). The direction matches Xiong et al. (2020), who
   showed post-norm has large output-layer gradients at initialization and
   requires warmup while pre-norm does not; our Level 2a measurement of a
   gradient norm of 7.95 at initialization, decaying 8x within 25 steps, is the
   same phenomenon observed directly. That pre-norm still degrades badly here
   suggests our peak learning rate (6.99e-3, ten times the paper's) is beyond
   what either layout tolerates unwarmed.

### 8.5 Dropout (Table 3 row D)

| `P_drop` | val loss | `d loss` (paired) | exceeds seed range? | val BLEU |
|---|---|---|---|---|
| 0.0 | 4.6989 | -0.136 | yes | 5.97 |
| **0.1 (baseline)** | **4.8349** | -- | -- | **6.29** |

Removing dropout **improves** validation loss at 400 steps, apparently
contradicting the paper's "dropout is very helpful in avoiding over-fitting".
It does not contradict it: 400 steps on 29,000 sentence pairs is roughly 0.8
epochs, so the model has not yet seen most of the training data once and there
is nothing to overfit. Dropout costs optimization speed immediately and repays
it only after overfitting begins. We predicted this inversion in advance
(Section 9, Limitation 2) precisely because it is the expected signature of a
regularization ablation run at too short a horizon, and we report it as
evidence about our experimental design rather than about the paper.

### 8.6 Warmup-to-budget scaling (unplanned, from a failed run)

Our first main run preserved the paper's warmup-to-budget *ratio* (4%), giving
warmup = 64 at `d_model` = 128. It trained badly, and diagnosing why produced
the study's most transferable finding. Because equation (3) sets peak learning
rate to `d_model^-0.5 * warmup^-0.5`:

| Configuration | warmup | peak LR | vs paper |
|---|---|---|---|
| Paper base (`d_model` 512) | 4,000 | 6.99e-4 | 1.0x |
| Scaled, 4% of budget | 64 | 1.105e-2 | **15.8x** |
| Scaled, used here | 160 | 6.99e-3 | 10.0x |
| Scaled, to match paper's peak LR | **16,000** | 6.99e-4 | 1.0x |

Matching the paper's peak learning rate at `d_model` = 128 would require a
warmup **longer than the entire affordable budget** -- the schedule cannot
simultaneously preserve peak magnitude and warmup shape under width reduction.
The empirical cost was large: the warmup = 64 run needed **2,400 steps** to
reach a validation loss (4.648) that the warmup = 160 run reached by **step
650** (4.609), a 3.7x sample-efficiency penalty, and its BLEU was still 2.08
at 2,400 steps against 5.64 at 650 steps for the better schedule. Inspecting
its output showed the classic failure: fluent German that never emits EOS and
degenerates into repetition, with length ratios of 2.8-5.2.

The retained run (`obs_warmup64`) is included in the results tables. The
practical lesson for anyone scaling this paper down: **`warmup_steps` is not a
free knob to rescale, because eq. 3 entangles it with peak learning rate.**
Either hold peak LR fixed by decoupling it (the `factor` argument in our
scheduler) or accept a higher peak and verify empirically -- but do not assume
that preserving the ratio preserves the schedule.

### 8.7 Ablation summary

All arms n = 1, seed 1337. Deltas are paired against the seed-1337 baseline
(4.8349). Baseline seed range = 0.058 in loss.

| Variant | val loss | `d loss` (paired) | exceeds seed range? | interpretation |
|---|---|---|---|---|
| `h` = 1 | 4.9515 | +0.117 | yes | **reproduces paper** |
| `h` = 4 | 4.8725 | +0.038 | no | inconclusive |
| `h` = 16 | 4.8338 | -0.001 | no | null result |
| `h` = 32 | 4.8317 | -0.003 | no | null result; **no drop-off observed** (weak test -- see 8.2) |
| Learned position emb. (row E) | 4.6776 | -0.157 | yes | untestable at this horizon |
| No position encoding (diagnostic) | 4.6663 | -0.169 | yes | horizon artifact |
| `P_drop` = 0.0 | 4.6989 | -0.136 | yes | horizon artifact |
| Pre-norm | 4.3400 | -0.495 | yes | contradicts paper text, matches paper's code |
| No warmup (post-norm) | 6.6017 | +1.767 | yes | **reproduces paper's design choice** |
| No warmup (pre-norm) | 5.5707 | +0.736 | yes | **consistent with Xiong et al.** |

Of ten ablations: three support a paper claim (`h` = 1 penalty, warmup
necessity twice over), two are diagnosable horizon artifacts, one contradicts
the paper text while matching the authors' code, and four are null or
inconclusive. We regard that as the honest yield of a 400-step, single-seed
grid -- and as an argument that the seed study was the single most valuable
experiment in this section.

---

## 9. Limitations

Stated plainly, because the compute constraints here are severe enough that
several standard claims a reproduction would want to make are simply not
available.

1. **No claim about the paper's headline results is tested.** We did not train
   to convergence on WMT14, so this study reproduces the *architecture and
   method*, not the 27.3/28.4 BLEU results. The correct reading is: the
   Transformer as specified in the paper is implementable from the text,
   trains stably under the paper's optimizer and schedule, and learns
   translation -- while the specific numbers remain untested here.
2. **Ablations are underpowered and short-horizon.** With 400-step runs, one
   seed per configuration, and a measured seed spread reported alongside every
   delta, only large effects are detectable. More importantly, a 400-step
   ranking need not survive to 100k steps: regularization ablations in
   particular are expected to *invert*, since dropout costs training speed
   early and pays off only once overfitting begins. Any ablation result here
   that agrees with the paper's Table 3 should be treated as encouraging, not
   confirmatory; any that disagrees is more likely a horizon artifact than a
   contradiction.
3. **The corpus is a poor proxy for the task.** Multi30k captions are short and
   domain-narrow; conclusions about long-range dependency modelling -- the
   paper's central motivation -- are essentially untestable at mean length ~17
   tokens.
4. **Single-seed main run.** The headline scaled BLEU has no error bar. The
   seed study quantifies noise at 400 steps but was not repeated at 1,600.
5. **BLEU is not one number.** The paper does not state its scorer; the
   tensor2tensor scorer of that era was tokenized and runs roughly 1 BLEU
   optimistic against detokenized `sacrebleu`. We report our from-scratch
   implementation and `sacrebleu` side by side rather than choosing.
6. **The big model was never instantiated for training**, so nothing about the
   base/big comparison is tested; only its parameter count is verified
   analytically.
7. **No fp16/bf16, no multi-device, no KV-cache in decoding.** Greedy and beam
   decoding recompute the full prefix at each step, which is `O(L^2)` work
   where a cache gives `O(L)`. This inflates our decode timings and is an
   implementation limitation, not a property of the architecture.
8. **Resumed runs do not have byte-identical batch order** to uninterrupted
   ones (model, both Adam moments, and scheduler step are restored exactly;
   the data stream restarts at the epoch implied by the step count).

---

## 10. Future work

Ordered by value per unit of additional compute.

1. **The cheapest missing experiment is more seeds, not more steps.** At this
   scale the seed spread is a substantial fraction of every ablation delta.
   Three seeds per configuration would cost 3x and would convert most of
   Section 8 from suggestive to conclusive.
2. **Run the ablation grid at two horizons** (e.g. 400 and 4,000 steps) to
   directly measure which conclusions are horizon artifacts. This tests the
   Limitation-2 concern rather than merely noting it, and would specifically
   check whether the dropout ordering inverts as predicted.
3. **A single GPU would make Level 2a meaningful.** The base architecture at
   ~400x speedup becomes ~0.4 s/step at a realistic batch size; 100k steps on
   Multi30k or IWSLT becomes a matter of hours, which would let the *paper's
   exact model* be trained to convergence on a real corpus.
4. **Test the `n < d` boundary directly.** Table 1 implies self-attention's
   advantage is regime-dependent. Sweeping sequence length across `d_model`
   and measuring wall-clock per token against a recurrent baseline would turn
   the paper's asymptotic argument into a measured crossover point -- an
   experiment the paper does not report.
5. **Attention-map analysis.** The paper claims heads specialize on syntactic
   and semantic structure. `MultiHeadAttention.store_attention` already
   captures the maps; what is missing is a quantitative probe (e.g. agreement
   between head attention and dependency arcs) rather than the qualitative
   figures the paper offers.
6. **Resolve the Table 3 parameter-count discrepancy** by counting parameters
   in the tensor2tensor `transformer_base` / `transformer_big` graphs directly,
   which would establish whether the difference is vocabulary, accounting, or
   a typo.
7. **Sinusoid extrapolation, actually tested.** The paper's stated reason for
   preferring sinusoids over learned embeddings is extrapolation to unseen
   lengths, and it does not test this. Training with a length cap and
   evaluating beyond it -- comparing sinusoidal against learned -- is a small,
   well-defined experiment that would settle a claim the paper leaves open.


---

## 11. Threats to validity

Separated from Limitations because these concern whether the *inferences* hold,
not what was left undone.

**Construct validity.** Our primary ablation metric is the label-smoothed
training objective evaluated on validation data, not the unsmoothed perplexity
the paper's Table 3 reports. The two are monotonically related for a fixed
smoothing rate, so rankings are comparable *across our own arms*, but our
ablation losses should not be read against Table 3's PPL column. Perplexities
quoted in Section 6 are computed from unsmoothed cross-entropy and are the
comparable quantity.

**Internal validity.** (i) Ablation arms are n = 1; the seed study bounds
baseline variance but not each arm's. (ii) The `obs_warmup64` comparison
underpinning Section 8.6 is confounded: the two runs differ in resume history
and therefore in batch ordering, and its `TrainConfig.warmup` field drifted
from the scheduler-restored value during chunked execution. It should be read
as a strong hint, not a controlled experiment. (iii) Resumed runs do not
reproduce batch order exactly.

**External validity.** Multi30k captions (mean length ~17 tokens, narrow
domain) cannot test the long-range-dependency modelling that motivates the
paper. Conclusions transfer to WMT-scale training only by assumption.

**Conclusion validity.** No significance tests, confidence intervals, or
multiple-comparison corrections are reported anywhere in this study. Every
comparative statement should be read as descriptive.

**Measurement validity.** Timing was performed on a shared vCPU with 1.70x
observed throughput variation; the per-step benchmark is extrapolated from
synthetic fixed-length micro-batches. Memory is process RSS, which includes
roughly 900 MiB of interpreter and framework baseline, so reported peaks
overstate model memory.

---

## 12. Conclusion

We reimplemented the Transformer from the paper text and verified it against
98 checks, then trained it under the paper's optimizer, schedule, loss,
regularization, tokenization, batching, and inference procedure at roughly
1/3,600 of the original token budget on a single CPU core. The architecture is
reproducible from the paper alone; the reported *results* are not reachable on
this hardware, and we quantified that gap (~400x slower; ~185 projected days
for the base budget; the big model infeasible on memory grounds) rather than
asserting it.

Three claims we would defend: removing learning-rate warmup collapses the
paper's post-norm architecture to BLEU 0.00, supplying the empirical
justification the paper omits; single-head attention is the only head setting
that measurably hurts, reproducing Table 3 row (A)'s central result; and
checkpoint averaging contributes +0.57 BLEU, inside the paper's own stated
0.3-0.6 range. Three we retract or qualify: `h` = 16 is not "our best setting"
(a baseline-selection artifact); beam search does not underperform greedy (a
subset-size artifact); and the parameter-count discrepancy is a
non-reproducibility under our assumptions rather than an arithmetic error in
the paper.

The most transferable lesson is negative and concerns scaling the recipe rather
than the architecture: equation (3) couples peak learning rate to both
`d_model` and `warmup_steps`, so preserving the paper's warmup-to-budget ratio
while shrinking the model silently inflates the peak learning rate -- 15.8x in
our case, costing roughly a 4x sample-efficiency penalty before we diagnosed
it.

---

## 13. Internal audit and errata

This report was subjected to a hostile self-review before release. The audit
examined saved artifacts rather than prose, and found errors that the prose
alone would not have revealed. All are corrected above; they are listed here
because an errata trail is more informative than a silently clean document.

| # | Error found | Status |
|---|---|---|
| 1 | The headline compute figures (408x, 189 days) had **no saved artifact** -- the benchmarking process was OOM-killed during the `big` config before writing `benchmark.json`, so the table was also absent from generated `results.md`. | **Fixed.** Each config now times in an isolated subprocess over 5 repeats; `benchmark.json` is written regardless of OOM. Restated as ~400x (386-419) and ~185 days (179-194). |
| 2 | Checkpoint averaging was described as used but the code path was **unreachable**: `avg_model` was constructed and then never passed to the evaluation loop, and `save_every=0` meant no periodic checkpoints existed. | **Fixed.** Averaging now runs over 5 checkpoints and is reported as its own row; it is the headline number. |
| 3 | Figure 5 was captioned "1200 steps"; the ablations ran for **400**. Hardcoded string. | **Fixed;** caption now derived from the run summaries. |
| 4 | Ablation effect sizes were computed against a 3-seed *mean* while every arm used seed 1337. Recomputing paired against seed 1337 **changed a conclusion**: `h` = 16 moved from "our best setting" to a null result. | **Fixed and retracted.** |
| 5 | "sigma" notation implied significance testing that was never performed (sd from n = 3, arms n = 1, 9 comparisons, no correction). | **Fixed.** Raw paired deltas only. |
| 6 | Abstract claimed the learning curve was "still improving monotonically"; BLEU was **not** monotonic (15.42 -> 14.69 -> 16.33). | **Fixed;** monotonicity now claimed only for validation loss, which does hold. |
| 7 | Headline compared our *greedy* BLEU against the paper's *beam-4 + averaged* BLEU. | **Fixed.** Decoding is now matched; 19.74 vs 27.3. |
| 8 | Beam search reported as worse than greedy -- an artifact of 150 vs 1,000 sentences. | **Retracted.** On matched full test sets beam 4 beats greedy by 0.65. |
| 9 | Two different throughputs for "the same architecture" (153 vs 308 tokens/s) were never reconciled; 330 and "~3.1 s/step" were quoted where 308 and 3.33 were correct. | **Fixed and reconciled** (vocabulary size + synthetic vs real batches). |
| 10 | "Verified", "proves", "exactly", "succeeds completely", "confirmed empirically" overstated what the evidence supports; ~99 of 107 tests compare our code to our own transcription. | **Fixed** throughout; test-independence caveat stated in the abstract. |
| 11 | Training time "19.3 min" presented as measured; it is derived and excludes per-chunk startup. | **Fixed;** labelled as a derived compute estimate. |
| 12 | Figure 2 plotted the paper's 27.3 WMT14 line on a Multi30k axis, implying a comparability the text denies. | **Fixed;** reference line removed, caption states non-comparability. |

Two audit findings are acknowledged but **not** fixed, for compute reasons, and
are recorded as open: ablation arms remain n = 1 (Section 11), and the
`obs_warmup64` comparison remains confounded (Section 11).

---

## 14. Reproducibility checklist

| Item | Status |
|---|---|
| Source code, all modules | included |
| Random seeds stated (1337; 7, 42 for variance) | yes |
| Dependency versions (torch 2.13.0+cu130, sacrebleu 2.6.0, numpy 2.4.4, matplotlib 3.10.8, Python 3.12.3) | yes |
| Hardware (1x Xeon vCPU, 3.9 GiB RAM, no GPU, `torch.get_num_threads()` = 1) | yes |
| Exact training commands | `run_all.sh` and README |
| Per-run config, logs, summaries | `experiments/runs/<name>/{config.json,log.jsonl,summary.json}` |
| Data acquisition + preprocessing script | `prepare_data.py`; tokenizer serialized to `bpe.json` |
| Evaluation procedure and scorer | `evaluation/bleu.py`, cross-checked against sacrebleu |
| Model checkpoints | omitted from the bundle for size; regenerable |
| Git commit hash | **missing** -- not under version control in this environment |
| Dependency lockfile with hashes | **missing** |
| Statistical tests / confidence intervals | **not performed** (Section 11) |
