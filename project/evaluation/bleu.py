"""Corpus-level BLEU, implemented from scratch (Papineni et al., 2002).

The paper reports BLEU on newstest2014 but does not state which scorer was
used; the tensor2tensor codebase of that era computed a tokenized BLEU
(`compute_bleu` on top of its own tokenizer), which is known to be
optimistic by roughly 1 BLEU relative to detokenized `sacrebleu`. Because
"BLEU" is not a single well-defined number, this module

  1. implements the original definition explicitly, and
  2. cross-checks it against `sacrebleu` on the same data
     (tests/test_bleu.py), reporting both in the results table.

Definition
----------
Modified n-gram precision, aggregated over the corpus:

    p_n = sum_C sum_{ngram in C} Count_clip(ngram)
          -----------------------------------------
          sum_C sum_{ngram in C} Count(ngram)

where Count_clip clips a candidate n-gram count to the maximum count of
that n-gram in any reference. Brevity penalty against the effective
reference length r (closest reference length, summed over the corpus) and
candidate length c:

    BP = 1                if c > r
       = exp(1 - r/c)     if c <= r

    BLEU = BP * exp( sum_{n=1..N} w_n * log p_n ),   w_n = 1/N, N = 4

Corpus-level aggregation (not a per-sentence average) is essential: the
numerators and denominators are summed over the whole corpus *before*
the ratio is taken. Sentence-averaged BLEU is a different, higher number.

Zero-count handling: if any p_n is 0 the geometric mean is 0. That is the
mathematically correct value and is what we return for the primary score;
for diagnostics on very short outputs we also expose smoothed variants.
"""

from __future__ import annotations

import collections
import math
import re
from typing import Dict, List, Sequence, Tuple

# The "13a" tokenization used by mteval-v13a / sacrebleu, reimplemented so
# that our from-scratch score is comparable to the standard one.
_TOK_PUNCT = re.compile(r"([\{-\~\[-\` -\&\(-\+\:-\@\/])")
_TOK_PERIOD_COMMA_PRE = re.compile(r"([^0-9])([\.,])")
_TOK_PERIOD_COMMA_POST = re.compile(r"([\.,])([^0-9])")
_TOK_DASH = re.compile(r"([0-9])(-)")


def tokenize_13a(line: str) -> List[str]:
    """mteval-v13a tokenization: separate punctuation, keep numeric commas."""
    line = line.replace("<skipped>", "").replace("-\n", "").replace("\n", " ")
    line = line.replace("&quot;", '"').replace("&amp;", "&")
    line = line.replace("&lt;", "<").replace("&gt;", ">")
    line = _TOK_PUNCT.sub(r" \1 ", line)
    line = _TOK_PERIOD_COMMA_PRE.sub(r"\1 \2 ", line)
    line = _TOK_PERIOD_COMMA_POST.sub(r" \1 \2", line)
    line = _TOK_DASH.sub(r"\1 \2 ", line)
    return line.split()


def ngram_counts(tokens: Sequence[str], n: int) -> collections.Counter:
    return collections.Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


class BLEUResult(dict):
    """Dict subclass so results are JSON-serializable but attribute-friendly."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __str__(self) -> str:
        p = "/".join(f"{100 * x:.1f}" for x in self["precisions"])
        return (f"BLEU = {self['bleu']:.2f}  {p}  "
                f"(BP = {self['bp']:.3f}, ratio = {self['ratio']:.3f}, "
                f"hyp_len = {self['hyp_len']}, ref_len = {self['ref_len']})")


def corpus_bleu(
    hypotheses: Sequence[str],
    references: Sequence[Sequence[str]] | Sequence[str],
    max_n: int = 4,
    tokenize: bool = True,
    smooth: str = "none",
    effective_order: bool = False,
) -> BLEUResult:
    """Corpus BLEU in [0, 100].

    Parameters
    ----------
    hypotheses : list of detokenized system outputs
    references : list of reference strings, or list of lists for multi-reference
    smooth : "none" | "floor" | "add-k" | "exp"
        Only affects zero-precision cases. "none" is the original definition.
    effective_order : bool
        If True, orders for which the corpus contains NO n-grams at all
        (total == 0, e.g. 4-grams in a corpus of 3-token sentences) are
        dropped from the geometric mean instead of forcing BLEU to 0. This
        is distinct from smoothing, which handles total > 0 but match == 0.
        Corpus-level BLEU on realistic data never needs it; it matters only
        for very short or few sentences. Default False = original definition.
    """
    if len(hypotheses) != len(references):
        raise ValueError(f"{len(hypotheses)} hypotheses vs {len(references)} references")
    refs: List[Sequence[str]] = [[r] if isinstance(r, str) else list(r) for r in references]

    matches = [0] * max_n
    totals = [0] * max_n
    hyp_len = 0
    ref_len = 0

    for hyp, ref_set in zip(hypotheses, refs):
        h_tok = tokenize_13a(hyp) if tokenize else hyp.split()
        r_toks = [tokenize_13a(r) if tokenize else r.split() for r in ref_set]

        hyp_len += len(h_tok)
        # Effective reference length: the reference length closest to the
        # candidate length (Papineni Section 2.2.2). Ties -> shorter.
        ref_len += min(((abs(len(r) - len(h_tok)), len(r)) for r in r_toks))[1]

        for n in range(1, max_n + 1):
            h_counts = ngram_counts(h_tok, n)
            if not h_counts:
                continue
            max_ref: collections.Counter = collections.Counter()
            for r in r_toks:
                rc = ngram_counts(r, n)
                for g, c in rc.items():
                    if c > max_ref[g]:
                        max_ref[g] = c
            clipped = sum(min(c, max_ref[g]) for g, c in h_counts.items())
            matches[n - 1] += clipped
            totals[n - 1] += sum(h_counts.values())

    precisions: List[float] = []
    for n in range(max_n):
        m, t = matches[n], totals[n]
        if t == 0:
            precisions.append(0.0)
        elif m == 0 and smooth == "floor":
            precisions.append(1e-9 / t)
        elif m == 0 and smooth == "exp":
            precisions.append(1.0 / (2 ** (n + 1) * t))
        elif smooth == "add-k":
            precisions.append((m + 1) / (t + 1))
        else:
            precisions.append(m / t)

    if effective_order:
        used = [p for p, t in zip(precisions, totals) if t > 0]
    else:
        used = precisions
    if used and min(used) > 0:
        geo = math.exp(sum(math.log(p) for p in used) / len(used))
    else:
        geo = 0.0

    if hyp_len == 0:
        bp = 0.0
    elif hyp_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / hyp_len)

    return BLEUResult(
        bleu=100.0 * bp * geo,
        precisions=precisions,
        bp=bp,
        ratio=(hyp_len / ref_len) if ref_len else 0.0,
        hyp_len=hyp_len,
        ref_len=ref_len,
        matches=matches,
        totals=totals,
        smooth=smooth,
        effective_order=effective_order,
        orders_used=len(used),
    )


def sacrebleu_score(hypotheses: Sequence[str], references: Sequence[str]) -> float | None:
    """Reference implementation cross-check. Returns None if unavailable."""
    try:
        import sacrebleu
    except ImportError:
        return None
    return float(sacrebleu.corpus_bleu(list(hypotheses), [list(references)]).score)


__all__ = ["corpus_bleu", "sacrebleu_score", "tokenize_13a", "ngram_counts", "BLEUResult"]
