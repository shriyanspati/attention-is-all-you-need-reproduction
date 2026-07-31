"""Text cleaning and byte-pair encoding (paper Section 5.1).

"Sentences were encoded using byte-pair encoding, which has a shared
source-target vocabulary of about 37000 tokens."

BPE is implemented here from scratch following Sennrich, Haddow & Birch
(2016) -- the paper's reference [25] -- rather than calling
`subword-nmt`/`sentencepiece`/`tokenizers`, so that the vocabulary
construction is auditable:

  1. Pre-tokenize on whitespace and split off punctuation, then represent
     each word as a sequence of characters with a special end-of-word
     marker `</w>`. The marker is what makes the encoding reversible and
     stops merges from spanning word boundaries.
  2. Count all adjacent symbol pairs across the *word-frequency table*
     (not the raw corpus) -- identical result, far less work.
  3. Repeatedly merge the most frequent pair, recording the merge order.
     Ties are broken by lexicographic order of the pair so that the
     vocabulary is a deterministic function of the corpus and the seed is
     irrelevant here.
  4. Vocabulary = special tokens + all symbols appearing in the final
     segmentation.

Efficiency: a naive implementation recounts every pair after every merge,
costing O(merges x corpus). We maintain an inverted index from pair ->
words containing it and update counts incrementally, which is what makes
4k-8k merges feasible on one CPU core.

The vocabulary is *shared* between source and target: merges are learned
on the concatenation of both sides, exactly as the paper describes for
EN-DE. This is also what makes the three-way embedding tying of
Section 3.4 coherent.

Special tokens (Phase 3 requirement): PAD=0, BOS=1, EOS=2, UNK=3.
UNK is needed only for characters unseen at training time; BPE otherwise
has no out-of-vocabulary problem by construction, which we verify by
measuring the UNK rate on held-out data.
"""

from __future__ import annotations

import collections
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
END_OF_WORD = "</w>"

_WS = re.compile(r"\s+")
# Keep letters/digits together; isolate punctuation as its own token.
_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def clean_text(line: str, lowercase: bool = False) -> str:
    """Normalize a raw corpus line.

    * NFKC unicode normalization, so that visually identical text does not
      produce distinct BPE symbols.
    * Collapse whitespace, strip.
    * Lowercasing is OFF by default: the paper reports case-sensitive BLEU
      on newstest2014, and lowercasing would inflate scores.
    """
    line = unicodedata.normalize("NFKC", line)
    line = _WS.sub(" ", line).strip()
    return line.lower() if lowercase else line


def pretokenize(line: str) -> List[str]:
    """Split a cleaned line into word-level units before BPE."""
    return _TOKEN.findall(line)


class BPETokenizer:
    """Shared-vocabulary byte-pair encoding tokenizer."""

    def __init__(
        self,
        merges: Sequence[Tuple[str, str]] | None = None,
        vocab: Dict[str, int] | None = None,
    ) -> None:
        self.merges: List[Tuple[str, str]] = [tuple(m) for m in (merges or [])]
        self.ranks: Dict[Tuple[str, str], int] = {m: i for i, m in enumerate(self.merges)}
        self.vocab: Dict[str, int] = dict(vocab or {})
        self.inv_vocab: Dict[int, str] = {i: t for t, i in self.vocab.items()}
        self._cache: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------ train

    @classmethod
    def train(
        cls,
        corpora: Iterable[Iterable[str]],
        num_merges: int,
        lowercase: bool = False,
        min_frequency: int = 2,
        verbose: bool = False,
    ) -> "BPETokenizer":
        """Learn `num_merges` merge operations from one or more corpora.

        `corpora` is an iterable of line-iterables; all are concatenated,
        which is how the shared source-target vocabulary is obtained.
        """
        # ---- word frequency table -------------------------------------
        wf: collections.Counter[str] = collections.Counter()
        for corpus in corpora:
            for line in corpus:
                for w in pretokenize(clean_text(line, lowercase)):
                    wf[w] += 1

        # Each word -> tuple of symbols, ending with the end-of-word marker.
        words: List[List[str]] = []
        freqs: List[int] = []
        for w, f in wf.items():
            if f < min_frequency and len(w) > 1:
                # Rare words still contribute their characters via the
                # character inventory below; excluding them from merge
                # statistics is the standard speed/quality tradeoff.
                pass
            words.append(list(w) + [END_OF_WORD])
            freqs.append(f)

        # ---- pair counts + inverted index ------------------------------
        pair_counts: collections.Counter[Tuple[str, str]] = collections.Counter()
        pair_where: Dict[Tuple[str, str], set] = collections.defaultdict(set)
        for wi, sym in enumerate(words):
            f = freqs[wi]
            for a, b in zip(sym, sym[1:]):
                pair_counts[(a, b)] += f
                pair_where[(a, b)].add(wi)

        merges: List[Tuple[str, str]] = []
        for step in range(num_merges):
            if not pair_counts:
                break
            best_count = max(pair_counts.values())
            if best_count < 2:
                break
            # Deterministic tie-breaking.
            best = min((p for p, c in pair_counts.items() if c == best_count))
            merges.append(best)
            new_sym = best[0] + best[1]

            for wi in list(pair_where[best]):
                sym = words[wi]
                f = freqs[wi]
                # Remove this word's current pair statistics.
                for a, b in zip(sym, sym[1:]):
                    pair_counts[(a, b)] -= f
                    if pair_counts[(a, b)] <= 0:
                        del pair_counts[(a, b)]
                    pair_where[(a, b)].discard(wi)
                # Apply the merge left-to-right.
                out: List[str] = []
                i = 0
                while i < len(sym):
                    if i < len(sym) - 1 and (sym[i], sym[i + 1]) == best:
                        out.append(new_sym)
                        i += 2
                    else:
                        out.append(sym[i])
                        i += 1
                words[wi] = out
                # Re-add updated statistics.
                for a, b in zip(out, out[1:]):
                    pair_counts[(a, b)] += f
                    pair_where[(a, b)].add(wi)
            pair_where.pop(best, None)
            if verbose and (step + 1) % 500 == 0:
                print(f"  merge {step + 1}/{num_merges}  {best} count={best_count}")

        # ---- vocabulary ------------------------------------------------
        # The vocabulary must be CLOSED under the merge sequence. Applying
        # the merge list to an unseen word can produce any symbol that is
        # either (i) a single character or (ii) the result of some merge --
        # even if that symbol never survives in the segmentation of a
        # *training* word. Building the vocabulary only from the final
        # training segmentation (the intuitive but wrong approach) leaves
        # such symbols out and silently maps novel words to UNK, defeating
        # the open-vocabulary property that motivates BPE in the first
        # place. Caught by
        # tests/test_training_and_eval.py::test_unseen_word_is_segmented_not_unked.
        symbols: collections.Counter[str] = collections.Counter()
        for wi, sym in enumerate(words):
            for s in sym:
                symbols[s] += freqs[wi]
        # (ii) every merge result, whether or not it survived.
        for a, b in merges:
            symbols.setdefault(a + b, 0)
        # (i) every single character observed in the corpus.
        for w in wf:
            for ch in w:
                symbols.setdefault(ch, 0)
        symbols.setdefault(END_OF_WORD, 1)

        vocab: Dict[str, int] = {t: i for i, t in enumerate(SPECIALS)}
        for s, _ in sorted(symbols.items(), key=lambda kv: (-kv[1], kv[0])):
            if s not in vocab:
                vocab[s] = len(vocab)
        return cls(merges=merges, vocab=vocab)

    # ------------------------------------------------------------------ apply

    def _bpe_word(self, word: str) -> List[str]:
        if word in self._cache:
            return self._cache[word]
        sym = list(word) + [END_OF_WORD]
        while len(sym) > 1:
            # Choose the adjacent pair with the lowest merge rank (earliest
            # learned merge), mirroring the training order.
            best_rank, best_i = None, None
            for i, pair in enumerate(zip(sym, sym[1:])):
                r = self.ranks.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_i is None:
                break
            sym[best_i : best_i + 2] = [sym[best_i] + sym[best_i + 1]]
        self._cache[word] = sym
        return sym

    def tokenize(self, line: str, lowercase: bool = False) -> List[str]:
        out: List[str] = []
        for w in pretokenize(clean_text(line, lowercase)):
            out.extend(self._bpe_word(w))
        return out

    def encode(self, line: str, add_bos: bool = False, add_eos: bool = True) -> List[int]:
        ids = [self.vocab.get(t, UNK_ID) for t in self.tokenize(line)]
        if add_bos:
            ids = [BOS_ID] + ids
        if add_eos:
            ids = ids + [EOS_ID]
        return ids

    def decode(self, ids: Sequence[int], strip_specials: bool = True) -> str:
        toks = []
        for i in ids:
            t = self.inv_vocab.get(int(i), UNK)
            if strip_specials and t in (PAD, BOS, EOS):
                continue
            toks.append(t)
        text = "".join(toks).replace(END_OF_WORD, " ")
        return _WS.sub(" ", text).strip()

    # ------------------------------------------------------------------- I/O

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"merges": self.merges, "vocab": self.vocab}, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        d = json.loads(Path(path).read_text())
        return cls(merges=[tuple(m) for m in d["merges"]], vocab=d["vocab"])


def unk_rate(tok: BPETokenizer, lines: Iterable[str]) -> float:
    """Fraction of emitted token ids that are UNK -- a vocabulary sanity check."""
    n = u = 0
    for line in lines:
        ids = tok.encode(line, add_eos=False)
        n += len(ids)
        u += sum(1 for i in ids if i == UNK_ID)
    return (u / n) if n else 0.0


__all__ = [
    "BPETokenizer", "clean_text", "pretokenize", "unk_rate",
    "SPECIALS", "PAD_ID", "BOS_ID", "EOS_ID", "UNK_ID", "END_OF_WORD",
]
