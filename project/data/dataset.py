"""Parallel dataset, right-shifting, and token-count dynamic batching.

Paper Section 5.1: "Sentence pairs were batched together by approximate
sequence length. Each training batch contained a set of sentence pairs
containing approximately 25000 source tokens and 25000 target tokens."

This is the detail most often quietly dropped in reimplementations, which
batch by a fixed *sentence* count instead. It matters for two reasons:

  * Compute efficiency. Padding waste is O(max_len - mean_len) per batch;
    sorting by length before bucketing keeps batches nearly rectangular.
  * Optimization. A fixed sentence count makes the number of tokens per
    gradient step vary by an order of magnitude between short and long
    batches, so the effective learning rate per token fluctuates. A fixed
    *token* count keeps gradient noise roughly constant, which is what the
    single global LR schedule of equation (3) implicitly assumes.

Right shift (Section 3.1, "the output embeddings are offset by one
position"):

    target sequence y = [y_1, ..., y_m, EOS]
    decoder input     = [BOS, y_1, ..., y_m]
    decoder target    = [y_1, ..., y_m, EOS]

so that predicting position i sees only positions < i. Both have the same
length; BOS is prepended and EOS is consumed as the final prediction.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

import torch

from .preprocessing import BOS_ID, EOS_ID, PAD_ID, BPETokenizer


@dataclass
class Example:
    src: List[int]      # source ids, ends with EOS
    tgt: List[int]      # target ids, ends with EOS (no BOS; added at batch time)

    @property
    def n_src(self) -> int:
        return len(self.src)

    @property
    def n_tgt(self) -> int:
        # Decoder input is [BOS] + tgt[:-1], i.e. the SAME length as tgt:
        # BOS is prepended and the final EOS is consumed as a prediction
        # target rather than fed in. Adding 1 here would over-reserve one
        # column of padding per batch.
        return len(self.tgt)


class ParallelCorpus:
    """Holds an encoded parallel corpus in memory."""

    def __init__(self, examples: Sequence[Example]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> Example:
        return self.examples[i]

    @classmethod
    def from_files(
        cls,
        src_path: str | Path,
        tgt_path: str | Path,
        tokenizer: BPETokenizer,
        max_len: int = 100,
        limit: int | None = None,
        drop_empty: bool = True,
    ) -> "ParallelCorpus":
        """Encode a sentence-aligned file pair.

        Filtering follows standard NMT practice (the paper does not specify
        its own length filter): pairs longer than `max_len` BPE tokens on
        either side are dropped from *training* data. Applying the same
        filter to test data would be scientifically invalid, so callers pass
        max_len=None-equivalent (a large value) for evaluation sets.
        """
        src_lines = Path(src_path).read_text(encoding="utf-8").splitlines()
        tgt_lines = Path(tgt_path).read_text(encoding="utf-8").splitlines()
        if len(src_lines) != len(tgt_lines):
            raise ValueError(f"unaligned corpus: {len(src_lines)} vs {len(tgt_lines)} lines")
        if limit is not None:
            src_lines, tgt_lines = src_lines[:limit], tgt_lines[:limit]

        ex: List[Example] = []
        for s, t in zip(src_lines, tgt_lines):
            si = tokenizer.encode(s, add_eos=True)
            ti = tokenizer.encode(t, add_eos=True)
            if drop_empty and (len(si) <= 1 or len(ti) <= 1):
                continue
            if len(si) > max_len or len(ti) > max_len:
                continue
            ex.append(Example(si, ti))
        return cls(ex)

    def token_stats(self) -> dict:
        s = [e.n_src for e in self.examples]
        t = [len(e.tgt) for e in self.examples]
        return {
            "pairs": len(self.examples),
            "src_tokens": sum(s),
            "tgt_tokens": sum(t),
            "src_mean_len": (sum(s) / len(s)) if s else 0.0,
            "tgt_mean_len": (sum(t) / len(t)) if t else 0.0,
            "src_max_len": max(s) if s else 0,
            "tgt_max_len": max(t) if t else 0,
        }


def collate(batch: Sequence[Example], pad_id: int = PAD_ID) -> dict:
    """Pad a list of Examples into tensors.

    Returns
    -------
    src    : (B, Ls) source ids, PAD-filled
    tgt_in : (B, Lt) decoder input  = [BOS] + tgt[:-1]... (right-shifted)
    tgt_out: (B, Lt) decoder target = tgt
    n_tokens : number of non-PAD target tokens (the loss denominator)
    """
    B = len(batch)
    Ls = max(e.n_src for e in batch)
    Lt = max(len(e.tgt) for e in batch)

    src = torch.full((B, Ls), pad_id, dtype=torch.long)
    tgt_in = torch.full((B, Lt), pad_id, dtype=torch.long)
    tgt_out = torch.full((B, Lt), pad_id, dtype=torch.long)

    for i, e in enumerate(batch):
        src[i, : len(e.src)] = torch.tensor(e.src, dtype=torch.long)
        shifted = [BOS_ID] + e.tgt          # (BOS, y_1, ..., y_m, EOS)
        tgt_in[i, : len(shifted) - 1] = torch.tensor(shifted[:-1], dtype=torch.long)
        tgt_out[i, : len(e.tgt)] = torch.tensor(e.tgt, dtype=torch.long)

    return {
        "src": src,
        "tgt_in": tgt_in,
        "tgt_out": tgt_out,
        "n_tokens": int((tgt_out != pad_id).sum()),
        "n_src_tokens": int((src != pad_id).sum()),
        "n_sentences": B,
    }


class TokenBatchSampler:
    """Groups examples into batches of ~max_tokens, bucketed by length.

    Procedure (following tensor2tensor's bucket-by-sequence-length):
      1. Shuffle, then sort within a large pool by (tgt_len, src_len) so
         that batches are length-homogeneous but epoch order still varies.
      2. Greedily fill a batch while
             max_len_so_far * (batch_size + 1) <= max_tokens
         on BOTH source and target side, which bounds the *padded* token
         count -- the quantity that actually determines memory and FLOPs.
      3. Shuffle the resulting batch order so consecutive steps do not see
         monotonically increasing lengths.
    """

    def __init__(
        self,
        corpus: ParallelCorpus,
        max_tokens: int = 25000,
        pool_multiplier: int = 100,
        shuffle: bool = True,
        seed: int = 1337,
        drop_last: bool = False,
    ) -> None:
        self.corpus = corpus
        self.max_tokens = max_tokens
        self.pool_multiplier = pool_multiplier
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.drop_last = drop_last

    def _batches(self) -> List[List[int]]:
        idx = list(range(len(self.corpus)))
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(idx)

        # Approximate global length sort via large pools: full sorting would
        # make epoch order deterministic, pools keep some stochasticity.
        pool = max(1, self.pool_multiplier) * 64
        batches: List[List[int]] = []
        for start in range(0, len(idx), pool):
            chunk = sorted(
                idx[start : start + pool],
                key=lambda i: (len(self.corpus[i].tgt), self.corpus[i].n_src),
            )
            cur: List[int] = []
            max_s = max_t = 0
            for i in chunk:
                e = self.corpus[i]
                ns, nt = max(max_s, e.n_src), max(max_t, e.n_tgt)
                if cur and (ns * (len(cur) + 1) > self.max_tokens
                            or nt * (len(cur) + 1) > self.max_tokens):
                    batches.append(cur)
                    cur, max_s, max_t = [i], e.n_src, e.n_tgt
                else:
                    cur.append(i)
                    max_s, max_t = ns, nt
            if cur and not (self.drop_last and len(cur) < 2):
                batches.append(cur)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        yield from self._batches()

    def __len__(self) -> int:
        return len(self._batches())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class DataLoader:
    """Minimal loader: TokenBatchSampler + collate. No worker processes
    (the container has a single CPU core, so workers would only add
    overhead)."""

    def __init__(self, corpus: ParallelCorpus, sampler: TokenBatchSampler, pad_id: int = PAD_ID):
        self.corpus, self.sampler, self.pad_id = corpus, sampler, pad_id

    def __iter__(self) -> Iterator[dict]:
        for batch_idx in self.sampler:
            yield collate([self.corpus[i] for i in batch_idx], self.pad_id)

    def __len__(self) -> int:
        return len(self.sampler)


def infinite_loader(loader: DataLoader, start_epoch: int = 0) -> Iterator[dict]:
    """Cycle a loader forever, advancing the sampler epoch each pass.

    Training is measured in *steps* in the paper (100k / 300k), not epochs,
    so the training loop consumes an endless stream.
    """
    epoch = start_epoch
    while True:
        loader.sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


__all__ = [
    "Example", "ParallelCorpus", "collate", "TokenBatchSampler",
    "DataLoader", "infinite_loader",
]
