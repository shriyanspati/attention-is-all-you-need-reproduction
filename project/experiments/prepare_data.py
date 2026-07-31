"""Build the shared BPE vocabulary and verify the Phase 3 data pipeline.

Also runs a vocabulary-scaling study: BPE vocabulary size, mean sequence
length and held-out UNK rate as a function of the merge count. This is what
justifies the vocabulary size actually used at reduced scale, instead of
asserting "approximately 37000 tokens" on a corpus that cannot support it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..data.dataset import ParallelCorpus, TokenBatchSampler, collate
from ..data.preprocessing import BPETokenizer, clean_text, pretokenize, unk_rate


def read(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/prepared")
    ap.add_argument("--merges", type=int, default=4000)
    ap.add_argument("--scaling-study", action="store_true")
    ap.add_argument("--max-len", type=int, default=64)
    args = ap.parse_args()

    raw = Path(args.raw)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_en, train_de = read(raw / "train.en"), read(raw / "train.de")
    val_en, val_de = read(raw / "val.en"), read(raw / "val.de")
    test_en, test_de = read(raw / "test_2016_flickr.en"), read(raw / "test_2016_flickr.de")

    report: dict = {"corpus": {
        "train_pairs": len(train_en), "val_pairs": len(val_en), "test_pairs": len(test_en),
        "train_en_words": sum(len(pretokenize(clean_text(l))) for l in train_en),
        "train_de_words": sum(len(pretokenize(clean_text(l))) for l in train_de),
    }}
    types_en = {w for l in train_en for w in pretokenize(clean_text(l))}
    types_de = {w for l in train_de for w in pretokenize(clean_text(l))}
    report["corpus"]["word_types_en"] = len(types_en)
    report["corpus"]["word_types_de"] = len(types_de)
    report["corpus"]["word_types_joint"] = len(types_en | types_de)
    print(json.dumps(report["corpus"], indent=2))

    # ---- vocabulary scaling study ------------------------------------------
    if args.scaling_study:
        study = []
        for m in (1000, 2000, 4000, 8000, 16000):
            t0 = time.time()
            tok = BPETokenizer.train([train_en, train_de], num_merges=m)
            dt = time.time() - t0
            mean_len = sum(len(tok.tokenize(l)) for l in val_en[:300]) / 300
            row = {
                "merges_requested": m,
                "merges_learned": len(tok.merges),
                "vocab_size": tok.vocab_size,
                "mean_val_en_tokens_per_sentence": round(mean_len, 2),
                "val_unk_rate": round(unk_rate(tok, val_en[:300] + val_de[:300]), 6),
                "train_seconds": round(dt, 1),
            }
            study.append(row)
            print(row, flush=True)
        report["vocab_scaling_study"] = study

    # ---- final tokenizer ----------------------------------------------------
    t0 = time.time()
    tok = BPETokenizer.train([train_en, train_de], num_merges=args.merges)
    print(f"trained BPE: {len(tok.merges)} merges, vocab {tok.vocab_size} "
          f"in {time.time() - t0:.1f}s", flush=True)
    tok.save(out / "bpe.json")

    report["tokenizer"] = {
        "merges_requested": args.merges,
        "merges_learned": len(tok.merges),
        "vocab_size": tok.vocab_size,
        "shared_source_target": True,
        "val_unk_rate": round(unk_rate(tok, val_en + val_de), 6),
        "test_unk_rate": round(unk_rate(tok, test_en + test_de), 6),
    }

    # ---- encode corpora -----------------------------------------------------
    splits = {}
    for name, (s, t, ml) in {
        "train": ((raw / "train.en"), (raw / "train.de"), args.max_len),
        "val": ((raw / "val.en"), (raw / "val.de"), 10_000),
        "test": ((raw / "test_2016_flickr.en"), (raw / "test_2016_flickr.de"), 10_000),
    }.items():
        corpus = ParallelCorpus.from_files(s, t, tok, max_len=ml)
        stats = corpus.token_stats()
        stats["length_filter_max_len"] = ml if ml < 1000 else None
        splits[name] = stats
        print(name, json.dumps(stats), flush=True)
    report["splits"] = splits

    # ---- dynamic batching verification -------------------------------------
    corpus = ParallelCorpus.from_files(raw / "train.en", raw / "train.de", tok,
                                       max_len=args.max_len)
    for budget in (1024, 4096, 25000):
        sampler = TokenBatchSampler(corpus, max_tokens=budget, seed=0)
        sizes, pads, tot = [], 0, 0
        for i, idx in enumerate(sampler):
            b = collate([corpus[j] for j in idx])
            sizes.append(b["n_sentences"])
            pads += int((b["tgt_in"] == 0).sum())
            tot += b["tgt_in"].numel()
            if i >= 400:
                break
        report.setdefault("dynamic_batching", []).append({
            "max_tokens": budget,
            "batches_per_epoch": len(sampler),
            "mean_sentences_per_batch": round(sum(sizes) / len(sizes), 1),
            "min_sentences_per_batch": min(sizes),
            "max_sentences_per_batch": max(sizes),
            "padding_fraction": round(pads / tot, 4),
        })
        print(report["dynamic_batching"][-1], flush=True)

    (out / "data_report.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {out / 'data_report.json'}")


if __name__ == "__main__":
    main()
