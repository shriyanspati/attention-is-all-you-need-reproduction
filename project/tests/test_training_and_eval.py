"""Verification of Sections 5.3, 5.4 (optimizer/schedule/regularization),
the BLEU scorer, and the data pipeline of Phase 3.
"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from ..data.dataset import (Example, ParallelCorpus, TokenBatchSampler, collate)
from ..data.preprocessing import (BOS_ID, EOS_ID, PAD_ID, UNK_ID, BPETokenizer,
                                  clean_text, pretokenize, unk_rate)
from ..evaluation.bleu import corpus_bleu, sacrebleu_score, tokenize_13a
from ..model.transformer import Transformer, TransformerConfig
from ..training.loss import LabelSmoothingLoss, token_cross_entropy
from ..training.scheduler import NoamScheduler, build_optimizer, noam_learning_rate


# ---------------------------------------------------------------- scheduler

class TestNoamSchedule(unittest.TestCase):
    def test_matches_equation_3_literally(self):
        d_model, warmup = 512, 4000
        for step in [1, 2, 100, 3999, 4000, 4001, 50_000, 100_000]:
            want = (d_model ** -0.5) * min(step ** -0.5, step * warmup ** -1.5)
            self.assertAlmostEqual(noam_learning_rate(step, d_model, warmup), want, places=15)

    def test_branches_cross_exactly_at_warmup(self):
        d_model, warmup = 512, 4000
        a = warmup ** -0.5
        b = warmup * warmup ** -1.5
        self.assertAlmostEqual(a, b, places=15)

    def test_peak_lr_value_for_base_model(self):
        """Peak = d_model^-0.5 * warmup^-0.5 = 512^-0.5 * 4000^-0.5."""
        peak = noam_learning_rate(4000, 512, 4000)
        self.assertAlmostEqual(peak, (512 ** -0.5) * (4000 ** -0.5), places=15)
        self.assertAlmostEqual(peak, 6.9877e-4, places=7)

    def test_peak_lr_is_smaller_for_the_big_model(self):
        """The schedule couples LR to d_model: wider models take smaller steps."""
        base = noam_learning_rate(4000, 512, 4000)
        big = noam_learning_rate(4000, 1024, 4000)
        self.assertLess(big, base)
        self.assertAlmostEqual(big / base, 1 / math.sqrt(2), places=10)

    def test_linear_warmup_then_inverse_sqrt_decay(self):
        d_model, warmup = 512, 4000
        lrs = [noam_learning_rate(s, d_model, warmup) for s in range(1, 20_000)]
        peak_idx = max(range(len(lrs)), key=lambda i: lrs[i])
        self.assertEqual(peak_idx + 1, warmup)
        for i in range(1, warmup - 1):
            self.assertGreater(lrs[i], lrs[i - 1])          # strictly increasing
        for i in range(warmup + 1, len(lrs)):
            self.assertLess(lrs[i], lrs[i - 1])             # strictly decreasing
        # Warmup branch is exactly linear in step.
        r = lrs[999] / lrs[499]
        self.assertAlmostEqual(r, 1000 / 500, places=10)
        # Decay branch is exactly inverse-sqrt.
        r2 = lrs[15999] / lrs[7999]
        self.assertAlmostEqual(r2, math.sqrt(8000 / 16000), places=10)

    def test_step_zero_is_zero_and_one_based_counting(self):
        self.assertEqual(noam_learning_rate(0, 512, 4000), 0.0)
        self.assertGreater(noam_learning_rate(1, 512, 4000), 0.0)

    def test_scheduler_applies_lr_to_optimizer_before_stepping(self):
        model = torch.nn.Linear(4, 4)
        opt = build_optimizer(model)
        self.assertEqual(opt.param_groups[0]["betas"], (0.9, 0.98))   # Section 5.3
        self.assertEqual(opt.param_groups[0]["eps"], 1e-9)
        sch = NoamScheduler(opt, d_model=512, warmup_steps=4000)
        model(torch.randn(2, 4)).sum().backward()
        lr = sch.step()
        self.assertEqual(sch.step_num, 1)
        self.assertAlmostEqual(opt.param_groups[0]["lr"], lr, places=15)
        self.assertAlmostEqual(lr, noam_learning_rate(1, 512, 4000), places=15)

    def test_state_dict_round_trip(self):
        opt = build_optimizer(torch.nn.Linear(2, 2))
        sch = NoamScheduler(opt, 512, 4000)
        for _ in range(7):
            sch.step()
        sch2 = NoamScheduler(build_optimizer(torch.nn.Linear(2, 2)), 1, 1)
        sch2.load_state_dict(sch.state_dict())
        self.assertEqual(sch2.step_num, 7)
        self.assertEqual(sch2.d_model, 512)


# --------------------------------------------------------------------- loss

class TestLabelSmoothing(unittest.TestCase):
    def test_zero_smoothing_equals_cross_entropy(self):
        torch.manual_seed(0)
        V = 11
        logits = torch.randn(6, V)
        target = torch.randint(1, V, (6,))
        crit = LabelSmoothingLoss(V, pad_id=0, smoothing=0.0)
        want = F.cross_entropy(logits, target)
        torch.testing.assert_close(crit(logits, target), want, rtol=1e-6, atol=1e-6)

    def test_analytic_value_for_uniform_prediction(self):
        """With uniform logits, log p = -log V for every class, so the
        cross-entropy with ANY target distribution summing to 1 is log V."""
        V = 20
        logits = torch.zeros(4, V)
        target = torch.randint(1, V, (4,))
        for eps in (0.0, 0.1, 0.3):
            crit = LabelSmoothingLoss(V, pad_id=0, smoothing=eps)
            self.assertAlmostEqual(float(crit(logits, target)), math.log(V), places=5)

    def test_smoothing_raises_loss_on_a_confident_correct_prediction(self):
        """Paper: label smoothing "hurts perplexity, as the model learns to
        be more unsure"."""
        V = 10
        logits = torch.full((1, V), -10.0)
        logits[0, 5] = 10.0
        target = torch.tensor([5])
        plain = float(LabelSmoothingLoss(V, 0, 0.0)(logits, target))
        smoothed = float(LabelSmoothingLoss(V, 0, 0.1)(logits, target))
        self.assertGreater(smoothed, plain)

    def test_target_distribution_sums_to_one_and_excludes_pad(self):
        """Reconstruct the implied target distribution from the loss values.

        For uniform-zero logits the loss is sum_k q_k * log V, so we can
        confirm sum(q) = 1. To confirm PAD carries no mass we compare a
        vocabulary where PAD would otherwise absorb eps/V.
        """
        V, eps = 8, 0.2
        crit = LabelSmoothingLoss(V, pad_id=0, smoothing=eps)
        self.assertEqual(crit.support, V - 1)          # PAD excluded, true class included
        crit_ex = LabelSmoothingLoss(V, pad_id=0, smoothing=eps, exclude_true=True)
        self.assertEqual(crit_ex.support, V - 2)       # PAD and true class excluded

    def test_pad_positions_excluded_from_loss(self):
        torch.manual_seed(0)
        V = 12
        logits = torch.randn(1, 5, V)
        target = torch.tensor([[3, 4, 5, PAD_ID, PAD_ID]])
        crit = LabelSmoothingLoss(V, pad_id=PAD_ID, smoothing=0.1)
        full = crit(logits, target)
        trimmed = crit(logits[:, :3], target[:, :3])
        torch.testing.assert_close(full, trimmed, rtol=1e-6, atol=1e-6)

    def test_all_pad_batch_returns_zero_not_nan(self):
        crit = LabelSmoothingLoss(9, pad_id=0, smoothing=0.1)
        loss = crit(torch.randn(1, 3, 9), torch.zeros(1, 3, dtype=torch.long))
        self.assertEqual(float(loss), 0.0)

    def test_perplexity_reported_from_unsmoothed_likelihood(self):
        torch.manual_seed(0)
        V = 15
        logits = torch.randn(2, 4, V)
        target = torch.randint(1, V, (2, 4))
        ce_sum, n = token_cross_entropy(logits, target, pad_id=0)
        self.assertEqual(n, 8)
        want = float(F.cross_entropy(logits.reshape(-1, V), target.reshape(-1), reduction="sum"))
        self.assertAlmostEqual(ce_sum, want, places=4)


# --------------------------------------------------------------------- BLEU

class TestBLEU(unittest.TestCase):
    def test_identical_hypothesis_scores_100(self):
        refs = ["the cat sat on the mat", "a quick brown fox jumps"]
        r = corpus_bleu(refs, refs)
        self.assertAlmostEqual(r["bleu"], 100.0, places=6)
        self.assertAlmostEqual(r["bp"], 1.0, places=10)

    def test_no_ngram_overlap_scores_zero(self):
        r = corpus_bleu(["completely different words here"], ["nothing alike whatsoever friend"])
        self.assertAlmostEqual(r["bleu"], 0.0, places=10)

    def test_modified_precision_clips_repeated_ngrams(self):
        """Papineni's motivating example: repeating a high-probability word
        must not achieve precision 1."""
        r = corpus_bleu(["the the the the the the the"],
                        ["the cat is on the mat"], max_n=1)
        # 'the' appears 7 times in the hypothesis but at most 2 in the
        # reference, so clipped unigram precision = 2/7.
        self.assertAlmostEqual(r["precisions"][0], 2 / 7, places=10)

    def test_brevity_penalty_applied_when_short(self):
        long_ref = "the quick brown fox jumps over the lazy dog again and again"
        r = corpus_bleu(["the quick brown fox"], [long_ref])
        self.assertLess(r["bp"], 1.0)
        self.assertAlmostEqual(r["bp"], math.exp(1 - r["ref_len"] / r["hyp_len"]), places=10)

    def test_no_penalty_when_longer_than_reference(self):
        r = corpus_bleu(["a b c d e f g h"], ["a b c"])
        self.assertAlmostEqual(r["bp"], 1.0, places=10)

    def test_corpus_level_not_sentence_averaged(self):
        """Aggregate n-gram counts first, THEN divide.

        Corpus BLEU is not the mean of sentence BLEUs: a sentence with zero
        4-gram overlap contributes 0 to the average but still contributes
        its matched unigrams and bigrams to the corpus numerators. Conflating
        the two is a common source of inflated reported scores.
        """
        hyps = ["the cat sat on the mat today",
                "a man plays guitar in the street now"]
        refs = ["the cat sat on a mat yesterday",
                "the man is playing the guitar on the street"]
        agg = corpus_bleu(hyps, refs)["bleu"]
        per_sentence = [corpus_bleu([h], [r])["bleu"] for h, r in zip(hyps, refs)]
        self.assertAlmostEqual(agg, 23.1187, places=3)
        self.assertAlmostEqual(sum(per_sentence) / 2, 21.7360, places=3)
        self.assertGreater(abs(agg - sum(per_sentence) / 2), 1.0)

    def test_multi_reference_takes_max_count(self):
        """Clipping uses the MAXIMUM count over references, so matching any
        one reference exactly yields precision 1 on every order."""
        r = corpus_bleu(["the cat sat quietly on the mat"],
                        [["a dog stood there in the yard",
                          "the cat sat quietly on the mat"]])
        self.assertAlmostEqual(r["bleu"], 100.0, places=6)

    def test_bleu4_is_zero_for_sentences_shorter_than_four_tokens(self):
        """Not a bug: a 3-token hypothesis contains no 4-grams, so the
        4th-order precision is 0/0 and the geometric mean collapses to 0.
        The original definition has no special case for this; smoothing
        variants exist precisely to paper over it, and we expose them
        rather than silently applying one."""
        strict = corpus_bleu(["the cat sat"], ["the cat sat"])
        self.assertEqual(strict["totals"][3], 0)
        self.assertAlmostEqual(strict["bleu"], 0.0, places=10)
        self.assertGreater(corpus_bleu(["the cat sat"], ["the cat sat"],
                                       max_n=3)["bleu"], 99.9)
        self.assertGreater(corpus_bleu(["the cat sat"], ["the cat sat"],
                                       effective_order=True)["bleu"], 99.9)
        self.assertEqual(corpus_bleu(["the cat sat"], ["the cat sat"],
                                     effective_order=True)["orders_used"], 3)

    def test_tokenizer_separates_punctuation(self):
        self.assertEqual(tokenize_13a("Hello, world!"), ["Hello", ",", "world", "!"])
        # Numeric commas are preserved (mteval-v13a behaviour).
        self.assertIn("1,000", tokenize_13a("about 1,000 people"))

    def test_agrees_with_sacrebleu_reference_implementation(self):
        """Cross-validation against the community-standard scorer."""
        hyps = [
            "the quick brown fox jumps over the lazy dog",
            "a man is playing a guitar on the street",
            "two dogs run through the tall grass together",
            "she opened the window because the room was warm",
        ]
        refs = [
            "a quick brown fox jumped over the lazy dog",
            "a man plays the guitar in the street",
            "two dogs are running through tall grass",
            "she opened the window since the room felt warm",
        ]
        mine = corpus_bleu(hyps, refs)["bleu"]
        theirs = sacrebleu_score(hyps, refs)
        if theirs is None:
            self.skipTest("sacrebleu unavailable")
        self.assertAlmostEqual(mine, theirs, delta=0.5)


# ----------------------------------------------------------- data pipeline

class TestBPE(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            "the cat sat on the mat", "the cats sat on the mats",
            "a dog ran to the park", "dogs ran to parks",
            "the quick brown fox jumps over the lazy dog",
        ] * 8

    def test_clean_text_normalizes_and_collapses_whitespace(self):
        self.assertEqual(clean_text("  hello   world \n"), "hello world")

    def test_pretokenize_isolates_punctuation(self):
        self.assertEqual(pretokenize("Hi, there!"), ["Hi", ",", "there", "!"])

    def test_special_tokens_occupy_first_four_ids(self):
        tok = BPETokenizer.train([self.corpus], num_merges=20)
        self.assertEqual(tok.vocab["<pad>"], PAD_ID)
        self.assertEqual(tok.vocab["<bos>"], BOS_ID)
        self.assertEqual(tok.vocab["<eos>"], EOS_ID)
        self.assertEqual(tok.vocab["<unk>"], UNK_ID)

    def test_more_merges_yield_larger_vocab_and_shorter_sequences(self):
        lens, sizes = [], []
        for m in (10, 60, 200):
            tok = BPETokenizer.train([self.corpus], num_merges=m)
            sizes.append(tok.vocab_size)
            lens.append(len(tok.tokenize("the quick brown fox jumps over the lazy dog")))
        self.assertEqual(sizes, sorted(sizes))
        self.assertGreaterEqual(lens[0], lens[-1])

    def test_encode_decode_round_trip(self):
        tok = BPETokenizer.train([self.corpus], num_merges=100)
        for s in ["the cat sat on the mat", "a dog ran to the park"]:
            ids = tok.encode(s, add_bos=True, add_eos=True)
            self.assertEqual(ids[0], BOS_ID)
            self.assertEqual(ids[-1], EOS_ID)
            self.assertEqual(tok.decode(ids), s)

    def test_unseen_word_is_segmented_not_unked(self):
        """BPE's core property: no OOV as long as characters are known."""
        tok = BPETokenizer.train([self.corpus], num_merges=100)
        ids = tok.encode("catdogfox", add_eos=False)
        self.assertNotIn(UNK_ID, ids)
        self.assertEqual(tok.decode(ids), "catdogfox")

    def test_unseen_character_becomes_unk(self):
        tok = BPETokenizer.train([self.corpus], num_merges=50)
        self.assertGreater(unk_rate(tok, ["日本語"]), 0.0)

    def test_shared_vocabulary_over_two_languages(self):
        src = ["the cat sat"] * 5
        tgt = ["die katze sass"] * 5
        tok = BPETokenizer.train([src, tgt], num_merges=40)
        self.assertNotIn(UNK_ID, tok.encode("the cat sat", add_eos=False))
        self.assertNotIn(UNK_ID, tok.encode("die katze sass", add_eos=False))

    def test_determinism(self):
        a = BPETokenizer.train([self.corpus], num_merges=50)
        b = BPETokenizer.train([self.corpus], num_merges=50)
        self.assertEqual(a.merges, b.merges)
        self.assertEqual(a.vocab, b.vocab)

    def test_save_load_round_trip(self):
        tok = BPETokenizer.train([self.corpus], num_merges=50)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bpe.json"
            tok.save(p)
            tok2 = BPETokenizer.load(p)
        self.assertEqual(tok.vocab, tok2.vocab)
        self.assertEqual(tok.tokenize("the cat sat"), tok2.tokenize("the cat sat"))


class TestBatching(unittest.TestCase):
    def test_collate_right_shifts_the_decoder_input(self):
        """Section 3.1: outputs are "offset by one position"."""
        ex = Example(src=[7, 8, EOS_ID], tgt=[20, 21, 22, EOS_ID])
        b = collate([ex])
        self.assertEqual(b["tgt_in"][0].tolist(), [BOS_ID, 20, 21, 22])
        self.assertEqual(b["tgt_out"][0].tolist(), [20, 21, 22, EOS_ID])
        # Target at position i is the decoder input at position i+1.
        self.assertEqual(b["tgt_in"][0, 1:].tolist(), b["tgt_out"][0, :-1].tolist())

    def test_collate_pads_to_batch_max_and_counts_tokens(self):
        b = collate([Example([5, EOS_ID], [9, EOS_ID]),
                     Example([5, 6, 7, EOS_ID], [9, 10, 11, 12, EOS_ID])])
        self.assertEqual(tuple(b["src"].shape), (2, 4))
        # Decoder input length == target length (BOS prepended, final EOS
        # consumed as a target), so Lt = 5 for the longest target, NOT 6.
        self.assertEqual(tuple(b["tgt_in"].shape), (2, 5))
        self.assertEqual(tuple(b["tgt_out"].shape), (2, 5))
        self.assertEqual(b["n_tokens"], 2 + 5)         # non-PAD target tokens
        self.assertEqual(b["src"][0].tolist(), [5, EOS_ID, PAD_ID, PAD_ID])

    def test_token_batching_respects_the_budget(self):
        torch.manual_seed(0)
        ex = [Example(list(range(4, 4 + n)), list(range(4, 4 + n)))
              for n in [3, 5, 8, 12, 20, 31, 4, 6, 9, 15] * 6]
        corpus = ParallelCorpus(ex)
        budget = 200
        sampler = TokenBatchSampler(corpus, max_tokens=budget, shuffle=True, seed=0)
        n_batches = 0
        for idx in sampler:
            batch = collate([corpus[i] for i in idx])
            padded_src = batch["src"].numel()
            padded_tgt = batch["tgt_in"].numel()
            self.assertLessEqual(padded_src, budget, "source token budget exceeded")
            self.assertLessEqual(padded_tgt, budget, "target token budget exceeded")
            n_batches += 1
        self.assertGreater(n_batches, 1)

    def test_every_example_appears_exactly_once_per_epoch(self):
        ex = [Example([4, 5, EOS_ID], [4, 5, EOS_ID]) for _ in range(37)]
        corpus = ParallelCorpus(ex)
        seen = [i for b in TokenBatchSampler(corpus, max_tokens=60, seed=3) for i in b]
        self.assertEqual(sorted(seen), list(range(37)))

    def test_length_bucketing_reduces_padding_waste(self):
        """The point of bucketing: fewer PAD tokens than random batching."""
        ex = [Example(list(range(4, 4 + n)), list(range(4, 4 + n)))
              for n in ([3] * 40 + [40] * 40)]
        corpus = ParallelCorpus(ex)
        bucketed = TokenBatchSampler(corpus, max_tokens=400, shuffle=False, seed=0)
        waste_b = 0
        for idx in bucketed:
            b = collate([corpus[i] for i in idx])
            waste_b += int((b["tgt_in"] == PAD_ID).sum())
        # Compare against fixed-size batches drawn in an adversarial order
        # (alternating short and long), which is what unsorted batching risks.
        order = [i for pair in zip(range(40), range(40, 80)) for i in pair]
        waste_r = 0
        for s in range(0, len(order), 8):
            b = collate([corpus[i] for i in order[s : s + 8]])
            waste_r += int((b["tgt_in"] == PAD_ID).sum())
        self.assertLess(waste_b, waste_r)

    def test_epoch_changes_batch_composition(self):
        ex = [Example(list(range(4, 4 + (i % 9) + 3)), list(range(4, 4 + (i % 7) + 3)))
              for i in range(60)]
        s = TokenBatchSampler(ParallelCorpus(ex), max_tokens=100, shuffle=True, seed=0)
        s.set_epoch(0); a = [list(b) for b in s]
        s.set_epoch(1); b = [list(b) for b in s]
        self.assertNotEqual(a, b)


class TestEndToEndOverfit(unittest.TestCase):
    """A model that cannot memorize a tiny batch has a wiring bug.

    This is the cheapest end-to-end learning test available: it exercises
    the loss, the mask, the shift, the optimizer and the schedule together.
    """

    def test_overfits_a_single_batch(self):
        torch.manual_seed(0)
        V = 30
        cfg = TransformerConfig(src_vocab_size=V, tgt_vocab_size=V, d_model=32,
                                num_heads=4, d_ff=64, num_encoder_layers=2,
                                num_decoder_layers=2, dropout=0.0)
        model = Transformer(cfg)
        crit = LabelSmoothingLoss(V, pad_id=PAD_ID, smoothing=0.0)
        opt = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.98), eps=1e-9)

        src = torch.randint(4, V, (4, 7))
        tgt = torch.randint(4, V, (4, 6))
        tgt_in = torch.cat([torch.full((4, 1), BOS_ID), tgt[:, :-1]], dim=1)

        first = None
        for _ in range(220):
            opt.zero_grad()
            loss = crit(model(src, tgt_in), tgt)
            loss.backward()
            opt.step()
            if first is None:
                first = float(loss)
        self.assertLess(float(loss), 0.05 * first,
                        f"failed to overfit: {first:.3f} -> {float(loss):.3f}")


if __name__ == "__main__":
    unittest.main()
