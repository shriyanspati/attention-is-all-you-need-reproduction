"""Tests for the embedding module (Section 3.4) and inference (Section 6.1)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from ..inference.decoding import average_checkpoints, length_penalty, select_checkpoints
from ..model.embeddings import build_embeddings, embedding_scale, verify_tying
from ..model.transformer import Transformer, TransformerConfig


class TestEmbeddings(unittest.TestCase):
    def test_tying_shares_the_same_object_not_just_shape(self):
        src, tgt, gen = build_embeddings(100, 100, 32, pad_id=0, tie=True)
        self.assertIs(src.weight, tgt.weight)
        self.assertIs(gen.weight, tgt.weight)

    def test_untied_matrices_are_distinct_objects(self):
        src, tgt, gen = build_embeddings(100, 100, 32, pad_id=0, tie=False)
        self.assertIsNot(src.weight, tgt.weight)
        self.assertIsNot(gen.weight, tgt.weight)

    def test_tying_requires_shared_vocabulary(self):
        with self.assertRaises(ValueError):
            build_embeddings(100, 90, 32, pad_id=0, tie=True)

    def test_embedding_scale_is_sqrt_d_model(self):
        self.assertAlmostEqual(embedding_scale(512), 512 ** 0.5, places=12)
        self.assertAlmostEqual(embedding_scale(64), 8.0, places=12)

    def test_verify_tying_reports_model_state(self):
        m = Transformer(TransformerConfig(src_vocab_size=40, tgt_vocab_size=40, d_model=64,
                                          num_heads=4, d_ff=64, num_encoder_layers=1,
                                          num_decoder_layers=1))
        info = verify_tying(m)
        self.assertTrue(info["src_is_tgt"] and info["tgt_is_generator"])
        self.assertAlmostEqual(info["embedding_scale"], 8.0, places=12)


class TestDecodingUtilities(unittest.TestCase):
    def test_length_penalty_is_one_at_unit_length_and_increasing(self):
        self.assertAlmostEqual(length_penalty(1), 1.0, places=12)
        self.assertLess(length_penalty(5), length_penalty(25))

    def test_length_penalty_alpha_zero_disables_it(self):
        for L in (1, 10, 40):
            self.assertAlmostEqual(length_penalty(L, alpha=0.0), 1.0, places=12)

    def test_average_checkpoints_is_the_arithmetic_mean(self):
        cfg = TransformerConfig(src_vocab_size=30, tgt_vocab_size=30, d_model=32,
                                num_heads=4, d_ff=64, num_encoder_layers=1,
                                num_decoder_layers=1)
        with tempfile.TemporaryDirectory() as d:
            models, paths = [], []
            for i in range(3):
                torch.manual_seed(i)
                m = Transformer(cfg)
                p = Path(d) / f"ckpt_{i:06d}.pt"
                torch.save({"model": m.state_dict(), "step": i}, p)
                models.append(m); paths.append(p)
            target = Transformer(cfg)
            average_checkpoints(paths, target)
            key = "encoder.layers.0.ff.w_1.weight"
            want = sum(m.state_dict()[key].float() for m in models) / 3
            torch.testing.assert_close(target.state_dict()[key], want, rtol=1e-6, atol=1e-6)

    def test_select_checkpoints_returns_last_n_oldest_first(self):
        with tempfile.TemporaryDirectory() as d:
            for i in (100, 200, 300, 400):
                (Path(d) / f"ckpt_{i:06d}.pt").touch()
            got = [p.name for p in select_checkpoints(d, last_n=3)]
            self.assertEqual(got, ["ckpt_000200.pt", "ckpt_000300.pt", "ckpt_000400.pt"])

    def test_average_of_empty_list_raises(self):
        with self.assertRaises(ValueError):
            average_checkpoints([], None)
