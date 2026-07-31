"""Verification of Sections 3.1, 3.3, 3.4: stacks, shapes, parameter counts.

The parameter-count tests are the sharpest available check that no
component is missing or duplicated: a closed-form count derived by hand
from the paper's description must match `sum(p.numel())` exactly. Any
missing LayerNorm, extra bias, or untied embedding shows up immediately.
"""

from __future__ import annotations

import unittest

import torch

from ..model.decoder import Decoder, DecoderLayer
from ..model.encoder import Encoder, EncoderLayer
from ..model.feed_forward import PositionwiseFeedForward
from ..model.transformer import Transformer, TransformerConfig


def analytic_param_count(
    vocab: int, d: int, d_ff: int, h: int, n_enc: int, n_dec: int,
    tie: bool = True, d_k: int | None = None, d_v: int | None = None,
) -> int:
    """Closed-form parameter count derived by hand from the paper.

    Per multi-head attention (bias-free projections, Section 3.2.2):
        W^Q, W^K : 2 * d * h * d_k
        W^V      : d * h * d_v
        W^O      : h * d_v * d
    Per feed-forward (eq. 2 includes biases b_1, b_2):
        2 * d * d_ff + d_ff + d
    Per LayerNorm: 2 * d (gain and bias)
    Encoder layer: 1 attention + 1 FFN + 2 LayerNorm
    Decoder layer: 2 attention + 1 FFN + 3 LayerNorm
    Embeddings: vocab * d, counted ONCE under three-way tying (Section 3.4)
    """
    d_k = d_k or d // h
    d_v = d_v or d // h
    attn = 2 * d * h * d_k + d * h * d_v + h * d_v * d
    ffn = 2 * d * d_ff + d_ff + d
    ln = 2 * d
    enc = n_enc * (attn + ffn + 2 * ln)
    dec = n_dec * (2 * attn + ffn + 3 * ln)
    emb = vocab * d if tie else 3 * vocab * d
    return emb + enc + dec


class TestSubmoduleShapes(unittest.TestCase):
    def test_feed_forward_shapes_and_relu(self):
        ff = PositionwiseFeedForward(512, 2048)
        self.assertEqual(tuple(ff.w_1.weight.shape), (2048, 512))
        self.assertEqual(tuple(ff.w_2.weight.shape), (512, 2048))
        x = torch.randn(2, 5, 512)
        self.assertEqual(tuple(ff(x).shape), (2, 5, 512))
        # Verify max(0, .) is applied to the hidden layer, not the output.
        with torch.no_grad():
            h = torch.relu(ff.w_1(x))
            self.assertTrue(bool((h >= 0).all()))
            torch.testing.assert_close(ff(x), ff.w_2(h), rtol=1e-6, atol=1e-6)

    def test_feed_forward_is_position_wise(self):
        """"applied to each position separately and identically" -- so the
        output at a position depends only on the input at that position."""
        ff = PositionwiseFeedForward(16, 32).eval()
        x = torch.randn(1, 6, 16)
        full = ff(x)
        for t in range(6):
            torch.testing.assert_close(ff(x[:, t : t + 1]), full[:, t : t + 1],
                                       rtol=1e-6, atol=1e-6)

    def test_encoder_layer_preserves_shape(self):
        layer = EncoderLayer(64, 8, 256, dropout=0.0)
        x = torch.randn(3, 9, 64)
        self.assertEqual(tuple(layer(x).shape), (3, 9, 64))

    def test_encoder_stack_depth(self):
        enc = Encoder(6, 64, 8, 256, dropout=0.0)
        self.assertEqual(len(enc.layers), 6)          # N = 6 (Section 3.1)
        self.assertEqual(tuple(enc(torch.randn(2, 11, 64)).shape), (2, 11, 64))

    def test_decoder_layer_has_three_sublayers(self):
        layer = DecoderLayer(64, 8, 256, dropout=0.0)
        self.assertTrue(hasattr(layer, "self_attn"))
        self.assertTrue(hasattr(layer, "cross_attn"))
        self.assertTrue(hasattr(layer, "ff"))
        # Three sub-layers => three LayerNorms (one per residual block).
        norms = [m for m in layer.modules() if isinstance(m, torch.nn.LayerNorm)]
        self.assertEqual(len(norms), 3)
        x = torch.randn(2, 5, 64)
        mem = torch.randn(2, 13, 64)
        self.assertEqual(tuple(layer(x, mem).shape), (2, 5, 64))

    def test_encoder_layer_has_two_sublayers(self):
        layer = EncoderLayer(64, 8, 256, dropout=0.0)
        norms = [m for m in layer.modules() if isinstance(m, torch.nn.LayerNorm)]
        self.assertEqual(len(norms), 2)

    def test_post_norm_output_is_layer_normalized(self):
        """The paper's form ends each sub-layer with LayerNorm, so encoder
        output should have ~zero mean and ~unit variance per position."""
        enc = Encoder(6, 128, 8, 256, dropout=0.0, norm_first=False).eval()
        out = enc(torch.randn(4, 10, 128))
        self.assertLess(float(out.mean(-1).abs().max()), 1e-4)
        self.assertAlmostEqual(float(out.var(-1, unbiased=False).mean()), 1.0, delta=0.05)

    def test_pre_norm_output_is_not_normalized_before_final_norm(self):
        enc_pre = Encoder(6, 128, 8, 256, dropout=0.0, norm_first=True).eval()
        self.assertIsNotNone(enc_pre.norm)
        enc_post = Encoder(6, 128, 8, 256, dropout=0.0, norm_first=False).eval()
        self.assertIsNone(enc_post.norm)


class TestFullModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_forward_shapes(self):
        cfg = TransformerConfig(src_vocab_size=100, tgt_vocab_size=100, d_model=64,
                                num_heads=8, d_ff=128, num_encoder_layers=2,
                                num_decoder_layers=2, dropout=0.0)
        m = Transformer(cfg)
        src = torch.randint(4, 100, (3, 9))
        tgt = torch.randint(4, 100, (3, 6))
        self.assertEqual(tuple(m.encode(src).shape), (3, 9, 64))
        self.assertEqual(tuple(m(src, tgt).shape), (3, 6, 100))

    def test_three_way_weight_tying(self):
        cfg = TransformerConfig(src_vocab_size=50, tgt_vocab_size=50, d_model=32,
                                num_heads=4, d_ff=64, num_encoder_layers=1,
                                num_decoder_layers=1, tie_embeddings=True)
        m = Transformer(cfg)
        self.assertIs(m.src_embed.weight, m.tgt_embed.weight)
        self.assertIs(m.generator.weight, m.src_embed.weight)

    def test_untied_variant_has_distinct_matrices(self):
        cfg = TransformerConfig(src_vocab_size=50, tgt_vocab_size=50, d_model=32,
                                num_heads=4, d_ff=64, num_encoder_layers=1,
                                num_decoder_layers=1, tie_embeddings=False)
        m = Transformer(cfg)
        self.assertIsNot(m.src_embed.weight, m.tgt_embed.weight)
        self.assertIsNot(m.generator.weight, m.tgt_embed.weight)

    def test_embedding_scaled_by_sqrt_d_model(self):
        """Section 3.4: "we multiply those weights by sqrt(d_model)"."""
        cfg = TransformerConfig(src_vocab_size=50, tgt_vocab_size=50, d_model=64,
                                num_heads=4, d_ff=64, num_encoder_layers=1,
                                num_decoder_layers=1, dropout=0.0,
                                positional_encoding="none")
        m = Transformer(cfg).eval()
        self.assertAlmostEqual(m.emb_scale, 8.0, places=10)      # sqrt(64)
        src = torch.randint(4, 50, (1, 4))
        with torch.no_grad():
            scaled = m.src_pos(m.src_embed(src) * m.emb_scale)
            raw = m.src_embed(src)
        torch.testing.assert_close(scaled, raw * 8.0, rtol=1e-6, atol=1e-6)

    def test_parameter_count_matches_closed_form(self):
        for V, d, d_ff, h, N in [(37000, 512, 2048, 8, 6),
                                 (37000, 1024, 4096, 16, 6),
                                 (5000, 256, 1024, 4, 3)]:
            cfg = TransformerConfig(src_vocab_size=V, tgt_vocab_size=V, d_model=d,
                                    d_ff=d_ff, num_heads=h, num_encoder_layers=N,
                                    num_decoder_layers=N)
            got = Transformer(cfg).count_parameters()
            want = analytic_param_count(V, d, d_ff, h, N, N, tie=True)
            self.assertEqual(got, want, f"V={V} d={d}: {got} != {want}")

    def test_base_model_parameter_count_vs_table_3(self):
        """Table 3 lists 65e6 parameters for base. We obtain 63.0e6.

        The discrepancy is 3%, and this test documents its source rather
        than hiding it. Under three-way tying, a 65M count requires a shared
        vocabulary of ~40.8k tokens, whereas Section 5.1 says "about 37000".
        Since the same architecture gives 214.2M for big against a reported
        213M, the two reported counts cannot both be reproduced with a
        single vocabulary size -- so at least one is rounded or computed
        under slightly different accounting.
        """
        base = analytic_param_count(37000, 512, 2048, 8, 6, 6)
        big = analytic_param_count(37000, 1024, 4096, 16, 6, 6)
        self.assertEqual(base, 63_045_632)      # 63.05e6, reported 65e6
        self.assertEqual(big, 214_171_648)
        # Within 5% of the reported values in both cases.
        self.assertLess(abs(base - 65e6) / 65e6, 0.05)
        self.assertLess(abs(big - 213e6) / 213e6, 0.05)
        # Vocabulary implied by each reported count, under our architecture:
        non_emb_base = base - 37000 * 512
        implied_base_vocab = (65e6 - non_emb_base) / 512
        non_emb_big = big - 37000 * 1024
        implied_big_vocab = (213e6 - non_emb_big) / 1024
        self.assertGreater(implied_base_vocab, 40000)     # ~40.8k
        self.assertLess(implied_big_vocab, 36500)         # ~35.9k
        # The two implied vocabularies are mutually inconsistent by >4k tokens.
        self.assertGreater(abs(implied_base_vocab - implied_big_vocab), 4000)

    def test_big_config_matches_table_3_row(self):
        cfg = TransformerConfig.big(src_vocab_size=37000, tgt_vocab_size=37000)
        self.assertEqual((cfg.d_model, cfg.d_ff, cfg.num_heads, cfg.dropout),
                         (1024, 4096, 16, 0.3))
        self.assertEqual(cfg.num_encoder_layers, 6)

    def test_dropout_active_in_train_inactive_in_eval(self):
        cfg = TransformerConfig(src_vocab_size=50, tgt_vocab_size=50, d_model=32,
                                num_heads=4, d_ff=64, num_encoder_layers=2,
                                num_decoder_layers=2, dropout=0.5)
        m = Transformer(cfg)
        src = torch.randint(4, 50, (2, 5))
        tgt = torch.randint(4, 50, (2, 5))
        m.train()
        torch.manual_seed(1); a = m(src, tgt)
        torch.manual_seed(2); b = m(src, tgt)
        self.assertGreater((a - b).abs().max().item(), 1e-6)
        m.eval()
        with torch.no_grad():
            self.assertLess((m(src, tgt) - m(src, tgt)).abs().max().item(), 1e-9)

    def test_gradients_reach_every_parameter(self):
        """A parameter with no gradient is a disconnected component."""
        cfg = TransformerConfig(src_vocab_size=30, tgt_vocab_size=30, d_model=32,
                                num_heads=4, d_ff=64, num_encoder_layers=2,
                                num_decoder_layers=2, dropout=0.0)
        m = Transformer(cfg)
        logits = m(torch.randint(4, 30, (2, 6)), torch.randint(4, 30, (2, 5)))
        logits.sum().backward()
        for name, p in m.named_parameters():
            self.assertIsNotNone(p.grad, f"{name} received no gradient")
            self.assertGreater(float(p.grad.abs().sum()), 0.0, f"{name} gradient is all zero")


class TestDecoding(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.cfg = TransformerConfig(src_vocab_size=40, tgt_vocab_size=40, d_model=32,
                                     num_heads=4, d_ff=64, num_encoder_layers=2,
                                     num_decoder_layers=2, dropout=0.0)
        self.model = Transformer(self.cfg).eval()

    def test_greedy_decode_starts_with_bos_and_respects_max_length(self):
        src = torch.randint(4, 40, (2, 6))
        out = self.model.greedy_decode(src, max_extra=5)
        self.assertTrue(bool((out[:, 0] == self.cfg.bos_id).all()))
        self.assertLessEqual(out.size(1), 6 + 5)

    def test_beam_search_returns_sequence(self):
        src = torch.randint(4, 40, (1, 5))
        out = self.model.beam_search(src, beam_size=4, max_extra=6)
        self.assertEqual(out.size(0), 1)
        self.assertEqual(int(out[0, 0]), self.cfg.bos_id)

    def test_beam_size_one_is_close_to_greedy(self):
        """Beam size 1 is greedy search (modulo EOS bookkeeping)."""
        src = torch.randint(4, 40, (1, 5))
        greedy = self.model.greedy_decode(src, max_extra=8)[0].tolist()
        beam = self.model.beam_search(src, beam_size=1, max_extra=8)[0].tolist()
        n = min(len(greedy), len(beam))
        self.assertEqual(greedy[:n], beam[:n])

    def test_length_penalty_monotone_in_length(self):
        """GNMT penalty lp(Y) = ((5+|Y|)/6)^alpha grows with length, so longer
        hypotheses are divided by a larger number -- it counteracts the bias
        of raw log-probability toward short outputs."""
        alpha = 0.6
        lp = lambda L: ((5.0 + L) / 6.0) ** alpha
        self.assertLess(lp(5), lp(20))
        self.assertAlmostEqual(lp(1), 1.0, places=10)


if __name__ == "__main__":
    unittest.main()
