"""Verification of masking (Sections 3.1 and 3.2.3).

The auto-regressive property is a *causality* claim, so the strongest test
is not "does the mask look lower-triangular" but "is the Jacobian
d output[i] / d input[j] exactly zero for j > i". We test the property
itself, through the full decoder stack, using autograd.
"""

from __future__ import annotations

import unittest

import torch

from ..model.attention import scaled_dot_product_attention
from ..model.masking import causal_mask, decoder_mask, padding_mask
from ..model.transformer import Transformer, TransformerConfig


class TestMaskConstruction(unittest.TestCase):
    def test_causal_mask_is_lower_triangular(self):
        m = causal_mask(5)[0, 0]
        self.assertEqual(tuple(m.shape), (5, 5))
        for i in range(5):
            for j in range(5):
                self.assertEqual(bool(m[i, j]), j <= i, f"({i},{j})")

    def test_padding_mask_shape_and_values(self):
        tokens = torch.tensor([[5, 6, 0, 0], [7, 8, 9, 0]])
        m = padding_mask(tokens, pad_id=0)
        self.assertEqual(tuple(m.shape), (2, 1, 1, 4))   # broadcasts over heads & queries
        self.assertEqual(m[0, 0, 0].tolist(), [True, True, False, False])
        self.assertEqual(m[1, 0, 0].tolist(), [True, True, True, False])

    def test_decoder_mask_is_conjunction(self):
        tokens = torch.tensor([[1, 5, 6, 0]])
        m = decoder_mask(tokens, pad_id=0)[0, 0]
        self.assertEqual(tuple(m.shape), (4, 4))
        # Row 2 may see keys 0,1,2 (causal) but key 3 is PAD -> excluded.
        self.assertEqual(m[2].tolist(), [True, True, True, False])
        self.assertEqual(m[3].tolist(), [True, True, True, False])

    def test_masked_positions_receive_zero_weight(self):
        Q = torch.randn(1, 1, 4, 8)
        K = torch.randn(1, 1, 4, 8)
        V = torch.randn(1, 1, 4, 8)
        mask = causal_mask(4)
        _, attn = scaled_dot_product_attention(Q, K, V, mask=mask)
        upper = attn[0, 0][~mask[0, 0]]
        self.assertEqual(upper.numel(), 6)
        self.assertTrue(bool((upper == 0).all()), f"leaked weight: {upper}")
        # Rows must still normalize despite suppression.
        torch.testing.assert_close(attn.sum(-1), torch.ones(1, 1, 4), rtol=1e-6, atol=1e-6)

    def test_no_nan_when_row_heavily_masked(self):
        """Motivates using finfo.min instead of -inf (see attention.py note)."""
        Q, K, V = (torch.randn(1, 1, 3, 4) for _ in range(3))
        mask = torch.zeros(1, 1, 3, 3, dtype=torch.bool)
        mask[..., 0] = True                      # only key 0 permitted
        out, attn = scaled_dot_product_attention(Q, K, V, mask=mask)
        self.assertFalse(bool(torch.isnan(out).any()))
        torch.testing.assert_close(attn[..., 0], torch.ones(1, 1, 3), rtol=1e-6, atol=1e-6)


class TestCausalityThroughFullModel(unittest.TestCase):
    """End-to-end auto-regressive property, not just the mask tensor."""

    def setUp(self):
        torch.manual_seed(0)
        self.cfg = TransformerConfig(
            src_vocab_size=40, tgt_vocab_size=40, num_encoder_layers=2,
            num_decoder_layers=2, d_model=32, num_heads=4, d_ff=64, dropout=0.0,
        )
        self.model = Transformer(self.cfg).eval()

    def test_jacobian_is_strictly_causal(self):
        """d logits[t] / d decoder_input_embedding[s] must be 0 for s > t."""
        L = 6
        src = torch.randint(4, 40, (1, 5))
        tgt = torch.randint(4, 40, (1, L))
        emb = self.model.tgt_embed(tgt) * self.model.emb_scale
        emb = emb.detach().requires_grad_(True)

        memory = self.model.encode(src)
        from ..model.masking import decoder_mask as dm
        x = self.model.tgt_pos(emb)
        h = self.model.decoder(x, memory, dm(tgt, self.cfg.pad_id), None)
        logits = self.model.generator(h)               # (1, L, V)

        for t in range(L):
            self.model.zero_grad()
            if emb.grad is not None:
                emb.grad = None
            logits[0, t].sum().backward(retain_graph=True)
            g = emb.grad[0].abs().sum(-1)              # (L,) sensitivity per position
            for s in range(L):
                if s > t:
                    self.assertLess(float(g[s]), 1e-12,
                                    f"position {t} leaked information from future position {s}")
            self.assertGreater(float(g[t]), 0.0, f"position {t} ignores its own input")

    def test_future_token_perturbation_does_not_change_past_logits(self):
        """Complementary black-box check of the same property."""
        src = torch.randint(4, 40, (1, 5))
        tgt = torch.randint(4, 40, (1, 7))
        with torch.no_grad():
            a = self.model(src, tgt)
            tgt2 = tgt.clone()
            tgt2[0, 5] = (tgt2[0, 5] + 7) % 40 + 4     # change a late token
            b = self.model(src, tgt2)
        torch.testing.assert_close(a[:, :5], b[:, :5], rtol=1e-6, atol=1e-6)
        self.assertGreater((a[:, 5:] - b[:, 5:]).abs().max().item(), 1e-6)

    def test_padding_invariance(self):
        """Appending PAD to a source must not change the model's outputs.

        If the source padding mask were missing or mis-shaped, encoder
        self-attention would attend to filler and this test would fail --
        the classic silent bug that makes results depend on batch grouping.
        """
        src = torch.randint(4, 40, (1, 5))
        tgt = torch.randint(4, 40, (1, 4))
        padded = torch.cat([src, torch.zeros(1, 6, dtype=torch.long)], dim=1)
        with torch.no_grad():
            a = self.model(src, tgt)
            b = self.model(padded, tgt)
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)

    def test_batch_composition_invariance(self):
        """A sentence's output must not depend on its batch-mates."""
        s1 = torch.tensor([[5, 6, 7]])
        s2 = torch.tensor([[8, 9, 10, 11, 12, 13]])
        t1 = torch.tensor([[1, 20, 21]])
        t2 = torch.tensor([[1, 22, 23, 24, 25]])
        pad = lambda x, L: torch.cat([x, torch.zeros(1, L - x.size(1), dtype=torch.long)], 1)
        with torch.no_grad():
            alone = self.model(s1, t1)
            together = self.model(torch.cat([pad(s1, 6), s2]), torch.cat([pad(t1, 5), t2]))
        torch.testing.assert_close(alone, together[:1, :3], rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
