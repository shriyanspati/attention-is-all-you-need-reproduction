"""Verification of Section 3.2: scaled dot-product and multi-head attention.

Strategy: every vectorized implementation is checked against an
independently written *naive* version that follows the paper's equations
literally, element by element, with explicit Python loops. A shape check
alone would not catch a transposed matmul or a missing scale factor; an
agreement check against a loop transcription of equation (1) does.
"""

from __future__ import annotations

import math
import unittest

import torch

from ..model.attention import MultiHeadAttention, scaled_dot_product_attention


def naive_attention(Q, K, V, mask=None):
    """Literal transcription of eq. (1) with explicit loops.

    Q: (Lq, Dk), K: (Lk, Dk), V: (Lk, Dv) -> (Lq, Dv), (Lq, Lk)
    """
    Lq, Dk = Q.shape
    Lk, Dv = V.shape
    out = torch.zeros(Lq, Dv, dtype=torch.float64)
    attn = torch.zeros(Lq, Lk, dtype=torch.float64)
    for i in range(Lq):
        scores = []
        for j in range(Lk):
            s = sum(Q[i, d] * K[j, d] for d in range(Dk)) / math.sqrt(Dk)
            if mask is not None and not bool(mask[i, j]):
                s = -float("inf")
            scores.append(s)
        m = max(scores)
        exps = [math.exp(s - m) if s != -float("inf") else 0.0 for s in scores]
        Z = sum(exps)
        for j in range(Lk):
            w = exps[j] / Z
            attn[i, j] = w
            for d in range(Dv):
                out[i, d] += w * V[j, d]
    return out, attn


class TestScaledDotProductAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_matches_naive_transcription_of_equation_1(self):
        Lq, Lk, Dk, Dv = 5, 7, 4, 6
        Q = torch.randn(Lq, Dk, dtype=torch.float64)
        K = torch.randn(Lk, Dk, dtype=torch.float64)
        V = torch.randn(Lk, Dv, dtype=torch.float64)
        ref_out, ref_attn = naive_attention(Q, K, V)
        out, attn = scaled_dot_product_attention(
            Q[None, None], K[None, None], V[None, None]
        )
        torch.testing.assert_close(out[0, 0], ref_out, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(attn[0, 0], ref_attn, rtol=1e-10, atol=1e-10)

    def test_matches_naive_with_mask(self):
        Lq, Lk, Dk, Dv = 4, 4, 3, 3
        Q = torch.randn(Lq, Dk, dtype=torch.float64)
        K = torch.randn(Lk, Dk, dtype=torch.float64)
        V = torch.randn(Lk, Dv, dtype=torch.float64)
        mask = torch.ones(Lq, Lk, dtype=torch.bool).tril()
        ref_out, ref_attn = naive_attention(Q, K, V, mask)
        out, attn = scaled_dot_product_attention(
            Q[None, None], K[None, None], V[None, None], mask=mask[None, None]
        )
        torch.testing.assert_close(out[0, 0], ref_out, rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(attn[0, 0], ref_attn, rtol=1e-9, atol=1e-9)

    def test_scale_factor_is_sqrt_dk_not_dk_or_one(self):
        """Guard against the two most common scaling bugs."""
        Dk = 16
        Q = torch.randn(1, 1, 3, Dk, dtype=torch.float64)
        K = torch.randn(1, 1, 3, Dk, dtype=torch.float64)
        V = torch.eye(3, dtype=torch.float64)[None, None]
        _, attn = scaled_dot_product_attention(Q, K, V)
        raw = Q @ K.transpose(-2, -1)
        expected = torch.softmax(raw / math.sqrt(Dk), dim=-1)
        wrong_dk = torch.softmax(raw / Dk, dim=-1)
        wrong_none = torch.softmax(raw, dim=-1)
        torch.testing.assert_close(attn, expected, rtol=1e-12, atol=1e-12)
        self.assertGreater((attn - wrong_dk).abs().max().item(), 1e-6)
        self.assertGreater((attn - wrong_none).abs().max().item(), 1e-6)

    def test_attention_weights_are_a_distribution(self):
        out, attn = scaled_dot_product_attention(
            torch.randn(2, 8, 5, 64), torch.randn(2, 8, 9, 64), torch.randn(2, 8, 9, 64)
        )
        self.assertEqual(tuple(out.shape), (2, 8, 5, 64))
        self.assertEqual(tuple(attn.shape), (2, 8, 5, 9))
        torch.testing.assert_close(attn.sum(-1), torch.ones(2, 8, 5), rtol=1e-6, atol=1e-6)
        self.assertTrue(bool((attn >= 0).all()))

    def test_dot_product_variance_grows_as_dk(self):
        """Empirical check of footnote 4, the stated motivation for 1/sqrt(dk).

        For q, k with iid unit-variance components, Var(q . k) = d_k. So the
        UNSCALED logits have std sqrt(d_k) and saturate softmax as d_k grows,
        while the scaled logits have std ~1 regardless of d_k.
        """
        torch.manual_seed(1)
        for dk in (8, 64, 512):
            q = torch.randn(20000, dk, dtype=torch.float64)
            k = torch.randn(20000, dk, dtype=torch.float64)
            dots = (q * k).sum(-1)
            self.assertAlmostEqual(dots.var().item() / dk, 1.0, delta=0.06)
            scaled = dots / math.sqrt(dk)
            self.assertAlmostEqual(scaled.var().item(), 1.0, delta=0.06)

    def test_scaling_prevents_softmax_saturation(self):
        """The consequence the paper cares about: gradient magnitude."""
        torch.manual_seed(2)
        dk = 512
        q = torch.randn(1, 1, 1, dk, dtype=torch.float64)
        k = torch.randn(1, 1, 32, dk, dtype=torch.float64)
        unscaled = torch.softmax((q @ k.transpose(-2, -1)).squeeze(), dim=-1)
        scaled = torch.softmax((q @ k.transpose(-2, -1)).squeeze() / math.sqrt(dk), dim=-1)
        # Entropy is a proxy for how peaked (saturated) the distribution is.
        ent = lambda p: float(-(p * p.clamp_min(1e-300).log()).sum())
        self.assertLess(ent(unscaled), ent(scaled))
        self.assertGreater(ent(scaled), 0.5 * math.log(32))


class TestMultiHeadAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_fused_projection_equals_per_head_loop(self):
        """h separate projections == one fused projection + reshape.

        This is the optimization that makes multi-head attention cheap; if
        the reshape/transpose order were wrong, heads would be interleaved
        incorrectly and this test would fail while shapes stayed valid.
        """
        d_model, h = 32, 4
        mha = MultiHeadAttention(d_model, h).double()
        x = torch.randn(2, 6, d_model, dtype=torch.float64)

        fused = mha(x, x, x)

        dk = d_model // h
        Wq, Wk, Wv = mha.w_q.weight, mha.w_k.weight, mha.w_v.weight
        heads = []
        for i in range(h):
            sl = slice(i * dk, (i + 1) * dk)
            # nn.Linear computes x @ W^T, so rows [sl] of W are head i's projection.
            q = x @ Wq[sl].T
            k = x @ Wk[sl].T
            v = x @ Wv[sl].T
            o, _ = scaled_dot_product_attention(q[:, None], k[:, None], v[:, None])
            heads.append(o[:, 0])
        manual = torch.cat(heads, dim=-1) @ mha.w_o.weight.T
        torch.testing.assert_close(fused, manual, rtol=1e-10, atol=1e-10)

    def test_output_shape_and_dimension_bookkeeping(self):
        mha = MultiHeadAttention(512, 8)
        self.assertEqual(mha.d_k, 64)          # d_k = d_model / h  (paper 3.2.2)
        self.assertEqual(mha.d_v, 64)
        q = torch.randn(3, 11, 512)
        kv = torch.randn(3, 17, 512)
        self.assertEqual(tuple(mha(q, kv, kv).shape), (3, 11, 512))

    def test_cross_attention_length_mismatch_allowed(self):
        """Encoder-decoder attention has Lq != Lk (Section 3.2.3)."""
        mha = MultiHeadAttention(64, 8)
        dec = torch.randn(2, 5, 64)
        enc = torch.randn(2, 13, 64)
        self.assertEqual(tuple(mha(dec, enc, enc).shape), (2, 5, 64))

    def test_reduced_dk_keeps_compute_comparable(self):
        """Table 3 row (B) reduces d_k without changing h."""
        mha = MultiHeadAttention(512, 8, d_k=16, d_v=16)
        self.assertEqual(mha.w_q.weight.shape[0], 8 * 16)
        self.assertEqual(tuple(mha(torch.randn(2, 4, 512), torch.randn(2, 4, 512),
                                   torch.randn(2, 4, 512)).shape), (2, 4, 512))

    def test_head_dimension_divisibility_enforced(self):
        with self.assertRaises(ValueError):
            MultiHeadAttention(512, 7)

    def test_permutation_equivariance_without_position_signal(self):
        """Self-attention alone is permutation-equivariant.

        This is exactly why Section 3.5 says position information "must" be
        injected. Verifying it here makes the positional-encoding ablation
        in Phase 8 interpretable rather than mysterious.
        """
        mha = MultiHeadAttention(32, 4).double().eval()
        x = torch.randn(1, 6, 32, dtype=torch.float64)
        perm = torch.randperm(6)
        a = mha(x, x, x)[:, perm]
        b = mha(x[:, perm], x[:, perm], x[:, perm])
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
