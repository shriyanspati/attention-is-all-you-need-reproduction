"""Verification of Section 3.5 positional encoding.

Beyond checking the formula, we test the two *claims* the paper makes about
these functions, since those claims are the stated reasons for choosing
them over learned embeddings.
"""

from __future__ import annotations

import math
import unittest

import torch

from ..model.positional_encoding import (
    LearnedPositionalEmbedding,
    NoPositionalEncoding,
    SinusoidalPositionalEncoding,
    build_position_encoding,
)


def naive_pe(max_len: int, d_model: int, base: float = 10000.0) -> torch.Tensor:
    """Literal double-loop transcription of the paper's two formulas."""
    pe = torch.zeros(max_len, d_model, dtype=torch.float64)
    for pos in range(max_len):
        for i in range(d_model // 2):
            denom = base ** (2 * i / d_model)
            pe[pos, 2 * i] = math.sin(pos / denom)
            pe[pos, 2 * i + 1] = math.cos(pos / denom)
    return pe


class TestSinusoidalPositionalEncoding(unittest.TestCase):
    def test_matches_naive_formula(self):
        d_model, L = 64, 40
        mod = SinusoidalPositionalEncoding(d_model, dropout=0.0, max_len=L)
        torch.testing.assert_close(
            mod.pe[0].double(), naive_pe(L, d_model), rtol=1e-6, atol=1e-6
        )

    def test_even_columns_sine_odd_columns_cosine(self):
        """Guards against the common (sin | cos) half-split layout, which is
        a different function from the paper's interleaved (sin, cos) pairs."""
        d_model = 16
        mod = SinusoidalPositionalEncoding(d_model, dropout=0.0, max_len=10)
        pe = mod.pe[0].double()
        # At pos = 0: sin(0) = 0 for every even column, cos(0) = 1 for every odd.
        torch.testing.assert_close(pe[0, 0::2], torch.zeros(d_model // 2, dtype=torch.float64),
                                   rtol=0, atol=1e-12)
        torch.testing.assert_close(pe[0, 1::2], torch.ones(d_model // 2, dtype=torch.float64),
                                   rtol=0, atol=1e-12)

    def test_paired_columns_share_a_frequency(self):
        """Column 2i and 2i+1 must use the SAME denominator."""
        d_model = 32
        pe = SinusoidalPositionalEncoding(d_model, dropout=0.0, max_len=50).pe[0].double()
        for i in range(d_model // 2):
            omega = 1.0 / (10000.0 ** (2 * i / d_model))
            for pos in (1, 7, 23):
                # sin^2 + cos^2 = 1 identifies a genuine (sin, cos) pair.
                s, c = pe[pos, 2 * i], pe[pos, 2 * i + 1]
                # The table is stored in float32 (the model's compute dtype),
                # so agreement is asserted to float32 precision (~1e-7), not
                # to the float64 precision of the reference computation.
                self.assertAlmostEqual(float(s * s + c * c), 1.0, places=6)
                self.assertAlmostEqual(float(s), math.sin(pos * omega), places=6)
                self.assertAlmostEqual(float(c), math.cos(pos * omega), places=6)

    def test_bounded_in_unit_interval(self):
        pe = SinusoidalPositionalEncoding(128, dropout=0.0, max_len=5000).pe
        self.assertLessEqual(float(pe.max()), 1.0 + 1e-6)
        self.assertGreaterEqual(float(pe.min()), -1.0 - 1e-6)

    def test_relative_offset_is_a_linear_function_of_position(self):
        """Paper claim: "for any fixed offset k, PE_{pos+k} can be represented
        as a linear function of PE_pos."

        Each (sin, cos) pair rotates by omega_i * k, so
        PE_{pos+k} = R(k) PE_pos with R(k) block-diagonal 2x2 rotations,
        independent of pos. We construct R(k) from the frequencies alone and
        verify it maps PE_pos to PE_{pos+k} for every position.
        """
        d_model, L = 32, 60
        pe = SinusoidalPositionalEncoding(d_model, dropout=0.0, max_len=L).pe[0].double()
        for k in (1, 3, 17):
            R = torch.zeros(d_model, d_model, dtype=torch.float64)
            for i in range(d_model // 2):
                omega = 1.0 / (10000.0 ** (2 * i / d_model))
                c, s = math.cos(omega * k), math.sin(omega * k)
                a, b = 2 * i, 2 * i + 1
                # [sin(w(p+k)); cos(w(p+k))] = [[c, s], [-s, c]] [sin(wp); cos(wp)]
                R[a, a], R[a, b] = c, s
                R[b, a], R[b, b] = -s, c
            pred = pe[: L - k] @ R.T
            # float32 storage -> ~1e-7 agreement; the identity itself is exact.
            torch.testing.assert_close(pred, pe[k:L], rtol=1e-5, atol=1e-6)

    def test_offset_matrix_is_position_independent(self):
        """The linear map above must be the SAME matrix for all positions,
        which is the property that makes relative attention learnable.

        Recovering R by least squares is numerically hopeless here (the
        low-frequency columns are nearly constant over any window, so the
        design matrix is severely ill-conditioned). Instead we verify the
        position-independence directly through the 2x2 block structure: for
        each frequency pair, the rotation angle between position p and p+k
        satisfies

            sin(w(p+k))cos(wp) - cos(w(p+k))sin(wp) = sin(wk)
            cos(w(p+k))cos(wp) + sin(w(p+k))sin(wp) = cos(wk)

        whose right-hand sides do not involve p at all. If the encoding used
        per-position frequencies, or mismatched sin/cos columns, these would
        vary with p.
        """
        d_model, L = 32, 60
        pe = SinusoidalPositionalEncoding(d_model, dropout=0.0, max_len=L).pe[0].double()
        for k in (1, 4, 11):
            for i in range(d_model // 2):
                a, b = 2 * i, 2 * i + 1
                sin_p, cos_p = pe[: L - k, a], pe[: L - k, b]
                sin_pk, cos_pk = pe[k:L, a], pe[k:L, b]
                sin_wk = sin_pk * cos_p - cos_pk * sin_p
                cos_wk = cos_pk * cos_p + sin_pk * sin_p
                # Constant across ALL positions p:
                self.assertLess(float(sin_wk.std()), 1e-6, f"pair {i}, k={k}")
                self.assertLess(float(cos_wk.std()), 1e-6, f"pair {i}, k={k}")
                omega = 1.0 / (10000.0 ** (2 * i / d_model))
                self.assertAlmostEqual(float(sin_wk.mean()), math.sin(omega * k), places=5)
                self.assertAlmostEqual(float(cos_wk.mean()), math.cos(omega * k), places=5)

    def test_wavelength_geometric_progression(self):
        """Paper: "The wavelengths form a geometric progression from 2*pi to
        10000 * 2*pi."

        The progression is exactly geometric. The lower endpoint is exactly
        2*pi; the upper endpoint is 2*pi * 10000^{(d-2)/d}, which only
        *approaches* 10000 * 2*pi. We assert the exact facts and record the
        endpoint imprecision rather than asserting the paper's rounding.
        """
        d_model = 512
        w = SinusoidalPositionalEncoding(d_model, dropout=0.0, max_len=8).wavelengths()
        self.assertAlmostEqual(float(w[0]), 2 * math.pi, places=10)
        ratios = w[1:] / w[:-1]
        torch.testing.assert_close(ratios, torch.full_like(ratios, float(ratios[0])),
                                   rtol=1e-10, atol=1e-10)
        upper = float(w[-1]) / (2 * math.pi)
        self.assertAlmostEqual(upper, 10000.0 ** ((d_model - 2) / d_model), places=6)
        # The largest wavelength is 2*pi * 10000^{510/512} = 9646.6 * 2*pi,
        # i.e. 3.5% BELOW the paper's stated upper endpoint of 10000 * 2*pi.
        # The progression is exactly geometric; only the stated endpoint is
        # approximate. We assert the true value rather than the paper's.
        self.assertLess(upper, 10000.0)
        self.assertAlmostEqual(upper, 9646.6162, places=3)
        self.assertGreater(upper / 10000.0, 0.96)

    def test_added_not_concatenated(self):
        """Section 3.5: encodings "have the same dimension d_model as the
        embeddings, so that the two can be summed"."""
        d_model = 32
        mod = SinusoidalPositionalEncoding(d_model, dropout=0.0, max_len=20)
        x = torch.zeros(2, 7, d_model)
        out = mod(x)
        self.assertEqual(tuple(out.shape), (2, 7, d_model))
        torch.testing.assert_close(out[0], mod.pe[0, :7], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(out[0], out[1], rtol=0, atol=0)   # same for all batch items

    def test_buffer_is_not_a_trained_parameter(self):
        mod = SinusoidalPositionalEncoding(32, dropout=0.0, max_len=10)
        self.assertEqual(list(mod.parameters()), [])

    def test_extends_beyond_table_for_longer_sequences(self):
        """Sinusoids are defined for any position; the paper cites this
        extrapolation ability as the reason for preferring them."""
        mod = SinusoidalPositionalEncoding(16, dropout=0.0, max_len=8)
        out = mod(torch.zeros(1, 50, 16))
        self.assertEqual(out.size(1), 50)
        torch.testing.assert_close(mod.pe[0, :50].double(), naive_pe(50, 16),
                                   rtol=1e-6, atol=1e-6)


class TestAlternatives(unittest.TestCase):
    def test_learned_variant_has_trainable_parameters(self):
        mod = LearnedPositionalEmbedding(32, dropout=0.0, max_len=64)
        self.assertEqual(mod.emb.weight.shape, (64, 32))
        self.assertTrue(mod.emb.weight.requires_grad)

    def test_none_variant_is_identity_in_eval(self):
        mod = NoPositionalEncoding(32, dropout=0.0).eval()
        x = torch.randn(2, 5, 32)
        torch.testing.assert_close(mod(x), x, rtol=0, atol=0)

    def test_factory(self):
        for kind, cls in [("sinusoidal", SinusoidalPositionalEncoding),
                          ("learned", LearnedPositionalEmbedding),
                          ("none", NoPositionalEncoding)]:
            self.assertIsInstance(build_position_encoding(kind, 16, 0.1), cls)
        with self.assertRaises(ValueError):
            build_position_encoding("rotary", 16, 0.1)


if __name__ == "__main__":
    unittest.main()
