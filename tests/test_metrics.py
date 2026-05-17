"""Tests for `patchforge.patch_metrics`, `per_patch_mse`, `per_patch_psnr`.

Contract: docs/THEORY.md §1.6 and §9.7.
"""
from __future__ import annotations

import math

import pytest
import torch

from patchforge import patch_metrics, per_patch_mse, per_patch_psnr

# ----------------------------------------------------------- patch_metrics ---

class TestPatchMetricsAccepts:
    def test_identical_inputs(self) -> None:
        a = torch.rand(1, 4, 4)
        m = patch_metrics(a, a)
        assert m["mae"] == 0.0
        assert m["mse"] == 0.0
        assert m["max_abs"] == 0.0
        assert m["psnr_db"] == float("inf")

    def test_known_mse(self) -> None:
        a = torch.zeros(1, 2, 2)
        b = torch.ones(1, 2, 2)
        m = patch_metrics(a, b, max_value=1.0)
        assert m["mae"] == pytest.approx(1.0)
        assert m["mse"] == pytest.approx(1.0)
        assert m["max_abs"] == pytest.approx(1.0)
        # PSNR = 10 log10(1/1) = 0 dB
        assert m["psnr_db"] == pytest.approx(0.0)

    def test_psnr_increases_with_smaller_error(self) -> None:
        a = torch.zeros(1, 4, 4)
        b_small = torch.full_like(a, 0.01)
        b_big = torch.full_like(a, 0.1)
        m_small = patch_metrics(a, b_small)
        m_big = patch_metrics(a, b_big)
        assert m_small["psnr_db"] > m_big["psnr_db"]

    def test_max_value_affects_psnr(self) -> None:
        a = torch.zeros(1, 4, 4)
        b = torch.full_like(a, 1.0)
        m1 = patch_metrics(a, b, max_value=1.0)
        m255 = patch_metrics(a, b, max_value=255.0)
        # MSE = 1.0; PSNR(255) - PSNR(1) = 10*log10(255^2) ≈ 48.13 dB
        assert m255["psnr_db"] - m1["psnr_db"] == pytest.approx(
            10 * math.log10(255 * 255), rel=1e-9
        )

    def test_works_on_patch_stack(self) -> None:
        # (L, C, h, w) is just "any same shape"
        a = torch.rand(16, 3, 4, 4)
        b = a + 0.001
        m = patch_metrics(a, b)
        assert m["max_abs"] == pytest.approx(0.001, abs=1e-6)

    def test_promotes_internally_to_float64(self) -> None:
        # float16 inputs would lose precision in accumulation; check result
        # is plausibly stable (not nan / inf for small differences)
        a = torch.zeros(1, 64, 64, dtype=torch.float32)
        b = torch.full_like(a, 1e-4)
        m = patch_metrics(a, b)
        assert math.isfinite(m["mse"])
        assert m["mse"] > 0


class TestPatchMetricsRejects:
    def test_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="shape mismatch"):
            patch_metrics(torch.zeros(1, 4, 4), torch.zeros(1, 4, 5))

    def test_dtype_mismatch(self) -> None:
        with pytest.raises(ValueError, match="dtype mismatch"):
            patch_metrics(
                torch.zeros(1, 4, 4, dtype=torch.float32),
                torch.zeros(1, 4, 4, dtype=torch.float64),
            )

    def test_non_tensor(self) -> None:
        with pytest.raises(TypeError, match=r"must be torch\.Tensor"):
            patch_metrics([1, 2], torch.zeros(2))  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [0, -1, float("inf"), float("nan"), "1"])
    def test_invalid_max_value(self, bad: object) -> None:
        a = torch.zeros(1, 4, 4)
        with pytest.raises(ValueError, match="max_value"):
            patch_metrics(a, a, max_value=bad)  # type: ignore[arg-type]


# ------------------------------------------------------------ per_patch_mse ---

class TestPerPatchMSE:
    def test_returns_one_per_patch(self) -> None:
        a = torch.zeros(5, 1, 4, 4)
        b = torch.ones(5, 1, 4, 4)
        out = per_patch_mse(a, b)
        assert out.shape == (5,)
        assert torch.allclose(out, torch.ones(5))

    def test_per_patch_varies(self) -> None:
        a = torch.zeros(3, 1, 2, 2)
        b = a.clone()
        b[0] = 0.0  # MSE 0
        b[1] = 0.5  # MSE 0.25
        b[2] = 1.0  # MSE 1
        out = per_patch_mse(a, b)
        assert torch.allclose(out, torch.tensor([0.0, 0.25, 1.0]))

    def test_rejects_non_4d(self) -> None:
        with pytest.raises(ValueError, match="4-D"):
            per_patch_mse(torch.zeros(3, 4, 4), torch.zeros(3, 4, 4))


# ----------------------------------------------------------- per_patch_psnr ---

class TestPerPatchPSNR:
    def test_returns_one_per_patch(self) -> None:
        a = torch.zeros(4, 1, 2, 2)
        b = torch.full_like(a, 0.1)
        out = per_patch_psnr(a, b)
        assert out.shape == (4,)
        # All four patches have identical MSE so identical PSNR
        assert torch.allclose(out, out[0].expand(4))

    def test_identical_patches_infinite_psnr(self) -> None:
        a = torch.rand(3, 1, 4, 4)
        out = per_patch_psnr(a, a)
        assert torch.all(torch.isinf(out))
        assert torch.all(out > 0)  # +inf, not -inf

    def test_mixed_inf_and_finite(self) -> None:
        a = torch.zeros(2, 1, 2, 2)
        b = a.clone()
        b[1] = 0.5  # second patch differs
        out = per_patch_psnr(a, b)
        assert torch.isinf(out[0]) and out[0] > 0
        assert torch.isfinite(out[1])

    def test_max_value_shifts_all(self) -> None:
        a = torch.zeros(2, 1, 2, 2, dtype=torch.float64)
        b = torch.full_like(a, 0.1)
        out1 = per_patch_psnr(a, b, max_value=1.0)
        out255 = per_patch_psnr(a, b, max_value=255.0)
        diff = (out255 - out1).unique()
        assert diff.shape == (1,)
        assert diff[0].item() == pytest.approx(
            10 * math.log10(255 * 255), rel=1e-12
        )

    def test_rejects_non_4d(self) -> None:
        with pytest.raises(ValueError, match="4-D"):
            per_patch_psnr(torch.zeros(3, 4, 4), torch.zeros(3, 4, 4))


# ------------------------------------------------------ integration with M3 ---

class TestRoundtripIntegration:
    """patch_metrics is the natural way to validate extract + reconstruct."""

    def test_exact_roundtrip_psnr_is_infinite(self) -> None:
        from patchforge import extract, reconstruct
        img = torch.arange(28 * 28, dtype=torch.float64).reshape(1, 28, 28)
        patches = extract(img, patch_size=7, stride=7)
        recon = reconstruct(patches, image_shape=img.shape, stride=7)
        m = patch_metrics(img, recon)
        assert m["psnr_db"] == float("inf")
        assert m["max_abs"] == 0.0
