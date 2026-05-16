"""Tests for `patchkit.pair` — contract from docs/THEORY.md §3 and §9.3."""
from __future__ import annotations

import pytest
import torch

from patchkit import PatchMeta, PatchPair, extract, pair


def _ramp(c: int, h: int, w: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.arange(c * h * w, dtype=dtype).reshape(c, h, w)


def _make_lr_hr(
    c: int, h_lr: int, w_lr: int, scale: int, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    lr = _ramp(c, h_lr, w_lr, dtype=dtype)
    hr = _ramp(c, h_lr * scale, w_lr * scale, dtype=dtype)
    return lr, hr


# ------------------------------------------------------------------ Aceita ----

class TestAccepts:
    def test_basic_2x_scale(self) -> None:
        lr, hr = _make_lr_hr(1, 8, 8, scale=2)
        result = pair(lr, hr, lr_patch_size=4, scale_factor=2, stride=4)
        assert isinstance(result, PatchPair)
        # num_h_lr = (8 - 4) // 4 + 1 = 2; total = 4
        assert result.lr_patches.shape == (4, 1, 4, 4)
        assert result.hr_patches.shape == (4, 1, 8, 8)
        assert len(result.metas) == 4

    def test_patch_count_matches_extract(self) -> None:
        lr, hr = _make_lr_hr(3, 16, 16, scale=3)
        result = pair(lr, hr, lr_patch_size=4, scale_factor=3, stride=2)
        expected_lr = extract(lr, patch_size=4, stride=2)
        expected_hr = extract(hr, patch_size=12, stride=6)
        assert torch.equal(result.lr_patches, expected_lr)
        assert torch.equal(result.hr_patches, expected_hr)

    def test_scale_1_is_valid(self) -> None:
        """scale_factor=1 means LR == HR (degenerate but valid)."""
        img = _ramp(1, 8, 8)
        result = pair(img, img, lr_patch_size=4, scale_factor=1, stride=4)
        assert result.lr_patches.shape == result.hr_patches.shape
        assert torch.equal(result.lr_patches, result.hr_patches)

    def test_rectangular_patch_and_stride(self) -> None:
        lr, hr = _make_lr_hr(1, 12, 18, scale=2)
        result = pair(lr, hr, lr_patch_size=(3, 6), scale_factor=2, stride=(3, 6))
        # num_h = (12-3)//3 + 1 = 4; num_w = (18-6)//6 + 1 = 3; total = 12
        assert result.lr_patches.shape == (12, 1, 3, 6)
        assert result.hr_patches.shape == (12, 1, 6, 12)

    def test_multichannel(self) -> None:
        lr, hr = _make_lr_hr(3, 8, 8, scale=4)
        result = pair(lr, hr, lr_patch_size=4, scale_factor=4, stride=4)
        assert result.lr_patches.shape == (4, 3, 4, 4)
        assert result.hr_patches.shape == (4, 3, 16, 16)

    def test_dtype_preserved(self) -> None:
        lr, hr = _make_lr_hr(1, 8, 8, scale=2, dtype=torch.float64)
        result = pair(lr, hr, lr_patch_size=4, scale_factor=2, stride=4)
        assert result.lr_patches.dtype == torch.float64
        assert result.hr_patches.dtype == torch.float64

    def test_empty_grid_when_geometry_too_big(self) -> None:
        """patch > LR image: both sides return empty (consistent with extract)."""
        lr, hr = _make_lr_hr(1, 4, 4, scale=2)
        result = pair(lr, hr, lr_patch_size=8, scale_factor=2, stride=4)
        assert result.lr_patches.shape[0] == 0
        assert result.hr_patches.shape[0] == 0
        assert result.metas == ()


# ----------------------------------------------------------- Region alignment --

class TestSameRegion:
    """The k-th LR patch and k-th HR patch cover the same image region."""

    def test_patches_are_subviews_of_their_images(self) -> None:
        lr, hr = _make_lr_hr(1, 8, 8, scale=2)
        result = pair(lr, hr, lr_patch_size=4, scale_factor=2, stride=4)
        for meta, lr_p, hr_p in zip(
            result.metas, result.lr_patches, result.hr_patches, strict=True
        ):
            r, c = meta.row, meta.col
            ph_lr, pw_lr = meta.lr_patch_size
            ph_hr, pw_hr = meta.hr_patch_size
            # LR patch matches the underlying region in LR coords
            assert torch.equal(lr_p, lr[:, r : r + ph_lr, c : c + pw_lr])
            # HR patch matches the SAME region in HR coords (scaled)
            r_hr, c_hr = r * 2, c * 2
            assert torch.equal(hr_p, hr[:, r_hr : r_hr + ph_hr, c_hr : c_hr + pw_hr])

    def test_row_major_ordering(self) -> None:
        lr, hr = _make_lr_hr(1, 16, 16, scale=2)
        result = pair(lr, hr, lr_patch_size=4, scale_factor=2, stride=4)
        # num_w_lr = (16-4)//4 + 1 = 4; patch 5 is at row=1, col=1 → (4, 4) in LR
        meta_5 = result.metas[5]
        assert (meta_5.row, meta_5.col) == (4, 4)


# ---------------------------------------------------------------- PatchMeta ---

class TestPatchMeta:
    def test_immutable(self) -> None:
        m = PatchMeta(
            patch_index=0, row=0, col=0,
            lr_patch_size=(4, 4), hr_patch_size=(8, 8),
        )
        with pytest.raises((AttributeError, TypeError)):
            m.row = 5  # type: ignore[misc]

    def test_image_id_propagated(self) -> None:
        lr, hr = _make_lr_hr(1, 8, 8, scale=2)
        result = pair(
            lr, hr, lr_patch_size=4, scale_factor=2, stride=4, image_id="mnist-7"
        )
        assert all(m.image_id == "mnist-7" for m in result.metas)

    def test_image_id_default_is_none(self) -> None:
        lr, hr = _make_lr_hr(1, 8, 8, scale=2)
        result = pair(lr, hr, lr_patch_size=4, scale_factor=2, stride=4)
        assert all(m.image_id is None for m in result.metas)

    def test_hr_size_derived(self) -> None:
        lr, hr = _make_lr_hr(1, 8, 8, scale=3)
        result = pair(lr, hr, lr_patch_size=(2, 4), scale_factor=3, stride=2)
        for m in result.metas:
            assert m.hr_patch_size == (m.lr_patch_size[0] * 3, m.lr_patch_size[1] * 3)


# ------------------------------------------------------------------ Rejeita ---

class TestRejects:
    def test_lr_not_tensor(self) -> None:
        with pytest.raises(TypeError, match="lr_image"):
            pair([1, 2], torch.zeros(1, 8, 8), 4, 2, 4)  # type: ignore[arg-type]

    def test_hr_not_tensor(self) -> None:
        with pytest.raises(TypeError, match="hr_image"):
            pair(torch.zeros(1, 4, 4), [1, 2], 4, 2, 4)  # type: ignore[arg-type]

    @pytest.mark.parametrize("ndim", [1, 2, 4])
    def test_lr_wrong_ndim(self, ndim: int) -> None:
        lr = torch.zeros([4] * ndim)
        hr = torch.zeros(1, 8, 8)
        with pytest.raises(ValueError, match=r"lr_image.*ndim==3"):
            pair(lr, hr, 4, 2, 4)

    @pytest.mark.parametrize("bad", [0, -1, 1.5, "2"])
    def test_scale_factor_invalid(self, bad: object) -> None:
        lr, hr = _make_lr_hr(1, 4, 4, scale=2)
        with pytest.raises(ValueError, match="scale_factor"):
            pair(lr, hr, 4, bad, 4)  # type: ignore[arg-type]

    def test_scale_factor_bool_rejected(self) -> None:
        lr, hr = _make_lr_hr(1, 4, 4, scale=2)
        with pytest.raises(ValueError, match="scale_factor"):
            pair(lr, hr, 4, True, 4)  # type: ignore[arg-type]

    def test_hr_shape_mismatch(self) -> None:
        lr = torch.zeros(1, 4, 4)
        hr = torch.zeros(1, 7, 8)  # wrong height
        with pytest.raises(ValueError, match="hr_image shape"):
            pair(lr, hr, 4, 2, 4)

    def test_channel_mismatch(self) -> None:
        lr = torch.zeros(1, 4, 4)
        hr = torch.zeros(3, 8, 8)
        with pytest.raises(ValueError, match="channel mismatch"):
            pair(lr, hr, 4, 2, 4)

    def test_dtype_mismatch(self) -> None:
        lr = torch.zeros(1, 4, 4, dtype=torch.float32)
        hr = torch.zeros(1, 8, 8, dtype=torch.float64)
        with pytest.raises(ValueError, match="dtype mismatch"):
            pair(lr, hr, 4, 2, 4)

    def test_lr_patch_size_nonpositive(self) -> None:
        lr, hr = _make_lr_hr(1, 8, 8, scale=2)
        with pytest.raises(ValueError, match="lr_patch_size must be positive"):
            pair(lr, hr, 0, 2, 4)

    def test_stride_nonpositive(self) -> None:
        lr, hr = _make_lr_hr(1, 8, 8, scale=2)
        with pytest.raises(ValueError, match="stride must be positive"):
            pair(lr, hr, 4, 2, 0)


# -------------------------------------------------------------- PatchPair API --

class TestPatchPair:
    def test_len(self) -> None:
        lr, hr = _make_lr_hr(1, 8, 8, scale=2)
        result = pair(lr, hr, lr_patch_size=4, scale_factor=2, stride=4)
        assert len(result) == 4

    def test_zip_iteration_works(self) -> None:
        lr, hr = _make_lr_hr(1, 8, 8, scale=2)
        result = pair(lr, hr, lr_patch_size=4, scale_factor=2, stride=4)
        count = 0
        for lr_p, hr_p, m in zip(
            result.lr_patches, result.hr_patches, result.metas, strict=True
        ):
            count += 1
            assert lr_p.shape == (1, 4, 4)
            assert hr_p.shape == (1, 8, 8)
            assert isinstance(m, PatchMeta)
        assert count == 4
