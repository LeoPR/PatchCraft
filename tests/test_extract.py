"""Tests for `patchkit.extract` — contract from docs/THEORY.md §10.1."""
from __future__ import annotations

import pytest
import torch

from patchkit import extract


def _ramp(c: int, h: int, w: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Deterministic image with unique pixel values for round-trip-style checks."""
    return torch.arange(c * h * w, dtype=dtype).reshape(c, h, w)


# ------------------------------------------------------------------ Aceita ----

class TestAccepts:
    def test_basic_shape(self) -> None:
        img = _ramp(3, 32, 32)
        out = extract(img, patch_size=8, stride=8)
        assert out.shape == (16, 3, 8, 8)

    def test_rectangular_patch_and_stride(self) -> None:
        img = _ramp(1, 20, 30)
        out = extract(img, patch_size=(4, 6), stride=(2, 3))
        # num_h = (20 - 4) // 2 + 1 = 9; num_w = (30 - 6) // 3 + 1 = 9
        assert out.shape == (81, 1, 4, 6)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
    def test_dtype_preserved_float(self, dtype: torch.dtype) -> None:
        img = torch.zeros(2, 8, 8, dtype=dtype)
        out = extract(img, patch_size=4, stride=4)
        assert out.dtype == dtype

    def test_dtype_uint8_unsupported_on_cpu(self) -> None:
        """torch.nn.functional.unfold's im2col_cpu has no Byte impl (torch 2.x).

        Documented limitation: caller must convert uint8 → float before extract.
        Listed in THEORY.md §10.1.
        """
        img = torch.zeros(1, 8, 8, dtype=torch.uint8)
        with pytest.raises(NotImplementedError):
            extract(img, patch_size=4, stride=4)

    def test_device_preserved_cpu(self) -> None:
        img = _ramp(1, 8, 8)
        out = extract(img, patch_size=4, stride=4)
        assert out.device == img.device

    @pytest.mark.gpu
    def test_device_preserved_cuda(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        img = _ramp(1, 8, 8).cuda()
        out = extract(img, patch_size=4, stride=4)
        assert out.device.type == "cuda"

    def test_single_channel(self) -> None:
        img = _ramp(1, 16, 16)
        out = extract(img, patch_size=4, stride=4)
        assert out.shape == (16, 1, 4, 4)

    def test_many_channels(self) -> None:
        img = _ramp(7, 16, 16)
        out = extract(img, patch_size=4, stride=4)
        assert out.shape == (16, 7, 4, 4)

    def test_dilation_gt_1(self) -> None:
        img = _ramp(1, 16, 16)
        # eff_h = 2 * (3 - 1) + 1 = 5; num_h = (16 - 5) // 1 + 1 = 12
        out = extract(img, patch_size=3, stride=1, dilation=2)
        assert out.shape == (144, 1, 3, 3)

    def test_stride_gt_patch_sparse_grid(self) -> None:
        """§10.1: stride > patch_size is allowed in extract (rejected in reconstruct)."""
        img = _ramp(1, 20, 20)
        # num_h = (20 - 4) // 6 + 1 = 3
        out = extract(img, patch_size=4, stride=6)
        assert out.shape == (9, 1, 4, 4)

    def test_patch_equals_image(self) -> None:
        img = _ramp(2, 8, 8)
        out = extract(img, patch_size=8, stride=8)
        assert out.shape == (1, 2, 8, 8)
        assert torch.equal(out[0], img)

    def test_patch_size_1(self) -> None:
        img = _ramp(1, 4, 4)
        out = extract(img, patch_size=1, stride=1)
        assert out.shape == (16, 1, 1, 1)
        # pixel-wise reshape: out[k, 0, 0, 0] == img.flatten()[k]
        assert torch.equal(out.flatten(), img.flatten())

    def test_patch_larger_than_image_returns_empty(self) -> None:
        img = _ramp(3, 4, 4)
        out = extract(img, patch_size=8, stride=1)
        assert out.shape == (0, 3, 8, 8)
        assert out.dtype == img.dtype
        assert out.device == img.device

    def test_patch_larger_only_one_axis(self) -> None:
        img = _ramp(1, 4, 16)
        out = extract(img, patch_size=(8, 4), stride=4)
        assert out.shape == (0, 1, 8, 4)

    def test_noncontiguous_input(self) -> None:
        img = _ramp(3, 16, 16).permute(0, 2, 1)  # still (3, 16, 16) but non-contig
        assert not img.is_contiguous()
        out = extract(img, patch_size=4, stride=4)
        assert out.shape == (16, 3, 4, 4)

    def test_truncation_drops_trailing(self) -> None:
        """Image 10x10, patch 4, stride 3 → trailing column dropped."""
        img = _ramp(1, 10, 10)
        # num_h = num_w = (10 - 4) // 3 + 1 = 3
        out = extract(img, patch_size=4, stride=3)
        assert out.shape == (9, 1, 4, 4)

    def test_row_major_ordering(self) -> None:
        """Patch k at (row, col) with row = k // num_w."""
        img = _ramp(1, 8, 8)
        out = extract(img, patch_size=2, stride=2)
        # num_w = 4; patch 5 → row=1, col=1 → top-left at (2, 2)
        expected = img[:, 2:4, 2:4]
        assert torch.equal(out[5], expected)


# ------------------------------------------------------------------ Rejeita ---

class TestRejects:
    def test_image_not_tensor(self) -> None:
        with pytest.raises(TypeError, match=r"must be torch\.Tensor"):
            extract([1, 2, 3], patch_size=2, stride=2)  # type: ignore[arg-type]

    @pytest.mark.parametrize("ndim", [1, 2, 4, 5])
    def test_image_wrong_ndim(self, ndim: int) -> None:
        img = torch.zeros([4] * ndim)
        with pytest.raises(ValueError, match="ndim==3"):
            extract(img, patch_size=2, stride=2)

    @pytest.mark.parametrize("bad", [0, -1, -100])
    def test_patch_size_nonpositive_int(self, bad: int) -> None:
        img = _ramp(1, 8, 8)
        with pytest.raises(ValueError, match="patch_size must be positive"):
            extract(img, patch_size=bad, stride=2)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_stride_nonpositive_int(self, bad: int) -> None:
        img = _ramp(1, 8, 8)
        with pytest.raises(ValueError, match="stride must be positive"):
            extract(img, patch_size=2, stride=bad)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_dilation_nonpositive_int(self, bad: int) -> None:
        img = _ramp(1, 8, 8)
        with pytest.raises(ValueError, match="dilation must be positive"):
            extract(img, patch_size=2, stride=2, dilation=bad)

    def test_patch_size_nonpositive_tuple(self) -> None:
        img = _ramp(1, 8, 8)
        with pytest.raises(ValueError, match="patch_size must be positive"):
            extract(img, patch_size=(0, 4), stride=2)

    @pytest.mark.parametrize("bad", [(2,), (2, 2, 2), (1, 2, 3, 4)])
    def test_patch_size_wrong_tuple_length(self, bad: tuple[int, ...]) -> None:
        img = _ramp(1, 8, 8)
        with pytest.raises(ValueError, match="must be int or"):
            extract(img, patch_size=bad, stride=2)  # type: ignore[arg-type]

    def test_patch_size_non_int_tuple(self) -> None:
        img = _ramp(1, 8, 8)
        with pytest.raises(ValueError, match="must contain ints"):
            extract(img, patch_size=(2.5, 2), stride=2)  # type: ignore[arg-type]

    def test_patch_size_string(self) -> None:
        img = _ramp(1, 8, 8)
        with pytest.raises(ValueError, match="must be int or"):
            extract(img, patch_size="4", stride=2)  # type: ignore[arg-type]

    def test_patch_size_bool_rejected(self) -> None:
        """`True` is technically `int` in Python; we want to reject it explicitly."""
        img = _ramp(1, 8, 8)
        with pytest.raises(ValueError):
            extract(img, patch_size=True, stride=2)  # type: ignore[arg-type]
