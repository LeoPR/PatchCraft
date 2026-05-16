"""Tests for `patchkit.reconstruct` — contract from docs/THEORY.md §9.2."""
from __future__ import annotations

import pytest
import torch

from patchkit import extract, reconstruct


def _ramp(c: int, h: int, w: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Image with every pixel unique — distinguishes pixels positionally."""
    return torch.arange(c * h * w, dtype=dtype).reshape(c, h, w)


# ----------------------------------------------------------------- Roundtrip --

class TestRoundtripExact:
    """`stride == patch_size`: each pixel covered exactly once, bit-exact."""

    def test_basic(self) -> None:
        img = _ramp(3, 32, 32)
        patches = extract(img, patch_size=8, stride=8)
        out = reconstruct(patches, image_shape=img.shape, stride=8)
        assert torch.equal(out, img)

    def test_rectangular_geometry(self) -> None:
        img = _ramp(1, 20, 30)
        patches = extract(img, patch_size=(4, 6), stride=(4, 6))
        out = reconstruct(patches, image_shape=img.shape, stride=(4, 6))
        assert torch.equal(out, img)

    def test_single_patch_equals_image(self) -> None:
        img = _ramp(2, 8, 8)
        patches = extract(img, patch_size=8, stride=8)
        out = reconstruct(patches, image_shape=img.shape, stride=8)
        assert torch.equal(out, img)

    def test_multichannel(self) -> None:
        img = _ramp(7, 16, 16)
        patches = extract(img, patch_size=4, stride=4)
        out = reconstruct(patches, image_shape=img.shape, stride=4)
        assert torch.equal(out, img)

    def test_patch_size_1(self) -> None:
        img = _ramp(1, 4, 4)
        patches = extract(img, patch_size=1, stride=1)
        out = reconstruct(patches, image_shape=img.shape, stride=1)
        assert torch.equal(out, img)


class TestRoundtripOverlap:
    """`stride < patch_size`: overlap; weighted reconstruction = original."""

    def test_half_overlap_basic(self) -> None:
        img = _ramp(1, 16, 16, dtype=torch.float64)
        patches = extract(img, patch_size=4, stride=2)
        out = reconstruct(patches, image_shape=img.shape, stride=2)
        assert torch.allclose(out, img, rtol=1e-12, atol=1e-12)

    def test_max_overlap_stride_1(self) -> None:
        img = _ramp(2, 8, 8, dtype=torch.float64)
        patches = extract(img, patch_size=3, stride=1)
        out = reconstruct(patches, image_shape=img.shape, stride=1)
        assert torch.allclose(out, img, rtol=1e-12, atol=1e-12)

    def test_asymmetric_overlap(self) -> None:
        img = _ramp(1, 12, 18, dtype=torch.float64)
        patches = extract(img, patch_size=(4, 6), stride=(2, 3))
        out = reconstruct(patches, image_shape=img.shape, stride=(2, 3))
        assert torch.allclose(out, img, rtol=1e-12, atol=1e-12)

    def test_float32_overlap_close(self) -> None:
        """float32 round-trip survives the divide-by-count step within rtol=1e-5."""
        img = _ramp(1, 16, 16, dtype=torch.float32)
        patches = extract(img, patch_size=4, stride=2)
        out = reconstruct(patches, image_shape=img.shape, stride=2)
        assert torch.allclose(out, img, rtol=1e-5, atol=1e-5)


class TestRoundtripPreservation:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preserved(self, dtype: torch.dtype) -> None:
        img = _ramp(1, 8, 8, dtype=dtype)
        patches = extract(img, patch_size=4, stride=4)
        out = reconstruct(patches, image_shape=img.shape, stride=4)
        assert out.dtype == dtype

    def test_device_preserved_cpu(self) -> None:
        img = _ramp(1, 8, 8)
        patches = extract(img, patch_size=4, stride=4)
        out = reconstruct(patches, image_shape=img.shape, stride=4)
        assert out.device == patches.device

    @pytest.mark.gpu
    def test_cuda_roundtrip(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        img = _ramp(1, 8, 8).cuda()
        patches = extract(img, patch_size=4, stride=4)
        out = reconstruct(patches, image_shape=img.shape, stride=4)
        assert out.device.type == "cuda"
        assert torch.equal(out, img)

    def test_accepts_torch_size(self) -> None:
        """torch.Tensor.shape returns torch.Size (a tuple subclass)."""
        img = _ramp(1, 8, 8)
        patches = extract(img, patch_size=4, stride=4)
        out = reconstruct(patches, image_shape=img.shape, stride=4)
        assert isinstance(img.shape, torch.Size)
        assert torch.equal(out, img)


class TestCountMap:
    """Count map correctness: independent verification with all-ones input."""

    def test_uniform_image_reconstructs_uniformly(self) -> None:
        """All-ones image: count-map division must produce all-ones out."""
        img = torch.ones(1, 8, 8)
        patches = extract(img, patch_size=4, stride=2)
        out = reconstruct(patches, image_shape=img.shape, stride=2)
        assert torch.allclose(out, img)

    def test_full_coverage_count_minimum_at_corners(self) -> None:
        """For 4x4 patch, stride 2, 8x8: corner pixel covered by 1 patch only."""
        img = torch.ones(1, 8, 8)
        patches = extract(img, patch_size=4, stride=2)
        out = reconstruct(patches, image_shape=img.shape, stride=2)
        # If count map at corners were wrong (e.g. 0), out[0, 0] would diverge.
        assert out[0, 0, 0].item() == pytest.approx(1.0)
        assert out[0, 7, 7].item() == pytest.approx(1.0)
        assert out[0, 4, 4].item() == pytest.approx(1.0)  # interior


# ------------------------------------------------------------------ Rejeita ---

class TestRejects:
    def test_patches_not_tensor(self) -> None:
        with pytest.raises(TypeError, match=r"must be torch\.Tensor"):
            reconstruct([1, 2, 3], image_shape=(1, 4, 4), stride=2)  # type: ignore[arg-type]

    @pytest.mark.parametrize("ndim", [1, 2, 3, 5])
    def test_patches_wrong_ndim(self, ndim: int) -> None:
        patches = torch.zeros([4] * ndim)
        with pytest.raises(ValueError, match="ndim==4"):
            reconstruct(patches, image_shape=(1, 16, 16), stride=4)

    def test_dilation_rejected(self) -> None:
        patches = torch.zeros(16, 1, 4, 4)
        with pytest.raises(ValueError, match="dilation==1"):
            reconstruct(patches, image_shape=(1, 16, 16), stride=4, dilation=2)

    def test_stride_greater_than_patch_rejected(self) -> None:
        """§9.2: sh > ph or sw > pw forbidden — partial coverage."""
        patches = torch.zeros(9, 1, 4, 4)
        with pytest.raises(ValueError, match="stride > patch_size"):
            reconstruct(patches, image_shape=(1, 20, 20), stride=6)

    def test_stride_greater_than_patch_one_axis(self) -> None:
        patches = torch.zeros(9, 1, 4, 4)
        with pytest.raises(ValueError, match="stride > patch_size"):
            reconstruct(patches, image_shape=(1, 16, 20), stride=(4, 6))

    @pytest.mark.parametrize("bad", [(1, 4), (1, 4, 4, 4), 4])
    def test_image_shape_wrong_arity(self, bad: object) -> None:
        patches = torch.zeros(4, 1, 4, 4)
        with pytest.raises(ValueError, match="3-tuple"):
            reconstruct(patches, image_shape=bad, stride=4)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [(1, 0, 4), (1, -4, 4), (0, 8, 8)])
    def test_image_shape_nonpositive(self, bad: tuple[int, int, int]) -> None:
        patches = torch.zeros(4, 1, 4, 4)
        with pytest.raises(ValueError, match="positive int"):
            reconstruct(patches, image_shape=bad, stride=4)

    def test_image_shape_non_int(self) -> None:
        patches = torch.zeros(4, 1, 4, 4)
        with pytest.raises(ValueError, match="positive int"):
            reconstruct(patches, image_shape=(1, 8.0, 8), stride=4)  # type: ignore[arg-type]

    def test_channel_mismatch(self) -> None:
        patches = torch.zeros(4, 3, 4, 4)
        with pytest.raises(ValueError, match="channel count"):
            reconstruct(patches, image_shape=(1, 8, 8), stride=4)

    def test_n_patches_too_few_for_geometry(self) -> None:
        """patches.shape[0] smaller than the grid implied by image_shape."""
        patches = torch.zeros(4, 1, 4, 4)
        with pytest.raises(ValueError, match="inconsistent with"):
            reconstruct(patches, image_shape=(1, 16, 16), stride=4)

    def test_n_patches_too_many_for_geometry(self) -> None:
        """patches.shape[0] larger than the grid implied by image_shape."""
        patches = torch.zeros(100, 1, 4, 4)
        with pytest.raises(ValueError, match="inconsistent with"):
            reconstruct(patches, image_shape=(1, 8, 8), stride=4)

    def test_image_too_small_for_patch(self) -> None:
        patches = torch.zeros(1, 1, 4, 4)
        with pytest.raises(ValueError, match="too small"):
            reconstruct(patches, image_shape=(1, 2, 2), stride=4)

    def test_stride_nonpositive(self) -> None:
        patches = torch.zeros(4, 1, 4, 4)
        with pytest.raises(ValueError, match="stride must be positive"):
            reconstruct(patches, image_shape=(1, 8, 8), stride=0)
