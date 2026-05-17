"""Tests for `patchforge.stitch` — contract from docs/THEORY.md §9.9."""
from __future__ import annotations

import pytest
import torch

from patchforge import extract, reconstruct, stitch


def _ramp(c: int, h: int, w: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.arange(c * h * w, dtype=dtype).reshape(c, h, w)


# ---------------------------------------------------- uniform == reconstruct --

class TestUniformEqualsReconstruct:
    """`weight="uniform"` is mathematically equivalent to `reconstruct`."""

    def test_exact_tiling_bit_exact(self) -> None:
        img = _ramp(1, 16, 16)
        patches = extract(img, patch_size=4, stride=4)
        out_stitch = stitch(patches, image_shape=img.shape, stride=4)
        out_recon = reconstruct(patches, image_shape=img.shape, stride=4)
        assert torch.equal(out_stitch, out_recon)

    def test_exact_tiling_recovers_image(self) -> None:
        img = _ramp(2, 12, 12)
        patches = extract(img, patch_size=4, stride=4)
        out = stitch(patches, image_shape=img.shape, stride=4, weight="uniform")
        assert torch.equal(out, img)

    def test_overlap_matches_reconstruct(self) -> None:
        img = _ramp(1, 16, 16, dtype=torch.float64)
        patches = extract(img, patch_size=4, stride=2)
        out_stitch = stitch(patches, image_shape=img.shape, stride=2)
        out_recon = reconstruct(patches, image_shape=img.shape, stride=2)
        assert torch.allclose(out_stitch, out_recon, rtol=1e-12, atol=1e-12)

    def test_overlap_recovers_image(self) -> None:
        img = _ramp(1, 16, 16, dtype=torch.float64)
        patches = extract(img, patch_size=4, stride=2)
        out = stitch(patches, image_shape=img.shape, stride=2)
        assert torch.allclose(out, img, rtol=1e-12, atol=1e-12)

    def test_rectangular_geometry(self) -> None:
        img = _ramp(1, 20, 30, dtype=torch.float64)
        patches = extract(img, patch_size=(4, 6), stride=(4, 6))
        out = stitch(patches, image_shape=img.shape, stride=(4, 6))
        assert torch.allclose(out, img, rtol=1e-12, atol=1e-12)


# -------------------------------------------------------------- hann window --

class TestHann:
    """Hann window emphasizes patch centers; corner-pixel artifact is documented."""

    def test_unmodified_overlap_recovers_interior(self) -> None:
        """With overlap, interior pixels still recover the original under Hann."""
        img = _ramp(1, 16, 16, dtype=torch.float64)
        patches = extract(img, patch_size=4, stride=2)
        out = stitch(patches, image_shape=img.shape, stride=2, weight="hann")
        # Interior strip (away from rows/cols 0 and H-1) is bit-correct-ish.
        assert torch.allclose(out[:, 2:-2, 2:-2], img[:, 2:-2, 2:-2],
                              rtol=1e-9, atol=1e-9)

    def test_corner_artifact_with_exact_tiling(self) -> None:
        """At stride==patch_size, image corners are covered by patches whose
        Hann edge-weight is 0 — the documented artifact makes them zero."""
        img = torch.full((1, 8, 8), 0.5)
        patches = extract(img, patch_size=4, stride=4)
        out = stitch(patches, image_shape=img.shape, stride=4, weight="hann")
        # Corner pixels of the image: covered by exactly 1 patch at its corner.
        assert out[0, 0, 0].item() == pytest.approx(0.0, abs=1e-6)
        assert out[0, 0, 7].item() == pytest.approx(0.0, abs=1e-6)
        assert out[0, 7, 0].item() == pytest.approx(0.0, abs=1e-6)
        assert out[0, 7, 7].item() == pytest.approx(0.0, abs=1e-6)

    def test_center_pixel_weighted_more_than_edge(self) -> None:
        """Hann at stride==patch_size: pixel offset (1, 1) inside the patch
        (Hann ≈ 0.75 each axis) should equal its source patch value exactly
        (single contributor in single-patch coverage). Pixel offset (0, 0)
        (Hann = 0 each axis) becomes the documented zero artifact."""
        # Single patch tile: image is the patch itself.
        img = torch.arange(16, dtype=torch.float64).reshape(1, 4, 4)
        patches = extract(img, patch_size=4, stride=4)
        out = stitch(patches, image_shape=img.shape, stride=4, weight="hann")
        # Interior pixel: single patch, single non-zero weight, ratio = patch value.
        assert out[0, 1, 1].item() == pytest.approx(img[0, 1, 1].item(), abs=1e-9)
        assert out[0, 2, 2].item() == pytest.approx(img[0, 2, 2].item(), abs=1e-9)
        # Edge / corner pixels: Hann weight at edge is 0 -> zero artifact.
        assert out[0, 0, 0].item() == pytest.approx(0.0, abs=1e-9)
        assert out[0, 3, 3].item() == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------- gaussian window --

class TestGaussian:
    def test_unmodified_recovers_full_image(self) -> None:
        """Gaussian weight is > 0 everywhere; round-trip on unmodified
        overlapping patches recovers the full image (no corner artifact)."""
        img = _ramp(1, 16, 16, dtype=torch.float64)
        patches = extract(img, patch_size=4, stride=2)
        out = stitch(patches, image_shape=img.shape, stride=2, weight="gaussian")
        assert torch.allclose(out, img, rtol=1e-9, atol=1e-9)

    def test_no_corner_artifact_at_exact_tiling(self) -> None:
        """Unlike Hann, Gaussian does not zero the corners — weight is
        ``exp(-((edge - center)^2) / (2 * sigma^2)) > 0``."""
        img = torch.full((1, 8, 8), 0.5)
        patches = extract(img, patch_size=4, stride=4)
        out = stitch(patches, image_shape=img.shape, stride=4, weight="gaussian")
        assert out[0, 0, 0].item() == pytest.approx(0.5, abs=1e-9)
        assert out[0, 7, 7].item() == pytest.approx(0.5, abs=1e-9)


# --------------------------------------------------------- type preservation --

class TestPreservation:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("weight", ["uniform", "hann", "gaussian"])
    def test_dtype_preserved(self, dtype: torch.dtype, weight: str) -> None:
        img = _ramp(1, 8, 8, dtype=dtype)
        patches = extract(img, patch_size=4, stride=4)
        out = stitch(patches, image_shape=img.shape, stride=4, weight=weight)  # type: ignore[arg-type]
        assert out.dtype == dtype

    def test_device_preserved_cpu(self) -> None:
        img = _ramp(1, 8, 8)
        patches = extract(img, patch_size=4, stride=4)
        out = stitch(patches, image_shape=img.shape, stride=4)
        assert out.device == patches.device

    @pytest.mark.gpu
    def test_cuda_roundtrip(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        img = _ramp(1, 8, 8).cuda()
        patches = extract(img, patch_size=4, stride=4)
        out = stitch(patches, image_shape=img.shape, stride=4)
        assert out.device.type == "cuda"
        assert torch.equal(out, img)

    def test_accepts_torch_size(self) -> None:
        img = _ramp(1, 8, 8)
        patches = extract(img, patch_size=4, stride=4)
        out = stitch(patches, image_shape=img.shape, stride=4)
        assert isinstance(img.shape, torch.Size)
        assert torch.equal(out, img)


# ----------------------------------------------------------------- rejects --

class TestRejects:
    def test_patches_not_tensor(self) -> None:
        with pytest.raises(TypeError, match=r"must be torch\.Tensor"):
            stitch([1, 2, 3], image_shape=(1, 4, 4), stride=2)  # type: ignore[arg-type]

    @pytest.mark.parametrize("ndim", [1, 2, 3, 5])
    def test_patches_wrong_ndim(self, ndim: int) -> None:
        patches = torch.zeros([4] * ndim)
        with pytest.raises(ValueError, match="ndim==4"):
            stitch(patches, image_shape=(1, 16, 16), stride=4)

    def test_int_dtype_rejected(self) -> None:
        patches = torch.zeros(16, 1, 4, 4, dtype=torch.int32)
        with pytest.raises(ValueError, match="floating-point"):
            stitch(patches, image_shape=(1, 16, 16), stride=4)

    def test_uint8_dtype_rejected(self) -> None:
        patches = torch.zeros(16, 1, 4, 4, dtype=torch.uint8)
        with pytest.raises(ValueError, match="floating-point"):
            stitch(patches, image_shape=(1, 16, 16), stride=4)

    def test_unknown_weight_rejected(self) -> None:
        patches = torch.zeros(16, 1, 4, 4)
        with pytest.raises(ValueError, match="weight must be one of"):
            stitch(patches, image_shape=(1, 16, 16), stride=4, weight="lanczos")  # type: ignore[arg-type]

    def test_dilation_rejected(self) -> None:
        patches = torch.zeros(16, 1, 4, 4)
        with pytest.raises(ValueError, match="dilation==1"):
            stitch(patches, image_shape=(1, 16, 16), stride=4, dilation=2)

    def test_stride_greater_than_patch_rejected(self) -> None:
        patches = torch.zeros(9, 1, 4, 4)
        with pytest.raises(ValueError, match="stride > patch_size"):
            stitch(patches, image_shape=(1, 20, 20), stride=6)

    def test_stride_greater_than_patch_one_axis(self) -> None:
        patches = torch.zeros(9, 1, 4, 4)
        with pytest.raises(ValueError, match="stride > patch_size"):
            stitch(patches, image_shape=(1, 16, 20), stride=(4, 6))

    @pytest.mark.parametrize("bad", [(1, 4), (1, 4, 4, 4), 4])
    def test_image_shape_wrong_arity(self, bad: object) -> None:
        patches = torch.zeros(4, 1, 4, 4)
        with pytest.raises(ValueError, match="3-tuple"):
            stitch(patches, image_shape=bad, stride=4)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [(1, 0, 4), (1, -4, 4), (0, 8, 8)])
    def test_image_shape_nonpositive(self, bad: tuple[int, int, int]) -> None:
        patches = torch.zeros(4, 1, 4, 4)
        with pytest.raises(ValueError, match="positive int"):
            stitch(patches, image_shape=bad, stride=4)

    def test_channel_mismatch(self) -> None:
        patches = torch.zeros(4, 3, 4, 4)
        with pytest.raises(ValueError, match="channel count"):
            stitch(patches, image_shape=(1, 8, 8), stride=4)

    def test_n_patches_inconsistent_too_few(self) -> None:
        patches = torch.zeros(4, 1, 4, 4)
        with pytest.raises(ValueError, match="inconsistent with"):
            stitch(patches, image_shape=(1, 16, 16), stride=4)

    def test_n_patches_inconsistent_too_many(self) -> None:
        patches = torch.zeros(100, 1, 4, 4)
        with pytest.raises(ValueError, match="inconsistent with"):
            stitch(patches, image_shape=(1, 8, 8), stride=4)

    def test_image_too_small_for_patch(self) -> None:
        patches = torch.zeros(1, 1, 4, 4)
        with pytest.raises(ValueError, match="too small"):
            stitch(patches, image_shape=(1, 2, 2), stride=4)

    def test_stride_nonpositive(self) -> None:
        patches = torch.zeros(4, 1, 4, 4)
        with pytest.raises(ValueError, match="stride must be positive"):
            stitch(patches, image_shape=(1, 8, 8), stride=0)
