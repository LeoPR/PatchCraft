"""Tests for `patchcraft.stitch`, contract from docs/THEORY.md §9.9."""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)

from patchcraft import extract, reconstruct, stitch


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
    """Hann window emphasizes patch centers and is strictly positive everywhere,
    so unmodified patches round-trip exactly under any tiling (§2.5)."""

    def test_unmodified_overlap_recovers_image(self) -> None:
        """With overlap, unmodified patches recover the full image under Hann."""
        img = _ramp(1, 16, 16, dtype=torch.float64)
        patches = extract(img, patch_size=4, stride=2)
        out = stitch(patches, image_shape=img.shape, stride=2, weight="hann")
        assert torch.allclose(out, img, rtol=1e-9, atol=1e-9)

    def test_exact_tiling_recovers_image(self) -> None:
        """Regression (0.2.0 audit): the old symmetric Hann was exactly 0 at both
        edges, so at stride == patch_size a 12x12 image with 4x4 patches came
        back with 108 of 144 pixels zeroed. The window is now the interior of a
        longer Hann window, strictly positive on every sample."""
        img = _ramp(1, 12, 12, dtype=torch.float64) + 1.0
        patches = extract(img, patch_size=4, stride=4)
        out = stitch(patches, image_shape=img.shape, stride=4, weight="hann")
        assert torch.allclose(out, img, rtol=1e-9, atol=1e-9)

    def test_no_zeroed_pixels_with_overlap(self) -> None:
        """Regression (0.2.0 audit): 13x13 with patch 4 stride 3 lost 105 of 169
        pixels under the old window, including black bands in the interior."""
        img = _ramp(1, 13, 13, dtype=torch.float64) + 1.0
        patches = extract(img, patch_size=4, stride=3)
        out = stitch(patches, image_shape=img.shape, stride=3, weight="hann")
        assert torch.allclose(out, img, rtol=1e-9, atol=1e-9)

    def test_patch_size_2_not_degenerate(self) -> None:
        """Regression (0.2.0 audit): patch_size=2 made the old window
        identically [0, 0] and stitch returned an all-zero image with no error."""
        img = _ramp(1, 8, 8, dtype=torch.float64)
        patches = extract(img, patch_size=2, stride=2)
        out = stitch(patches, image_shape=img.shape, stride=2, weight="hann")
        assert torch.allclose(out, img, rtol=1e-9, atol=1e-9)

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 7, 16])
    def test_kernel_strictly_positive(self, n: int) -> None:
        """The 1-D Hann window is strictly positive on every sample."""
        from patchcraft.stitch import _hann_1d
        w = _hann_1d(n, torch.float64, torch.device("cpu"))
        assert w.shape == (n,)
        assert (w > 0).all()

    def test_center_pixel_weighted_more_than_edge(self) -> None:
        """Hann at stride==patch_size: pixel offset (1, 1) inside the patch
        (higher Hann weight) and pixel offset (0, 0) (lower weight) both equal
        their source patch values, because coverage has a single contributor."""
        # Single patch tile: image is the patch itself.
        img = torch.arange(16, dtype=torch.float64).reshape(1, 4, 4)
        patches = extract(img, patch_size=4, stride=4)
        out = stitch(patches, image_shape=img.shape, stride=4, weight="hann")
        assert out[0, 1, 1].item() == pytest.approx(img[0, 1, 1].item(), abs=1e-9)
        assert out[0, 2, 2].item() == pytest.approx(img[0, 2, 2].item(), abs=1e-9)
        assert out[0, 0, 0].item() == pytest.approx(img[0, 0, 0].item(), abs=1e-9)
        assert out[0, 3, 3].item() == pytest.approx(img[0, 3, 3].item(), abs=1e-9)


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
        """Unlike Hann, Gaussian does not zero the corners, because weight is
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


class TestHalfPrecisionAccumulation:
    """Same fp16/bf16 overflow fix as `reconstruct`; see
    tests/test_reconstruct.py::TestHalfPrecisionAccumulation."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("weight", ["uniform", "hann", "gaussian"])
    def test_large_constant_no_overflow(self, dtype: torch.dtype, weight: str) -> None:
        img = torch.full((1, 16, 16), 10000.0, dtype=dtype)
        patches = extract(img, patch_size=3, stride=1)
        out = stitch(patches, image_shape=img.shape, stride=1, weight=weight)  # type: ignore[arg-type]
        assert out.dtype == dtype
        assert torch.isfinite(out).all()
        assert torch.allclose(out, img)


class TestCoverageGuard:
    """Regression (0.2.0 audit): same truncated-grid defect as `reconstruct`;
    see tests/test_reconstruct.py::TestCoverageGuard."""

    def test_truncated_grid_10x10_patch_4(self) -> None:
        patches = torch.zeros(4, 1, 4, 4)
        with pytest.raises(ValueError, match="coverage"):
            stitch(patches, image_shape=(1, 10, 10), stride=4)

    def test_truncated_grid_13x13_patch_5(self) -> None:
        patches = torch.zeros(4, 1, 5, 5)
        with pytest.raises(ValueError, match="coverage"):
            stitch(patches, image_shape=(1, 13, 13), stride=5)

    def test_exact_coverage_boundary_still_accepted(self) -> None:
        img = torch.arange(100, dtype=torch.float32).reshape(1, 10, 10)
        patches = extract(img, patch_size=4, stride=2)  # (4-1)*2+4 == 10
        out = stitch(patches, image_shape=img.shape, stride=2)
        assert torch.allclose(out, img)


# --------------------------- characterization: pre-0.3.0 reference (Task 5) --


def _stitch_reference(patches, image_shape, stride, weight):
    """The pre-0.3.0 stitch implementation, as ground truth."""
    from patchcraft.stitch import _window_kernel
    n_patches, c, ph, pw = patches.shape
    h, w = image_shape[1], image_shape[2]
    sh, sw = (stride, stride) if isinstance(stride, int) else stride
    accum_dtype = (
        torch.float32
        if patches.dtype in (torch.float16, torch.bfloat16)
        else patches.dtype
    )
    kernel = _window_kernel(weight, ph, pw, accum_dtype, patches.device)
    weighted = patches.to(accum_dtype) * kernel
    num_flat = weighted.permute(1, 2, 3, 0).reshape(c * ph * pw, n_patches).unsqueeze(0)
    folded_num = F.fold(num_flat, (h, w), (ph, pw), stride=(sh, sw))
    kernel_flat = kernel.flatten().unsqueeze(1).repeat(1, n_patches).unsqueeze(0)
    folded_den = F.fold(kernel_flat, (h, w), (ph, pw), stride=(sh, sw))
    return (folded_num / folded_den.clamp(min=1e-6))[0].to(patches.dtype)


@pytest.mark.parametrize("c,h,w", [(1, 8, 8), (3, 16, 16), (3, 32, 24)])
@pytest.mark.parametrize(
    "ph,pw,sh,sw",
    [(2, 2, 2, 2), (4, 4, 2, 2), (3, 3, 1, 1), (4, 6, 2, 3)],
)
@pytest.mark.parametrize("weight", ["uniform", "hann", "gaussian"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_stitch_matches_reference(c, h, w, ph, pw, sh, sw, weight, dtype):
    num_h = (h - ph) // sh + 1
    num_w = (w - pw) // sw + 1
    if (num_h - 1) * sh + ph != h or (num_w - 1) * sw + pw != w:
        pytest.skip("geometry does not cover exactly")
    img = torch.randn(c, h, w, dtype=dtype)
    from patchcraft import extract
    patches = extract(img, (ph, pw), (sh, sw))
    got = stitch(patches, (c, h, w), (sh, sw), weight=weight)
    ref = _stitch_reference(patches, (c, h, w), (sh, sw), weight)
    assert got.shape == ref.shape and got.dtype == ref.dtype
    if weight == "uniform":
        assert torch.equal(got, ref)  # integer counts: bit-exact
    else:
        # denominator summation order differs from F.fold: ULP-level
        torch.testing.assert_close(got, ref)


@pytest.mark.parametrize("n", [2, 3, 8, 32, 128])
def test_gaussian_kernel_stays_above_exp_minus_4(n):
    """2-D gaussian kernel minimum is strictly above exp(-4): the 1-D profile
    exceeds exp(-2) at both edges (exponent -2*((n-1)/n)^2 for n >= 4, closer
    to 0 for n in {2, 3}), so the outer-product corner exceeds exp(-4)."""
    from patchcraft.stitch import _window_kernel
    k = _window_kernel("gaussian", n, n, torch.float64, torch.device("cpu"))
    assert k.min().item() > math.exp(-4)
