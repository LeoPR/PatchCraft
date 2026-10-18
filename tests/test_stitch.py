"""Tests for `patchcraft.stitch`, contract from docs/THEORY.md §9.9."""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)

from patchcraft import extract, reconstruct, stitch
from tests._rng import bit_equal, rand_image


def _ramp(c: int, h: int, w: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.arange(c * h * w, dtype=dtype).reshape(c, h, w)


# ---------------------------------------------------- uniform == reconstruct --

class TestUniformEqualsReconstruct:
    """`weight="uniform"` is mathematically equivalent to `reconstruct`."""

    def test_exact_tiling_bit_exact(self) -> None:
        img = rand_image(1, 16, 16, torch.float32, seed=501)
        patches = extract(img, patch_size=4, stride=4)
        out_stitch = stitch(patches, image_shape=img.shape, stride=4)
        out_recon = reconstruct(patches, image_shape=img.shape, stride=4)
        assert bit_equal(out_stitch, out_recon)

    def test_exact_tiling_returns_what_the_fold_returns(self) -> None:
        """The non-overlapping uniform path skips the fold, and must still
        return the fold's bits: 0.0 + x, which turns -0.0 into +0.0 and
        quiets a NaN. A bare rearrangement would keep -0.0, and torch.equal
        cannot tell the two apart, so this compares integer views against
        F.fold itself."""
        vals = torch.tensor([-0.0, 0.0, float("nan"), float("inf"), -1.5, 2.0**-140])
        patches = vals[torch.arange(4 * 1 * 3 * 3) % vals.numel()].reshape(4, 1, 3, 3)
        out = stitch(patches, image_shape=(1, 6, 6), stride=3)
        cols = patches.permute(1, 2, 3, 0).reshape(9, 4).unsqueeze(0)
        folded = F.fold(cols, output_size=(6, 6), kernel_size=3, stride=3)[0]
        assert bit_equal(out, folded / 1.0)
        assert not torch.signbit(out[out == 0]).any()

    def test_exact_tiling_recovers_image(self) -> None:
        img = rand_image(2, 12, 12, torch.float32, seed=502)
        patches = extract(img, patch_size=4, stride=4)
        out = stitch(patches, image_shape=img.shape, stride=4, weight="uniform")
        assert bit_equal(out, img)

    @pytest.mark.parametrize("accel", ["1", "0"], ids=["accel", "pure"])
    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
    )
    @pytest.mark.parametrize(
        ("h", "w", "p", "s"), [(16, 16, 4, 2), (12, 10, (4, 3), (2, 1)), (9, 7, 3, (1, 2))]
    )
    def test_overlap_is_reconstruct_bit_for_bit(
        self,
        monkeypatch: pytest.MonkeyPatch,
        accel: str,
        dtype: torch.dtype,
        h: int,
        w: int,
        p: int | tuple[int, int],
        s: int | tuple[int, int],
    ) -> None:
        """On an overlapping grid the uniform window delegates to reconstruct,
        so the two agree bit for bit, counts of 3 included, on patches that
        disagree and on signed zeros, infinities and NaN payloads."""
        monkeypatch.setenv("PATCHCRAFT_ACCEL", accel)
        n, _, ph, pw = extract(torch.zeros(1, h, w), p, s).shape
        g = torch.Generator().manual_seed(503)
        patches = (torch.rand(n, 2, ph, pw, generator=g) * 2 - 1).to(dtype)
        flat = patches.view(-1)
        flat[:5] = torch.tensor([0.0, -0.0, float("inf"), float("-inf"), float("nan")])
        out_stitch = stitch(patches, image_shape=(2, h, w), stride=s)
        out_recon = reconstruct(patches, image_shape=(2, h, w), stride=s)
        assert bit_equal(out_stitch, out_recon)

    def test_overlap_recovers_image(self) -> None:
        """counts are 1, 2, 4 (powers of two): uniform stitch is bit-exact
        here, since it is reconstruct's arithmetic on an overlapping grid."""
        img = rand_image(1, 16, 16, torch.float64, seed=504)
        patches = extract(img, patch_size=4, stride=2)
        out = stitch(patches, image_shape=img.shape, stride=2)
        assert bit_equal(out, img)

    def test_rectangular_geometry(self) -> None:
        img = rand_image(1, 20, 30, torch.float64, seed=505)
        patches = extract(img, patch_size=(4, 6), stride=(4, 6))
        out = stitch(patches, image_shape=img.shape, stride=(4, 6))
        assert bit_equal(out, img)


# -------------------------------------------------------------- hann window --

class TestHann:
    """Hann window emphasizes patch centers and is strictly positive everywhere,
    so unmodified patches round-trip under any tiling, within the count bound
    in floating point (§2.5)."""

    def test_unmodified_overlap_recovers_image(self) -> None:
        """With overlap, unmodified patches recover the full image under Hann."""
        img = rand_image(1, 16, 16, torch.float64, seed=511)
        patches = extract(img, patch_size=4, stride=2)
        out = stitch(patches, image_shape=img.shape, stride=2, weight="hann")
        assert torch.allclose(out, img, rtol=1e-9, atol=1e-9)

    def test_exact_tiling_recovers_image(self) -> None:
        """Regression (0.2.0 audit): the old symmetric Hann was exactly 0 at both
        edges, so at stride == patch_size a 12x12 image with 4x4 patches came
        back with 108 of 144 pixels zeroed. The window is now the interior of a
        longer Hann window, strictly positive on every sample."""
        img = rand_image(1, 12, 12, torch.float64, seed=512) + 1.0
        patches = extract(img, patch_size=4, stride=4)
        out = stitch(patches, image_shape=img.shape, stride=4, weight="hann")
        assert torch.allclose(out, img, rtol=1e-9, atol=1e-9)

    def test_no_zeroed_pixels_with_overlap(self) -> None:
        """Regression (0.2.0 audit): 13x13 with patch 4 stride 3 lost 105 of 169
        pixels under the old window, including black bands in the interior."""
        img = rand_image(1, 13, 13, torch.float64, seed=513) + 1.0
        patches = extract(img, patch_size=4, stride=3)
        out = stitch(patches, image_shape=img.shape, stride=3, weight="hann")
        assert torch.allclose(out, img, rtol=1e-9, atol=1e-9)

    def test_patch_size_2_not_degenerate(self) -> None:
        """Regression (0.2.0 audit): patch_size=2 made the old window
        identically [0, 0] and stitch returned an all-zero image with no error."""
        img = rand_image(1, 8, 8, torch.float64, seed=514)
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
        img = rand_image(1, 16, 16, torch.float64, seed=521)
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
        img = rand_image(1, 8, 8, torch.float32, seed=506).cuda()
        patches = extract(img, patch_size=4, stride=4)
        out = stitch(patches, image_shape=img.shape, stride=4)
        assert out.device.type == "cuda"
        assert bit_equal(out, img)

    def test_accepts_torch_size(self) -> None:
        img = rand_image(1, 8, 8, torch.float32, seed=507)
        patches = extract(img, patch_size=4, stride=4)
        out = stitch(patches, image_shape=img.shape, stride=4)
        assert isinstance(img.shape, torch.Size)
        assert bit_equal(out, img)


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
        """Grid that ends exactly on the image edge must not raise."""
        img = rand_image(1, 10, 10, torch.float32, seed=508)
        patches = extract(img, patch_size=4, stride=2)  # (4-1)*2+4 == 10
        out = stitch(patches, image_shape=img.shape, stride=2)
        assert bit_equal(out, img)


# --------------------------- characterization: pre-0.3.0 reference (Task 5) --


def _stitch_reference(patches, image_shape, stride, weight):
    """The pre-0.3.0 stitch implementation, as ground truth."""
    # Imports _window_kernel from the CURRENT module (not frozen) on purpose: the
    # 1-D window helpers are numerically unchanged in 0.3.0, but a future change
    # to _window_kernel would move this reference.
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


# ---------------------------------------------- Denominator floor (defect) --
@pytest.mark.parametrize("patch", [128, 200, 256])
def test_hann_corner_band_survives_large_patches(patch):
    """The denominator floor must not swallow a legitimate window weight.

    The 2-D hann corner weight is ``(pi/(n+1))**4``, which drops below 1e-6 at
    patch 99. An absolute ``clamp(min=1e-6)`` therefore divided the corner band
    of every larger hann stitch by 1e-6 instead of by the real weight. Measured
    before the fix at 640x640, patch 256, stride 128: max error 0.94 on data in
    [0, 1] and 960 wrong pixels, identical in float64 because the constant, not
    the precision, was the cause.
    """
    from patchcraft import extract
    from patchcraft.stitch import _window_kernel

    corner = _window_kernel("hann", patch, patch, torch.float64, torch.device("cpu")).min()
    assert corner < 1e-6, "geometry no longer exercises the defect"

    from tests._rng import coverage_counts, within_pixel_bound

    img = rand_image(1, patch * 2, patch * 2, torch.float64, seed=patch)
    patches = extract(img, patch, stride=patch // 2)
    got = stitch(patches, img.shape, stride=patch // 2, weight="hann")
    s = patch // 2
    assert within_pixel_bound(got, img, coverage_counts(patch * 2, patch * 2, patch, patch, s, s))


def test_denominator_floor_is_dtype_relative_not_absolute():
    """The floor is the dtype's smallest normal, so it never fires on a real
    weight: even at patch 1024 the 2-D hann corner is 8.8e-11, which is 27
    orders of magnitude above float32's tiny."""
    from patchcraft.stitch import _window_kernel

    for dtype in (torch.float32, torch.float64):
        corner = _window_kernel("hann", 1024, 1024, dtype, torch.device("cpu")).min().item()
        assert corner > torch.finfo(dtype).tiny * 1e3


# ------------------------------------- Window denominator, a direct sum (defect) --
#
# From 0.3.0 to 0.6.0 each 1-D window sum was the difference of two cumulative
# sums of its stride residue, which cancelled on the last (patch - stride) rows
# and columns, where every covering patch sees the pixel near its far edge.
# Measured on 0.6.0, float32 hann: all-ones patches came back 2.4e-03 above 1
# at 288x288, patch 256, stride 32 (2.0e+04 eps), a single 256 patch at
# stride 128 missed by 6,696 eps, and the sum reached exactly zero first at
# patch 616 and stride 1, so the floor divided covered pixels by the smallest
# normal and a 1x616 strip came back near 2e33. Of the two tests above, the
# second checks the window's corner weight alone, and the first checked this
# denominator only to 1e-9 in float64, which the cancellation stayed under:
# 421, 1,400 and 245 eps at its three patches. It now holds the count bound.

# (h, w, patch, stride), each covering its image exactly.
_DENOMINATOR_GEOMETRIES = [
    (288, 288, 256, 32),            # two patches per axis, a band of 224
    (256, 256, 256, 128),           # one patch at a stride below its size
    (264, 264, 256, 8),             # 32 terms per residue
    (512, 512, 128, 64),            # half overlap, as docs/PERFORMANCE.md measures
    (288, 96, (256, 64), (16, 8)),  # rectangular in every parameter
    (1, 616, (1, 616), (1, 1)),     # the sum cancelled to exactly zero here
    (96, 96, 64, 1),                # coverage up to 33 per axis, 1,089 in all
]
_DENOMINATOR_DTYPES = [torch.float32, torch.float64, torch.float16, torch.bfloat16]


def _pair(v: int | tuple[int, int]) -> tuple[int, int]:
    return (v, v) if isinstance(v, int) else v


@pytest.fixture(params=["1", "0"], ids=["accel", "pure"])
def both_paths(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("PATCHCRAFT_ACCEL", request.param)
    return str(request.param)


@pytest.mark.parametrize("dtype", _DENOMINATOR_DTYPES, ids=str)
@pytest.mark.parametrize("weight", ["hann", "gaussian"])
@pytest.mark.parametrize(("h", "w", "patch", "stride"), _DENOMINATOR_GEOMETRIES)
def test_round_trip_within_the_count_bound(
    both_paths: str,
    dtype: torch.dtype,
    weight: str,
    h: int,
    w: int,
    patch: int | tuple[int, int],
    stride: int | tuple[int, int],
) -> None:
    """stitch(extract(x)) is within (K + 1) * eps * |x| of x, K the coverage.

    That is the bound reconstruct carries outside the count-map rule (THEORY
    §9.2), and THEORY §2.5 counts the roundings behind it. Half precision
    accumulates in float32, far below its own resolution at these coverages,
    so the cast back lands on the input bit for bit.
    """
    from tests._rng import coverage_counts, within_pixel_bound

    ph, pw = _pair(patch)
    sh, sw = _pair(stride)
    img = rand_image(2, h, w, dtype, seed=h * 7 + w + ph + sh)
    out = stitch(extract(img, patch, stride), img.shape, stride, weight=weight)  # type: ignore[arg-type]
    assert out.dtype == dtype
    if dtype in (torch.float16, torch.bfloat16):
        assert bit_equal(out, img)
    else:
        assert within_pixel_bound(out, img, coverage_counts(h, w, ph, pw, sh, sw))


@pytest.mark.parametrize("dtype", _DENOMINATOR_DTYPES, ids=str)
@pytest.mark.parametrize("weight", ["hann", "gaussian"])
@pytest.mark.parametrize(("h", "w", "patch", "stride"), _DENOMINATOR_GEOMETRIES)
def test_stitch_of_ones_is_one(
    both_paths: str,
    dtype: torch.dtype,
    weight: str,
    h: int,
    w: int,
    patch: int | tuple[int, int],
    stride: int | tuple[int, int],
) -> None:
    """All-ones patches come back within (K + 1) * eps of 1, and exactly 1 in
    half precision. It does not depend on drawn data, whose seeded half
    precision bits differ between torch 2.6 and 2.14."""
    from tests._rng import coverage_counts, within_pixel_bound

    ph, pw = _pair(patch)
    sh, sw = _pair(stride)
    n = extract(torch.zeros(1, h, w), patch, stride).shape[0]
    out = stitch(torch.ones(n, 2, ph, pw, dtype=dtype), (2, h, w), stride, weight=weight)  # type: ignore[arg-type]
    target = torch.ones(2, h, w, dtype=dtype)
    if dtype in (torch.float16, torch.bfloat16):
        assert bit_equal(out, target)
    else:
        assert within_pixel_bound(out, target, coverage_counts(h, w, ph, pw, sh, sw))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
@pytest.mark.parametrize("kind", ["hann", "gaussian"])
@pytest.mark.parametrize(
    ("n", "step", "num"),
    [(256, 32, 1), (256, 32, 9), (128, 64, 7), (128, 16, 2), (32, 16, 31), (512, 8, 2),
     (616, 1, 1), (64, 1, 70), (7, 3, 4), (5, 5, 3), (256, 8, 32), (256, 4, 64),
     (64, 1, 3), (100, 3, 7), (256, 8, 17), (256, 4, 33), (616, 1, 33)],
)
def test_window_fold_is_a_direct_sum(
    dtype: torch.dtype, kind: str, n: int, step: int, num: int
) -> None:
    """Each float32 window sum is its exact sum rounded once, and each float64
    one is within h * eps of the exact sum of its h terms.

    The float32 claim rests on float64 holding the exact sum, which the test
    checks before it relies on it; the float64 sum is rounded at each of its
    additions. The cumsum difference broke both, by 1.0e+04 eps for hann at
    n=256, step=32, num=1 in float32, and returned 0 at n=616, step=1. Both
    branches are covered: at least as many patches as window blocks, whose
    prefix and suffix sums reach 64 terms here, and fewer, whose runs reach
    33. The runs are the cases that would catch a float32 accumulation, since
    torch's CPU cumsum already accumulates float32 in double. The tenth case
    does not overlap and passed before too.
    """
    from fractions import Fraction

    from patchcraft.stitch import _fold_window_1d, _window_1d

    length = (num - 1) * step + n
    w = _window_1d(kind, n, dtype, torch.device("cpu"))  # type: ignore[arg-type]
    got = _fold_window_1d(w, length, num, step).tolist()
    vals = w.tolist()
    eps = torch.finfo(dtype).eps
    for y in range(length):
        terms = [vals[y - i * step] for i in range(num) if 0 <= y - i * step < n]
        ref = math.fsum(terms)
        if dtype == torch.float32:
            assert Fraction(ref) == sum(map(Fraction, terms)), "float64 lost the exact sum"
            want = torch.tensor(ref, dtype=torch.float64).to(torch.float32).item()
            assert got[y] == want, (y, got[y], want)
        else:
            assert abs(got[y] - ref) <= len(terms) * eps * ref, (y, got[y], ref)


@pytest.mark.parametrize("dtype", _DENOMINATOR_DTYPES, ids=str)
@pytest.mark.parametrize("weight", ["hann", "gaussian"])
@pytest.mark.parametrize(
    ("shape", "patch", "stride"),
    [((3, 16, 16), 4, 2), ((1, 10, 10), 8, 2)],  # more patches than blocks, and fewer
)
def test_meta_patches_come_back_as_a_meta_image(
    both_paths: str,
    dtype: torch.dtype,
    weight: str,
    shape: tuple[int, int, int],
    patch: int,
    stride: int,
) -> None:
    """A meta tensor has a shape and a dtype and no data, which is how a caller
    sizes a pipeline or counts its FLOPs before computing anything. The
    window sums stay on the window's device, so the image stays on meta."""
    n = extract(torch.zeros(1, *shape[1:]), patch, stride).shape[0]
    patches = torch.empty(n, shape[0], patch, patch, dtype=dtype, device="meta")
    out = stitch(patches, shape, stride, weight=weight)  # type: ignore[arg-type]
    assert out.device.type == "meta" and out.shape == shape and out.dtype == dtype
