"""Tests for `patchcraft.reconstruct`, contract from docs/THEORY.md §9.2."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)

from patchcraft import extract, reconstruct
from tests._rng import bit_equal, coverage_counts, rand_image, within_pixel_bound


def _ramp(c: int, h: int, w: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Image with every pixel unique, which distinguishes pixels positionally."""
    return torch.arange(c * h * w, dtype=dtype).reshape(c, h, w)


# ----------------------------------------------------------------- Roundtrip --

class TestRoundtripExact:
    """`stride == patch_size`: each pixel covered exactly once, bit-exact."""

    def test_basic(self) -> None:
        img = rand_image(3, 32, 32, torch.float32, seed=101)
        patches = extract(img, patch_size=8, stride=8)
        out = reconstruct(patches, image_shape=img.shape, stride=8)
        assert bit_equal(out, img)

    def test_rectangular_geometry(self) -> None:
        img = rand_image(1, 20, 30, torch.float64, seed=102)
        patches = extract(img, patch_size=(4, 6), stride=(4, 6))
        out = reconstruct(patches, image_shape=img.shape, stride=(4, 6))
        assert bit_equal(out, img)

    def test_single_patch_equals_image(self) -> None:
        img = rand_image(2, 8, 8, torch.float32, seed=103)
        patches = extract(img, patch_size=8, stride=8)
        out = reconstruct(patches, image_shape=img.shape, stride=8)
        assert bit_equal(out, img)

    def test_multichannel(self) -> None:
        img = rand_image(7, 16, 16, torch.float32, seed=104)
        patches = extract(img, patch_size=4, stride=4)
        out = reconstruct(patches, image_shape=img.shape, stride=4)
        assert bit_equal(out, img)

    def test_patch_size_1(self) -> None:
        img = rand_image(1, 4, 4, torch.float32, seed=105)
        patches = extract(img, patch_size=1, stride=1)
        out = reconstruct(patches, image_shape=img.shape, stride=1)
        assert bit_equal(out, img)


class TestRoundtripOverlap:
    """`stride < patch_size`: exact iff every count-map value is a power of
    two (ADR 0003); otherwise bounded per pixel by (k+1)*eps*|v|."""

    def test_half_overlap_basic(self) -> None:
        """counts are 1, 2, 4 -> inside the predicate -> bit-exact."""
        img = rand_image(1, 16, 16, torch.float64, seed=201)
        patches = extract(img, patch_size=4, stride=2)
        out = reconstruct(patches, image_shape=img.shape, stride=2)
        assert bit_equal(out, img)

    def test_max_overlap_stride_1(self) -> None:
        """p=3 s=1 puts a 3 in the count map -> outside the predicate;
        the per-pixel bound (k+1)*eps*|v| is the contract (Amendment A)."""
        img = rand_image(2, 8, 8, torch.float64, seed=202)
        patches = extract(img, patch_size=3, stride=1)
        out = reconstruct(patches, image_shape=img.shape, stride=1)
        assert within_pixel_bound(out, img, coverage_counts(8, 8, 3, 3, 1, 1))

    def test_asymmetric_overlap(self) -> None:
        """counts on both axes are {1, 2} -> 2-D map in {1, 2, 4} -> exact."""
        img = rand_image(1, 12, 18, torch.float64, seed=203)
        patches = extract(img, patch_size=(4, 6), stride=(2, 3))
        out = reconstruct(patches, image_shape=img.shape, stride=(2, 3))
        assert bit_equal(out, img)

    def test_float32_overlap_within_pixel_bound(self) -> None:
        """Amendment A: outside the predicate the error is bounded per pixel
        by (k+1)*eps*|v|, and it grows with the coverage count, so no fixed ULP
        figure applies. (Was: `rtol=1e-5` on a p4 s2 geometry, which is
        *inside* the predicate and therefore bit-exact, so the old assertion
        tested nothing about the error regime.)"""
        img = rand_image(1, 16, 16, torch.float32, seed=204)
        patches = extract(img, patch_size=4, stride=1)
        out = reconstruct(patches, image_shape=img.shape, stride=1)
        assert within_pixel_bound(out, img, coverage_counts(16, 16, 4, 4, 1, 1))


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
        img = rand_image(1, 8, 8, torch.float32, seed=106).cuda()
        patches = extract(img, patch_size=4, stride=4)
        out = reconstruct(patches, image_shape=img.shape, stride=4)
        assert out.device.type == "cuda"
        assert bit_equal(out, img)

    def test_accepts_torch_size(self) -> None:
        """torch.Tensor.shape returns torch.Size (a tuple subclass)."""
        img = rand_image(1, 8, 8, torch.float32, seed=107)
        patches = extract(img, patch_size=4, stride=4)
        out = reconstruct(patches, image_shape=img.shape, stride=4)
        assert isinstance(img.shape, torch.Size)
        assert bit_equal(out, img)


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

class TestHalfPrecisionAccumulation:
    """Regression (0.2.0 audit, §9.2): F.fold accumulates the sum of all
    overlapping patches *before* the count-map division, so a float16 constant
    image of 10000.0 with patch 3 stride 1 returned inf in 144 of 256 pixels.
    Accumulation is now done in float32 for half-precision inputs."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_large_constant_no_overflow(self, dtype: torch.dtype) -> None:
        img = torch.full((1, 16, 16), 10000.0, dtype=dtype)
        patches = extract(img, patch_size=3, stride=1)
        out = reconstruct(patches, image_shape=img.shape, stride=1)
        assert out.dtype == dtype
        assert torch.isfinite(out).all()
        assert torch.allclose(out, img)

    def test_int_dtype_rejected_with_clear_error(self) -> None:
        """Integer dtypes fail inside F.fold with a raw NotImplementedError;
        raise a clear ValueError instead (§9.1/§9.2: integers unsupported)."""
        patches = torch.zeros(4, 1, 4, 4, dtype=torch.uint8)
        with pytest.raises(ValueError, match="floating-point"):
            reconstruct(patches, image_shape=(1, 8, 8), stride=4)


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
        """§9.2: sh > ph or sw > pw forbidden, partial coverage."""
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


class TestCoverageGuard:
    """Regression (0.2.0 audit): the grid-consistency check validated only the
    patch *count*, never the *coverage*. A truncated grid returned a
    partly-black image instead of raising, contradicting §9.2."""

    def test_truncated_grid_10x10_patch_4(self) -> None:
        """10x10 with patch 4 stride 4 covers only 8 rows/cols: 36 of 100
        pixels came back zero in 0.2.0. Now raises."""
        patches = torch.zeros(4, 1, 4, 4)
        with pytest.raises(ValueError, match="coverage"):
            reconstruct(patches, image_shape=(1, 10, 10), stride=4)

    def test_truncated_grid_13x13_patch_5(self) -> None:
        """13x13 with patch 5 stride 5 covers only 10 rows/cols: 69 of 169
        pixels came back zero in 0.2.0. Now raises."""
        patches = torch.zeros(4, 1, 5, 5)
        with pytest.raises(ValueError, match="coverage"):
            reconstruct(patches, image_shape=(1, 13, 13), stride=5)

    def test_truncated_grid_overlap(self) -> None:
        """Overlap does not excuse truncation: 9x9 with patch 4 stride 2
        covers (3-1)*2+4 = 8 rows/cols, one short."""
        patches = torch.zeros(9, 1, 4, 4)
        with pytest.raises(ValueError, match="coverage"):
            reconstruct(patches, image_shape=(1, 9, 9), stride=2)

    def test_exact_coverage_boundary_still_accepted(self) -> None:
        """Grid that ends exactly on the image edge must not raise."""
        img = rand_image(1, 10, 10, torch.float32, seed=301)
        patches = extract(img, patch_size=4, stride=2)  # (4-1)*2+4 == 10
        out = reconstruct(patches, image_shape=img.shape, stride=2)
        assert bit_equal(out, img)


# ----------------------------------------------------- Characterization (0.3.0) --


def _fold_reference(patches, image_shape, stride):
    """The pre-0.3.0 reconstruct implementation, as ground truth."""
    n_patches, c, ph, pw = patches.shape
    h, w = image_shape[1], image_shape[2]
    sh, sw = (stride, stride) if isinstance(stride, int) else stride
    accum_dtype = (
        torch.float32
        if patches.dtype in (torch.float16, torch.bfloat16)
        else patches.dtype
    )
    work = patches.to(accum_dtype)
    flat = work.permute(1, 2, 3, 0).reshape(c * ph * pw, n_patches).unsqueeze(0)
    folded = F.fold(flat, (h, w), (ph, pw), stride=(sh, sw))
    ones = torch.ones(1, ph * pw, n_patches, dtype=accum_dtype, device=patches.device)
    count = F.fold(ones, (h, w), (ph, pw), stride=(sh, sw))
    return (folded / count.clamp(min=1e-6))[0].to(patches.dtype)


@pytest.mark.parametrize("c,h,w", [(1, 8, 8), (3, 16, 16), (3, 32, 24), (4, 15, 15)])
@pytest.mark.parametrize(
    "ph,pw,sh,sw",
    [
        (2, 2, 2, 2),   # exact non-overlap -> fast path
        (4, 4, 4, 4),   # exact non-overlap, larger
        (4, 4, 2, 2),   # overlap 50%
        (3, 3, 1, 1),   # max overlap
        (4, 6, 2, 3),   # rectangular patch, anisotropic stride
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
def test_reconstruct_matches_fold_reference(c, h, w, ph, pw, sh, sw, dtype):
    num_h = (h - ph) // sh + 1
    num_w = (w - pw) // sw + 1
    if (num_h - 1) * sh + ph != h or (num_w - 1) * sw + pw != w:
        pytest.skip("geometry does not cover exactly")
    img = torch.randn(c, h, w, dtype=dtype)
    from patchcraft import extract
    patches = extract(img, (ph, pw), (sh, sw))
    got = reconstruct(patches, image_shape=(c, h, w), stride=(sh, sw))
    ref = _fold_reference(patches, (c, h, w), (sh, sw))
    assert got.shape == ref.shape
    assert got.dtype == ref.dtype
    assert torch.equal(got, ref)


# ------------------------------------------------- Result independence ------
@pytest.mark.parametrize("shape,patch", [((1, 8, 8), 8), ((3, 4, 4), 4), ((2, 1, 1), 1)])
def test_single_patch_grid_does_not_alias_the_caller(shape, patch):
    """A one-patch grid is a pure rearrangement, and reshape returned a view.

    The caller's patches and the returned image then shared storage, so a write
    to the patches changed an image this function had already returned.
    `extract` guards the mirror image of this case.
    """
    from patchcraft import extract

    img = rand_image(*shape, torch.float32, seed=11)
    patches = extract(img, patch, stride=patch)
    out = reconstruct(patches, img.shape, stride=patch)

    assert out.untyped_storage().data_ptr() != patches.untyped_storage().data_ptr()
    snapshot = out.clone()
    patches.zero_()
    assert torch.equal(out, snapshot)


# ------------------------------------------- Why half precision is promoted --
def test_float16_overflows_the_fold_but_bfloat16_cannot():
    """THEORY 9.2 promoted both half formats citing fp16's finite maximum.

    Only one of them overflows. `bfloat16` carries `float32`'s exponent, so the
    promotion buys it precision rather than range, and until 0.5.2 the contract
    section stated a reason that applies to one of the two formats.
    """
    h = w = 16
    kernel, stride = 3, 1
    cols = F.unfold(
        torch.full((1, 1, h, w), 10000.0), kernel_size=kernel, stride=stride
    )

    numerators = {}
    for dtype in (torch.float16, torch.bfloat16):
        folded = F.fold(
            cols.to(dtype), output_size=(h, w), kernel_size=kernel, stride=stride
        )
        numerators[dtype] = folded

    assert torch.isinf(numerators[torch.float16]).any(), "fp16 no longer overflows here"
    assert not torch.isinf(numerators[torch.bfloat16]).any()
    assert numerators[torch.bfloat16].max().item() < torch.finfo(torch.bfloat16).max


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_round_trip_survives_the_overflow_case(dtype):
    """Whatever the reason for the promotion, neither format may return inf."""
    img = torch.full((1, 16, 16), 10000.0, dtype=dtype)
    patches = extract(img, 3, stride=1)
    out = reconstruct(patches, img.shape, stride=1)
    assert not torch.isinf(out).any()
    assert out.dtype == dtype
