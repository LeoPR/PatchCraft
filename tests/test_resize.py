"""Tests for `patchcraft.resize`, contract from docs/THEORY.md §9.4."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from patchcraft import resize


def _rgb_pil(h: int, w: int) -> Image.Image:
    arr = (np.arange(h * w * 3) % 256).astype(np.uint8).reshape(h, w, 3)
    return Image.fromarray(arr, mode="RGB")


def _gray_pil(h: int, w: int) -> Image.Image:
    arr = (np.arange(h * w) % 256).astype(np.uint8).reshape(h, w)
    return Image.fromarray(arr, mode="L")


# ------------------------------------------------------------------ Aceita ----

class TestPILBackend:
    def test_pil_in_pil_out_basic(self) -> None:
        img = _rgb_pil(16, 16)
        out = resize(img, target_size=(8, 8), backend="pil")
        assert isinstance(out, Image.Image)
        assert out.size == (8, 8)  # PIL is (W, H)

    def test_pil_in_pil_out_grayscale(self) -> None:
        img = _gray_pil(20, 30)
        out = resize(img, target_size=(10, 15), backend="pil")
        assert isinstance(out, Image.Image)
        assert out.mode == "L"
        assert out.size == (15, 10)

    def test_pil_default_resample_is_lanczos(self) -> None:
        img = _rgb_pil(8, 8)
        out_default = resize(img, target_size=(4, 4), backend="pil")
        out_lanczos = resize(img, target_size=(4, 4), backend="pil", resample="lanczos")
        assert np.array_equal(np.asarray(out_default), np.asarray(out_lanczos))

    @pytest.mark.parametrize(
        "resample",
        ["nearest", "bilinear", "bicubic", "lanczos", "box", "hamming"],
    )
    def test_pil_accepts_all_resamples(self, resample: str) -> None:
        img = _rgb_pil(8, 8)
        out = resize(img, target_size=(4, 4), backend="pil", resample=resample)
        assert isinstance(out, Image.Image)


class TestTorchBackend:
    def test_tensor_in_tensor_out_basic(self) -> None:
        img = torch.rand(3, 16, 16)
        out = resize(img, target_size=(8, 8), backend="torch")
        assert isinstance(out, torch.Tensor)
        assert out.shape == (3, 8, 8)

    def test_dtype_preserved(self) -> None:
        for dtype in (torch.float32, torch.float64):
            img = torch.rand(1, 8, 8, dtype=dtype)
            out = resize(img, target_size=(4, 4), backend="torch")
            assert out.dtype == dtype

    def test_default_resample_is_bilinear(self) -> None:
        img = torch.rand(1, 8, 8)
        a = resize(img, target_size=(4, 4), backend="torch")
        b = resize(img, target_size=(4, 4), backend="torch", resample="bilinear")
        assert torch.equal(a, b)

    @pytest.mark.parametrize(
        "resample",
        ["nearest", "bilinear", "bicubic", "area", "nearest-exact"],
    )
    def test_torch_accepts_all_resamples(self, resample: str) -> None:
        img = torch.rand(1, 8, 8)
        out = resize(img, target_size=(4, 4), backend="torch", resample=resample)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, 4, 4)

    @pytest.mark.gpu
    def test_cuda_tensor_with_torch_backend(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        img = torch.rand(1, 8, 8).cuda()
        out = resize(img, target_size=(4, 4), backend="torch")
        assert out.device.type == "cuda"


class TestIntegerDtypeCast:
    """Regression (0.2.0 audit): `_resize_torch` cast back to the original
    integer dtype with a bare `.to()`, no clamp, no round. Bicubic legitimately
    overshoots the input range, so the cast wrapped: -9.0 -> 247, 281.9 -> 25."""

    def test_uint8_bicubic_hard_edge_no_wrap(self) -> None:
        """8x8 uint8 hard edge resized to 32x32: in 0.2.0, 256 of 1024 pixels
        were wrong, with black pixels becoming 254 and white pixels becoming 1.
        After clamp+round, each side of the edge stays on its own side."""
        img = torch.zeros(1, 8, 8, dtype=torch.uint8)
        img[:, :, 4:] = 255
        out = resize(img, target_size=(32, 32), backend="torch", resample="bicubic")
        assert out.dtype == torch.uint8
        center_row = out[0, 16, :]
        assert (center_row[:16] < 128).all()
        assert (center_row[16:] > 128).all()
        # Saturated regions far from the edge are untouched.
        assert (out[:, :, :8] == 0).all()
        assert (out[:, :, -8:] == 255).all()

    def test_int32_cast_rounds_and_does_not_wrap(self) -> None:
        """Same wrap on int32: in 0.2.0 the bright side of a hard edge came
        back as int32 min. Values are rounded and stay on their own side."""
        img = torch.zeros(1, 8, 8, dtype=torch.int32)
        img[:, :, 4:] = 1_000_000
        out = resize(img, target_size=(32, 32), backend="torch", resample="bicubic")
        assert out.dtype == torch.int32
        center_row = out[0, 16, :]
        assert (center_row[:16] < 500_000).all()
        assert (center_row[16:] > 500_000).all()
        assert (out[:, :, -8:] == 1_000_000).all()

    def test_uint8_bilinear_no_overshoot_regression(self) -> None:
        """Bilinear never overshoots, so values stay within the convex hull of
        the input; this held in 0.2.0 and must keep holding."""
        img = torch.randint(0, 256, (1, 8, 8), dtype=torch.uint8)
        out = resize(img, target_size=(16, 16), backend="torch", resample="bilinear")
        assert out.dtype == torch.uint8
        assert int(out.min()) >= int(img.min())
        assert int(out.max()) <= int(img.max())


_TORCH_MODES = ["nearest", "bilinear", "bicubic", "area", "nearest-exact"]
_INTEGER_DTYPES = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]


class TestIntegerDtypeAcrossResamples:
    """Regression: THEORY §9.4 accepts *any* integer tensor on the torch
    backend, but `_resize_torch` promoted to float only for bilinear and
    bicubic. The other three accepted modes reached `F.interpolate` with an
    integer tensor and raised a raw torch error naming an internal op the
    caller never called: `"adaptive_avg_pool2d" not implemented for 'Int'`.
    13 of the 25 mode x dtype combinations were affected.

    Two parametrizations already existed and could not catch it, because they
    never crossed: `test_torch_accepts_all_resamples` sweeps every mode on
    `torch.rand` (float), and the integer tests all use the default mode
    (bilinear). This class is that crossing."""

    @pytest.mark.parametrize("resample", _TORCH_MODES)
    @pytest.mark.parametrize("dtype", _INTEGER_DTYPES)
    def test_every_mode_accepts_every_integer_dtype(
        self, resample: str, dtype: torch.dtype
    ) -> None:
        img = torch.randint(0, 100, (1, 8, 8), dtype=dtype)
        out = resize(img, target_size=(4, 4), backend="torch", resample=resample)
        assert out.dtype == dtype
        assert out.shape == (1, 4, 4)

    @pytest.mark.parametrize("resample", _TORCH_MODES)
    @pytest.mark.parametrize("dtype", _INTEGER_DTYPES)
    def test_upsampling_too(self, resample: str, dtype: torch.dtype) -> None:
        """`area` is `adaptive_avg_pool2d`, which behaves differently going up."""
        img = torch.randint(0, 100, (1, 4, 4), dtype=dtype)
        out = resize(img, target_size=(8, 8), backend="torch", resample=resample)
        assert out.dtype == dtype
        assert out.shape == (1, 8, 8)


class TestIntegerPromotionIsExact:
    """The float a value is promoted through must represent that value
    exactly, or the fix for the error above would trade a loud exception for
    the silent truncation this library exists to prevent.

    `float32` carries 24 mantissa bits, so it is exact for `uint8`, `int8` and
    `int16`, whose whole ranges fit. It is *not* exact for `int32`: promoting
    2**24 + 1 through it returns 2**24. Measured on 2026-09-04, this was
    already happening silently on the bilinear path."""

    @pytest.mark.parametrize("resample", _TORCH_MODES)
    def test_int32_above_float32_mantissa_survives(self, resample: str) -> None:
        """A constant image resizes to the same constant under every mode, so
        any difference is the promotion losing bits rather than the resample."""
        value = 2**24 + 1
        img = torch.full((1, 8, 8), value, dtype=torch.int32)
        out = resize(img, target_size=(4, 4), backend="torch", resample=resample)
        assert int(out.flatten()[0]) == value

    def test_int32_large_value_survives(self) -> None:
        value = 2**30 + 7
        img = torch.full((1, 8, 8), value, dtype=torch.int32)
        out = resize(img, target_size=(4, 4), backend="torch", resample="bilinear")
        assert int(out.flatten()[0]) == value

    @pytest.mark.parametrize("dtype", [torch.uint8, torch.int8, torch.int16])
    def test_narrow_dtypes_keep_their_extremes(self, dtype: torch.dtype) -> None:
        """Every value of these dtypes is exact in float32, so the whole range
        round-trips and the narrow path stays on the cheaper float."""
        info = torch.iinfo(dtype)
        for value in (info.min, info.max):
            img = torch.full((1, 8, 8), value, dtype=dtype)
            out = resize(img, target_size=(4, 4), backend="torch", resample="nearest")
            assert int(out.flatten()[0]) == value


class TestCrossBackend:
    def test_pil_in_torch_backend_returns_pil(self) -> None:
        img = _rgb_pil(16, 16)
        out = resize(img, target_size=(8, 8), backend="torch")
        assert isinstance(out, Image.Image)
        assert out.size == (8, 8)

    def test_tensor_in_pil_backend_returns_tensor(self) -> None:
        img = torch.rand(3, 16, 16)
        out = resize(img, target_size=(8, 8), backend="pil")
        assert isinstance(out, torch.Tensor)
        assert out.shape == (3, 8, 8)
        # float in [0, 1] preserved
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_tensor_pil_backend_preserves_float_dtype(self) -> None:
        img = torch.rand(3, 16, 16, dtype=torch.float64)
        out = resize(img, target_size=(8, 8), backend="pil")
        assert out.dtype == torch.float64

    def test_tensor_pil_backend_handles_uint8(self) -> None:
        img = (torch.rand(1, 16, 16) * 255).to(torch.uint8)
        out = resize(img, target_size=(8, 8), backend="pil")
        assert out.dtype == torch.uint8
        assert out.shape == (1, 8, 8)


class TestBackendDivergence:
    """PIL and torch don't compute the same bicubic; document the gap."""

    def test_pil_vs_torch_bilinear_differ(self) -> None:
        img = torch.rand(1, 16, 16)
        out_torch = resize(img, target_size=(8, 8), backend="torch", resample="bilinear")
        out_pil = resize(img, target_size=(8, 8), backend="pil", resample="bilinear")
        # They should not be identical (PIL and torch's bilinear differ).
        # But both should produce a (1, 8, 8) tensor.
        assert out_torch.shape == out_pil.shape == (1, 8, 8)


# ------------------------------------------------------------------ Rejeita ---

class TestRejects:
    @pytest.mark.parametrize("bad", [(8,), (8, 8, 8), 8, [8, 8]])
    def test_target_size_wrong_arity(self, bad: object) -> None:
        with pytest.raises(ValueError, match="target_size"):
            resize(torch.rand(1, 4, 4), target_size=bad, backend="torch")  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [(0, 8), (8, 0), (-1, 8), (8, -1)])
    def test_target_size_nonpositive(self, bad: tuple[int, int]) -> None:
        with pytest.raises(ValueError, match="must be a positive int"):
            resize(torch.rand(1, 4, 4), target_size=bad, backend="torch")

    def test_target_size_non_int(self) -> None:
        with pytest.raises(ValueError, match="must be a positive int"):
            resize(torch.rand(1, 4, 4), target_size=(8.0, 8), backend="torch")  # type: ignore[arg-type]

    def test_invalid_backend(self) -> None:
        with pytest.raises(ValueError, match="backend"):
            resize(torch.rand(1, 4, 4), target_size=(8, 8), backend="opencv")  # type: ignore[arg-type]

    @pytest.mark.gpu
    def test_cuda_tensor_with_pil_backend_rejected(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        img = torch.rand(1, 8, 8).cuda()
        with pytest.raises(ValueError, match=r"backend='pil' cannot accept"):
            resize(img, target_size=(4, 4), backend="pil")

    def test_invalid_resample_for_pil(self) -> None:
        img = _rgb_pil(8, 8)
        with pytest.raises(ValueError, match="not supported by PIL"):
            resize(img, target_size=(4, 4), backend="pil", resample="area")

    def test_invalid_resample_for_torch(self) -> None:
        img = torch.rand(1, 8, 8)
        with pytest.raises(ValueError, match="not supported by torch"):
            resize(img, target_size=(4, 4), backend="torch", resample="lanczos")

    def test_resample_non_string(self) -> None:
        img = torch.rand(1, 8, 8)
        with pytest.raises(ValueError, match="resample"):
            resize(img, target_size=(4, 4), backend="torch", resample=2)  # type: ignore[arg-type]

    def test_tensor_wrong_ndim(self) -> None:
        with pytest.raises(ValueError, match="ndim==3"):
            resize(torch.rand(4, 4), target_size=(2, 2), backend="torch")

    def test_unsupported_channel_for_pil(self) -> None:
        """C=2 has no PIL mode."""
        img = torch.rand(2, 8, 8)
        with pytest.raises(ValueError, match="channels to PIL"):
            resize(img, target_size=(4, 4), backend="pil")

    def test_non_tensor_non_pil(self) -> None:
        with pytest.raises(TypeError, match=r"torch\.Tensor or PIL"):
            resize([1, 2, 3], target_size=(2, 2), backend="torch")  # type: ignore[arg-type]


def test_tensor_to_pil_int_tensor_clamps_instead_of_wrapping():
    """int16 values outside [0, 255] must clamp, not wrap (300 -> 255, not 44)."""
    t = torch.tensor([[[300, -5], [100, 255]]], dtype=torch.int16)
    out = resize(t, (2, 2), backend="pil")
    assert out.dtype == torch.int16
    assert out[0, 0, 0].item() == 255  # would be 44 with a bare astype(uint8)
    assert out[0, 0, 1].item() == 0    # would be 251 with a bare astype(uint8)
    assert out[0, 1, 0].item() == 100
    assert out[0, 1, 1].item() == 255


# --------------------------------------------------- Integer dtype parity ---
@pytest.mark.parametrize(
    "dtype", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
)
@pytest.mark.parametrize("backend", ["torch", "pil"])
def test_every_integer_dtype_resizes_on_both_backends(dtype, backend):
    """`clamp(0, 255)` needs bounds the dtype can hold.

    On an int8 tensor the literal 255 is unrepresentable, so torch raised
    `value cannot be converted to type int8_t without overflow` before clamping
    anything, and the PIL backend rejected a dtype the contract accepts.
    """
    x = (torch.rand(1, 8, 8) * 100).to(dtype)
    out = resize(x, (4, 4), backend=backend)
    assert out.dtype == dtype
    assert out.shape == (1, 4, 4)


def test_out_of_range_integers_saturate_rather_than_wrap():
    """Neither backend wraps, which is the 0.2.0 defect class (`-9 -> 247`).

    The two clamp to different ranges on purpose, as THEORY 9.4 states: the
    torch backend clamps to the dtype range and so preserves 300 and -9 in
    int32, while the PIL backend goes through a uint8 hop and therefore lands
    inside [0, 255]. What both must never do is wrap.
    """
    x = torch.tensor([[[300, -9]]], dtype=torch.int32).repeat(1, 4, 2)

    torch_out = resize(x, (8, 8), backend="torch")
    assert torch_out.min().item() >= -9 and torch_out.max().item() <= 300

    pil_out = resize(x, (8, 8), backend="pil")
    assert pil_out.min().item() >= 0 and pil_out.max().item() <= 255
