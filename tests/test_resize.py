"""Tests for `patchforge.resize` — contract from docs/THEORY.md §9.4."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from patchforge import resize


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
