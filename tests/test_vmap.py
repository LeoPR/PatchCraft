"""torch.vmap over the one-image primitives (THEORY §9.1, SCOPE.md).

The scope is one image per call, and the documented way to batch is the
caller's: a loop, a DataLoader, or torch.vmap. vmap passes in tensors with no
storage of their own, so any code that asks a tensor for its storage or data
pointer breaks under it. Until 0.5.6, extract and the non-overlapping
reconstruct did exactly that, and the accelerator read data pointers, so vmap
failed on most calls. These tests pin vmap to a plain loop, bit for bit, on
both code paths.
"""
from __future__ import annotations

import pytest
import torch

from patchcraft import extract, reconstruct, stitch
from tests._rng import bit_equal

_DTYPES = [torch.float32, torch.float64, torch.float16, torch.bfloat16]
_SIZE = 32


@pytest.fixture(params=["1", "0"], ids=["accel", "pure"])
def accel(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("PATCHCRAFT_ACCEL", request.param)
    return str(request.param)


def _batch(dtype: torch.dtype) -> torch.Tensor:
    g = torch.Generator().manual_seed(11)
    return torch.rand(3, 2, _SIZE, _SIZE, generator=g).to(dtype)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize(
    ("patch", "stride", "dilation"),
    [(8, 4, 1), (8, 8, 1), (8, 4, 2), (1, 1, 1), (_SIZE, _SIZE, 1)],
)
def test_vmap_extract_equals_loop(
    accel: str, dtype: torch.dtype, patch: int, stride: int, dilation: int
) -> None:
    batch = _batch(dtype)
    got = torch.vmap(lambda im: extract(im, patch, stride, dilation=dilation))(batch)
    want = torch.stack([extract(im, patch, stride, dilation=dilation) for im in batch])
    assert bit_equal(got, want)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize(("patch", "stride"), [(8, 4), (8, 8), (_SIZE, _SIZE)])
@pytest.mark.parametrize("op", ["reconstruct", "uniform", "hann"])
def test_vmap_fold_equals_loop(
    accel: str, dtype: torch.dtype, patch: int, stride: int, op: str
) -> None:
    shape = (2, _SIZE, _SIZE)
    patches = torch.stack([extract(im, patch, stride) for im in _batch(dtype)])

    def fold(p: torch.Tensor) -> torch.Tensor:
        if op == "reconstruct":
            return reconstruct(p, shape, stride)
        return stitch(p, shape, stride, weight=op)  # type: ignore[arg-type]

    got = torch.vmap(fold)(patches)
    want = torch.stack([fold(p) for p in patches])
    assert bit_equal(got, want)


def test_vmap_round_trip_is_exact(accel: str) -> None:
    """The documented use: batch the round trip, inside the exactness rule."""
    batch = _batch(torch.float32)
    patches = torch.vmap(lambda im: extract(im, 8, 4))(batch)
    back = torch.vmap(lambda p: reconstruct(p, (2, _SIZE, _SIZE), 4))(patches)
    assert bit_equal(back, batch)
