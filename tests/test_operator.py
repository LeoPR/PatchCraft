"""The linear operator behind extract, reconstruct and stitch (THEORY §2.6).

extract is a selection matrix U: every output entry copies one pixel. Its
adjoint is the fold, U^T U is the diagonal of the count map, reconstruct is
the least-squares inverse (U^T U)^-1 U^T, and stitch is the weighted one,
(U^T W U)^-1 U^T W. These tests pin each of those statements, so the section
that states them can be falsified, and they pin the three recipes GUIDE
builds on them: a filter applied per tile with a halo, the padded round trip,
and what averaging does to a per-patch quantization.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)

from patchcraft import extract, reconstruct, stitch
from patchcraft.stitch import _window_kernel
from tests._rng import bit_equal, coverage_counts, rand_image

# (h, w, ph, pw, sh, sw), each covering its image exactly. The second puts a 3
# in the count map, the third does not overlap at all.
_GEOMETRIES = [
    (16, 16, 4, 4, 2, 2),
    (12, 10, 4, 3, 2, 1),
    (9, 15, 3, 5, 3, 5),
    (11, 13, 5, 3, 3, 2),
]


@pytest.fixture(params=["1", "0"], ids=["accel", "pure"])
def accel(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("PATCHCRAFT_ACCEL", request.param)
    return str(request.param)


def _patches_like(x: torch.Tensor, ph: int, pw: int, sh: int, sw: int, seed: int) -> torch.Tensor:
    """A random patch stack of the right shape, almost surely not extract of any image."""
    n = extract(x, patch_size=(ph, pw), stride=(sh, sw)).shape[0]
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, x.shape[0], ph, pw, generator=g, dtype=x.dtype)


def _counts(h: int, w: int, ph: int, pw: int, sh: int, sw: int) -> torch.Tensor:
    return torch.from_numpy(coverage_counts(h, w, ph, pw, sh, sw)).to(torch.float64)


@pytest.mark.parametrize("geom", _GEOMETRIES)
def test_the_gradient_of_extract_is_the_fold(geom: tuple[int, ...]) -> None:
    """U^T P, read off autograd, is the count map times reconstruct(P)."""
    h, w, ph, pw, sh, sw = geom
    x = rand_image(2, h, w, torch.float64, seed=1).requires_grad_(True)
    p = _patches_like(x.detach(), ph, pw, sh, sw, seed=2)
    (ut_p,) = torch.autograd.grad(extract(x, patch_size=(ph, pw), stride=(sh, sw)), x, p)
    k = _counts(h, w, ph, pw, sh, sw)
    torch.testing.assert_close(ut_p, k * reconstruct(p, image_shape=x.shape, stride=(sh, sw)))
    # The adjoint identity <U x, P> == <x, U^T P>.
    lhs = (extract(x.detach(), patch_size=(ph, pw), stride=(sh, sw)) * p).sum()
    torch.testing.assert_close(lhs, (x.detach() * ut_p).sum())


@pytest.mark.parametrize("geom", _GEOMETRIES)
def test_u_transpose_u_is_the_count_map(geom: tuple[int, ...]) -> None:
    h, w, ph, pw, sh, sw = geom
    x = rand_image(2, h, w, torch.float64, seed=3).requires_grad_(True)
    ux = extract(x, patch_size=(ph, pw), stride=(sh, sw))
    (utu_x,) = torch.autograd.grad(ux, x, ux.detach())
    torch.testing.assert_close(utu_x, _counts(h, w, ph, pw, sh, sw) * x.detach())


@pytest.mark.parametrize("geom", _GEOMETRIES)
def test_reconstruct_is_the_least_squares_inverse(accel: str, geom: tuple[int, ...]) -> None:
    """For any patch stack, reconstruct minimises ||extract(x) - P||^2."""
    h, w, ph, pw, sh, sw = geom
    shape = (2, h, w)
    p = _patches_like(torch.zeros(shape, dtype=torch.float64), ph, pw, sh, sw, seed=4)
    x_hat = reconstruct(p, image_shape=shape, stride=(sh, sw)).requires_grad_(True)
    loss = ((extract(x_hat, patch_size=(ph, pw), stride=(sh, sw)) - p) ** 2).sum()
    (grad,) = torch.autograd.grad(loss, x_hat)
    assert grad.abs().max() < 1e-12


@pytest.mark.parametrize("weight", ["uniform", "hann", "gaussian"])
@pytest.mark.parametrize("geom", _GEOMETRIES)
def test_stitch_is_the_weighted_least_squares_inverse(
    accel: str, geom: tuple[int, ...], weight: str
) -> None:
    """stitch minimises the window-weighted ||extract(x) - P||^2."""
    h, w, ph, pw, sh, sw = geom
    shape = (2, h, w)
    p = _patches_like(torch.zeros(shape, dtype=torch.float64), ph, pw, sh, sw, seed=5)
    kernel = _window_kernel(weight, ph, pw, torch.float64, torch.device("cpu"))
    x_hat = stitch(p, shape, stride=(sh, sw), weight=weight).requires_grad_(True)
    resid = extract(x_hat, patch_size=(ph, pw), stride=(sh, sw)) - p
    (grad,) = torch.autograd.grad((kernel * resid**2).sum(), x_hat)
    assert grad.abs().max() < 1e-12


def test_a_pointwise_operation_commutes_with_extract() -> None:
    """Quantizing before or after cutting gives the same patches, bit for bit."""
    x = rand_image(3, 16, 16, torch.float32, seed=6)

    def quantize(t: torch.Tensor) -> torch.Tensor:
        return torch.round(t * 15) / 15

    assert bit_equal(extract(quantize(x), 4, 2), quantize(extract(x, 4, 2)))


def test_a_filter_per_tile_with_a_halo_is_the_whole_image_filter() -> None:
    """Overlap-save: tiles of p + 2r at stride p, a valid convolution, a rearrangement."""
    x = rand_image(1, 32, 32, torch.float32, seed=7)
    p, r = 8, 2
    kernel = torch.ones(1, 1, 2 * r + 1, 2 * r + 1) / (2 * r + 1) ** 2
    padded = F.pad(x.unsqueeze(0), (r, r, r, r), mode="reflect")[0]
    whole = F.conv2d(padded.unsqueeze(0), kernel)[0]
    tiles = extract(padded, patch_size=p + 2 * r, stride=p)
    tiled = reconstruct(F.conv2d(tiles, kernel), image_shape=x.shape, stride=p)
    # Convolution chooses its own summation order, so the test allows for it;
    # the GUIDE run shows the two equal bit for bit.
    torch.testing.assert_close(tiled, whole, rtol=0, atol=1e-6)


@pytest.mark.parametrize("stride", [8, 4])
def test_the_padded_round_trip_returns_the_image(accel: str, stride: int) -> None:
    """pad -> extract -> reconstruct -> crop, at sizes no tiling covers."""
    x = rand_image(3, 27, 33, torch.float32, seed=8)
    patch = 8
    pad_h = -(27 - patch) % stride
    pad_w = -(33 - patch) % stride
    padded = F.pad(x.unsqueeze(0), (0, pad_w, 0, pad_h), mode="reflect")[0]
    back = reconstruct(extract(padded, patch, stride), image_shape=padded.shape, stride=stride)
    # Stride 4 is half the patch, so every count is 1, 2 or 4 and the round
    # trip is exact by the rule in THEORY §9.2; stride 8 only rearranges.
    assert bit_equal(back[:, :27, :33], x)


def test_averaging_a_per_patch_quantization_leaves_its_alphabet() -> None:
    """Patches that decide differently about a shared pixel average to a value in between."""
    x = rand_image(1, 16, 16, torch.float32, seed=9)
    for stride, stays in ((4, True), (2, False)):
        ps = extract(x, 4, stride)
        binary = (ps > ps.mean(dim=(1, 2, 3), keepdim=True)).to(x.dtype)
        values = set(reconstruct(binary, image_shape=x.shape, stride=stride).unique().tolist())
        assert (values <= {0.0, 1.0}) is stays
