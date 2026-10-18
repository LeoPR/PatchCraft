"""Forward-mode AD through reconstruct and stitch, on both code paths (THEORY §2.6).

The fold is linear, so the tangent of ``reconstruct(P)`` along ``T`` is
``reconstruct(T)``, and likewise for ``stitch``; torch's own derivative of the
pure path returns exactly that, bit for bit, in all four float dtypes. The
accelerator used to drop it. It handed a call back to torch only when the
patches required grad, and a ``torch.autograd.forward_ad`` dual does not, so
the dual went through the native kernel, which writes a plain tensor. Measured
on 0.6.0 at torch 2.6.0 and 2.14.1: 88 of 96 calls (six geometries, four
operations, four dtypes) lost the tangent with the accelerator and none
without it, exactly the calls that reach the kernel; ``torch.func.linearize``
and ``torch.autograd.functional.jacobian(strategy="forward-mode")`` returned
all-zero derivatives, and a tangent that required grad lost its graph.
``torch.func.jvp`` was never affected: it passes wrappers without storage,
which the kernel already declined.
"""
from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
import torch.autograd.forward_ad as fwad

from patchcraft import _accel, extract, reconstruct, stitch
from tests._rng import bit_equal, rand_image

real_accel = pytest.mark.skipif(
    not _accel.accel_available(), reason="native accelerator not built into this install"
)

_DTYPES = [torch.float32, torch.float64, torch.float16, torch.bfloat16]
_OPS = ["reconstruct", "uniform", "hann", "gaussian"]

# (c, h, w, ph, pw, sh, sw, exact): exact when every count is 1, 2 or 4. The
# windowed denominator takes prefix and suffix sums where an axis has at least
# as many patches as stride-long window blocks, and van Herk run sums where it
# has fewer: "few-patches" is the one geometry here of the second kind.
_GEOMETRIES = {
    "overlap": (3, 16, 16, 4, 4, 2, 2, True),
    "overlap-rect": (2, 16, 14, 4, 6, 2, 4, True),
    "overlap-count-3": (2, 12, 10, 4, 3, 2, 1, False),
    "no-overlap": (3, 16, 16, 4, 4, 4, 4, True),
    "one-axis": (3, 16, 14, 4, 4, 4, 2, True),
    "few-patches": (1, 10, 10, 8, 8, 2, 2, True),
}


@pytest.fixture(params=["1", "0"], ids=["accel", "pure"])
def accel(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("PATCHCRAFT_ACCEL", request.param)
    return str(request.param)


def _fold(
    op: str, shape: tuple[int, int, int], stride: tuple[int, int]
) -> Callable[[torch.Tensor], torch.Tensor]:
    if op == "reconstruct":
        return lambda p: reconstruct(p, shape, stride)
    return lambda p: stitch(p, shape, stride, weight=op)  # type: ignore[arg-type]


def _stack(geom: str, dtype: torch.dtype, seed: int) -> tuple[torch.Tensor, tuple[int, int, int]]:
    c, h, w, ph, pw, sh, sw, _ = _GEOMETRIES[geom]
    n = extract(torch.zeros(1, h, w), (ph, pw), (sh, sw)).shape[0]
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(n, c, ph, pw, generator=gen, dtype=dtype), (c, h, w)


@pytest.mark.parametrize("dtype", _DTYPES, ids=str)
@pytest.mark.parametrize("op", _OPS)
@pytest.mark.parametrize("geom", list(_GEOMETRIES))
def test_a_dual_keeps_its_tangent_through_the_fold(
    accel: str, geom: str, op: str, dtype: torch.dtype
) -> None:
    """The tangent of op(P) along T is op(T), bit for bit, on both paths."""
    _, _, _, _, _, sh, sw, _ = _GEOMETRIES[geom]
    patches, shape = _stack(geom, dtype, seed=11)
    tangent, _ = _stack(geom, dtype, seed=23)
    fold = _fold(op, shape, (sh, sw))
    with fwad.dual_level():
        primal, got = fwad.unpack_dual(fold(fwad.make_dual(patches, tangent)))
        assert got is not None, "the tangent was dropped"
        got, primal = got.clone(), primal.clone()
    assert bit_equal(primal, fold(patches))
    assert bit_equal(got, fold(tangent))


@pytest.mark.parametrize("dtype", _DTYPES, ids=str)
@pytest.mark.parametrize("op", _OPS)
@pytest.mark.parametrize("geom", list(_GEOMETRIES))
def test_the_round_trip_of_a_dual_returns_its_tangent(
    accel: str, geom: str, op: str, dtype: torch.dtype
) -> None:
    """extract then fold a dual image: the tangent makes the round trip the primal makes.

    Where every count is 1, 2 or 4, reconstruct (and stitch under the uniform
    window, the same arithmetic) returns the input, and then the tangent comes
    back as the input's tangent, bit for bit. Elsewhere, and under a window, it
    comes back as the round trip of the tangent taken without AD, bit for bit.
    """
    c, h, w, ph, pw, sh, sw, exact = _GEOMETRIES[geom]
    x = rand_image(c, h, w, dtype, seed=31)
    v = rand_image(c, h, w, dtype, seed=37)
    fold = _fold(op, (c, h, w), (sh, sw))
    with fwad.dual_level():
        dual = fwad.make_dual(x, v)
        primal, got = fwad.unpack_dual(fold(extract(dual, (ph, pw), (sh, sw))))
        assert got is not None, "the tangent was dropped"
        got, primal = got.clone(), primal.clone()
    assert bit_equal(got, fold(extract(v, (ph, pw), (sh, sw))))
    if exact and op in ("reconstruct", "uniform"):
        assert bit_equal(primal, x)
        assert bit_equal(got, v)


@pytest.mark.parametrize("op", _OPS)
def test_a_tangent_that_requires_grad_keeps_its_graph(accel: str, op: str) -> None:
    """Reverse over forward: the gradient with respect to the tangent is op's adjoint."""
    patches, shape = _stack("overlap", torch.float64, seed=41)
    tangent, _ = _stack("overlap", torch.float64, seed=43)
    tangent.requires_grad_(True)
    cot = rand_image(*shape, torch.float64, seed=47)
    fold = _fold(op, shape, (2, 2))
    with fwad.dual_level():
        got = fwad.unpack_dual(fold(fwad.make_dual(patches, tangent))).tangent
        assert got is not None and got.requires_grad
        (grad,) = torch.autograd.grad((got * cot).sum(), tangent)
    leaf = tangent.detach().clone().requires_grad_(True)
    (want,) = torch.autograd.grad((fold(leaf) * cot).sum(), leaf)
    assert bit_equal(grad, want)


@pytest.mark.parametrize("op", _OPS)
@pytest.mark.parametrize("geom", ["overlap", "few-patches"])
def test_linearize_returns_the_fold_of_the_tangent(accel: str, geom: str, op: str) -> None:
    """torch.func.linearize traces a forward-AD call; with the tangent lost it was zero.

    It replays the ops it recorded, so it is also the test that sees a write
    into a view of a fresh buffer in the window sums, which the replay drops;
    the two geometries reach both of their branches."""
    _, _, _, _, _, sh, sw, _ = _GEOMETRIES[geom]
    patches, shape = _stack(geom, torch.float64, seed=51)
    tangent, _ = _stack(geom, torch.float64, seed=53)
    fold = _fold(op, shape, (sh, sw))
    out, jvp_fn = torch.func.linearize(fold, patches)
    assert bit_equal(out, fold(patches))
    assert bit_equal(jvp_fn(tangent), fold(tangent))


@pytest.mark.parametrize("op", _OPS)
def test_the_forward_mode_jacobian_is_the_reverse_one(accel: str, op: str) -> None:
    """jacobian(strategy="forward-mode") was all zeros with the kernel."""
    # 1x6x6, patch 4, stride 2: a 36 x 64 Jacobian, small enough to materialise.
    gen = torch.Generator().manual_seed(61)
    patches = torch.rand(4, 1, 4, 4, generator=gen, dtype=torch.float64)
    fold = _fold(op, (1, 6, 6), (2, 2))
    fwd = torch.autograd.functional.jacobian(fold, patches, strategy="forward-mode", vectorize=True)
    rev = torch.autograd.functional.jacobian(fold, patches)
    # Under a window, forward mode divides by the denominator and reverse mode
    # multiplies by its reciprocal, so they agree to rounding, not to the bit.
    torch.testing.assert_close(fwd, rev, rtol=1e-12, atol=1e-15)
    assert fwd.abs().max() > 0


@pytest.mark.parametrize("op", _OPS)
def test_vmap_inside_a_dual_level_equals_a_loop(accel: str, op: str) -> None:
    """A torch.vmap tensor must reach no tangent check before its storage check.

    Inside a dual level, forward_ad.unpack_dual raises on a vmap tensor ("no
    batching rule for aten::_unpack_dual"). Checking the tangent first, as the
    simplest fix for the lost tangent would, broke this call, which 0.6.0
    answered correctly: the vmap tensor has no storage, so the kernel declines
    it before asking.
    """
    gen = torch.Generator().manual_seed(65)
    batch = torch.rand(2, 49, 3, 4, 4, generator=gen)
    fold = _fold(op, (3, 16, 16), (2, 2))
    with fwad.dual_level():
        got = torch.vmap(fold)(batch)
    assert bit_equal(got, torch.stack([fold(p) for p in batch]))


@pytest.mark.parametrize("op", ["reconstruct", "hann"])
def test_a_dual_keeps_its_tangent_under_compile(accel: str, op: str) -> None:
    """At torch 2.6, Dynamo reads a dual's tangent as None while it traces, so
    a tangent check traced by Dynamo let the dual through to the kernel; the
    eligibility checks therefore run outside the trace."""
    patches, shape = _stack("overlap", torch.float32, seed=81)
    tangent, _ = _stack("overlap", torch.float32, seed=83)
    fold = _fold(op, shape, (2, 2))
    torch._dynamo.reset()
    compiled = torch.compile(fold, backend="eager")
    with fwad.dual_level():
        got = fwad.unpack_dual(compiled(fwad.make_dual(patches, tangent))).tangent
        assert got is not None, "the tangent was dropped under torch.compile"
        got = got.clone()
    assert bit_equal(got, fold(tangent))


@real_accel
def test_fold_weighted_declines_a_dual(monkeypatch: pytest.MonkeyPatch) -> None:
    """The primitive hands a dual back to torch, in the patches or the kernel."""
    monkeypatch.setenv("PATCHCRAFT_ACCEL", "1")
    gen = torch.Generator().manual_seed(71)
    patches = torch.rand(49, 3, 4, 4, generator=gen)
    kernel = torch.rand(4, 4, generator=gen)
    assert _accel.fold_weighted(patches, (3, 16, 16), (2, 2), kernel) is not None
    with fwad.dual_level():
        dual = fwad.make_dual(patches, torch.ones_like(patches))
        assert _accel.fold_weighted(dual, (3, 16, 16), (2, 2), None) is None
        dual_kernel = fwad.make_dual(kernel, torch.ones_like(kernel))
        assert _accel.fold_weighted(patches, (3, 16, 16), (2, 2), dual_kernel) is None
        # A plain tensor inside an active level still takes the kernel.
        assert _accel.fold_weighted(patches, (3, 16, 16), (2, 2), None) is not None
