"""Audited data helpers for round-trip tests (G1; docs/FOCO-1.0.md §4).

Three generators of false negatives are banned from round-trip assertions:

- **Integer ramps** (``torch.arange``): small integers are exactly
  representable, so sums and divisions land exactly where generic float data
  does not. Use :func:`rand_image` whenever the *value* matters; ramps remain
  valid where only order/position matters (row-major layout, all-ones count
  maps, coverage guards).
- **Widened float32** (``x.double()``): the low 29 mantissa bits come out
  zero, so the data round-trips where true float64 noise does not. Generate
  directly in the target dtype.
- **``torch.equal`` on NaN data**: not reflexive (NaN != NaN even with
  identical bits). :func:`bit_equal` compares integer views and is NaN-safe.

The exactness predicate (ADR 0003): the ``extract``/``reconstruct`` round
trip is bit-exact iff every value of the overlap count map is a power of
two. :func:`exact_axes_pow2` computes it with an O(H+W) per-axis closed
form; :func:`count_map_pow2` materializes the count map with integer
difference arrays, which is an independent code path, so cross-checking the two
means something. Both assume exact coverage; without it there is no
round-trip at all (``reconstruct`` raises).
"""
from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "bit_equal",
    "count_map_pow2",
    "coverage_counts",
    "exact_axes_pow2",
    "rand_image",
    "within_pixel_bound",
]

_INT_VIEW: dict[torch.dtype, torch.dtype] = {
    torch.float16: torch.int16,
    torch.float32: torch.int32,
    torch.float64: torch.int64,
}


def rand_image(c: int, h: int, w: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    """Uniform [0, 1) image generated directly in ``dtype``, never widened
    from a narrower dtype (that would leave half the mantissa zeroed)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.rand(c, h, w, dtype=dtype, generator=gen)


def bit_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bitwise equality via the integer view, which is NaN-safe unlike
    ``torch.equal`` (which is not reflexive on NaN)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool((a.view(_INT_VIEW[a.dtype]) == b.view(_INT_VIEW[b.dtype])).all())


def exact_axes_pow2(h: int, w: int, ph: int, pw: int, sh: int, sw: int) -> bool:
    """Closed-form predicate, O(H+W). On one axis (length ``n``, patch ``p``,
    stride ``s``, exact coverage) the number of patches covering pixel ``i``
    is ``hi - lo + 1`` with ``hi = min(i // s, (n - p) // s)`` and
    ``lo = max(0, ceil((i - p + 1) / s))``. True iff every distinct count on
    both axes is a power of two."""
    for n, p, s in ((h, ph, sh), (w, pw, sw)):
        starts = (n - p) // s
        counts: set[int] = set()
        for i in range(n):
            hi = min(i // s, starts)
            lo = max(0, -((-(i - p + 1)) // s))
            counts.add(hi - lo + 1)
        if any(c & (c - 1) != 0 for c in counts):
            return False
    return True


def coverage_counts(h: int, w: int, ph: int, pw: int, sh: int, sw: int) -> np.ndarray:
    """The ``(h, w)`` integer count map, built with per-axis difference
    arrays (place every patch interval, cumsum, outer product), with no closed
    form and no ``F.fold``, so it cross-checks :func:`exact_axes_pow2`
    independently."""
    axes: list[np.ndarray] = []
    for n, p, s in ((h, ph, sh), (w, pw, sw)):
        diff = np.zeros(n + 1, dtype=np.int64)
        starts = np.arange(0, n - p + 1, s)
        np.add.at(diff, starts, 1)
        np.add.at(diff, starts + p, -1)
        axes.append(np.cumsum(diff[:-1]))
    return np.outer(axes[0], axes[1])


def count_map_pow2(h: int, w: int, ph: int, pw: int, sh: int, sw: int) -> bool:
    """Predicate via the materialized count map: every value a power of two."""
    vals = np.unique(coverage_counts(h, w, ph, pw, sh, sw))
    return bool(((vals & (vals - 1)) == 0).all())


def within_pixel_bound(
    out: torch.Tensor, img: torch.Tensor, counts: np.ndarray
) -> bool:
    """Amendment A bound: ``|out - img| <= (k + 1) * eps * |img|`` per pixel,
    with ``k`` the pixel's coverage count. Finite data only (NaN -> False)."""
    k = torch.from_numpy(counts).to(img.dtype).unsqueeze(0).expand_as(img)
    eps = torch.finfo(img.dtype).eps
    err = (out - img).abs()
    bound = (k + 1.0) * eps * img.abs()
    return bool((err <= bound).all())
