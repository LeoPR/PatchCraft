"""Naive reference implementation of extract+reconstruct (FOCO §2, D6).

The substitute for the lost `hand.py` x `pc.py` consumer gate: pure Python
loops, no `F.fold`, no code shared with the library. If the fast paths and
this reference agree bit for bit inside the predicate, and both stay within
the per-pixel bound outside it, so the arithmetic and not just the API is right.

Runs in both accel modes on purpose: `reconstruct` dispatches internally, so
an active accelerator is exercised here without any extra code (the full
accel x pure equivalence grid lives in tests/test_accel.py).
"""
from __future__ import annotations

import pytest
import torch

from patchcraft import extract, reconstruct
from tests._rng import (
    bit_equal,
    count_map_pow2,
    coverage_counts,
    rand_image,
    within_pixel_bound,
)

Geometry = tuple[int, int, int, int, int, int]  # (h, w, ph, pw, sh, sw)

_GRID: list[Geometry] = [
    (8, 8, 4, 4, 4, 4),     # exact tile
    (16, 16, 4, 4, 2, 2),   # pow2 overlap (counts 1, 2, 4)
    (9, 9, 3, 3, 3, 3),     # exact tile
    (13, 13, 4, 4, 3, 3),   # pow2 overlap, counts {1, 2}
    (10, 10, 4, 4, 2, 2),   # pow2 overlap
    (12, 18, 4, 6, 2, 3),   # pow2 rectangular
    (7, 7, 3, 3, 2, 2),     # pow2, counts {1, 2}
    (14, 14, 4, 4, 1, 1),   # FOCO anchor: outside (counts include 3)
    (9, 9, 4, 4, 1, 1),     # outside: counts {1, 2, 3, 4}
    (24, 24, 9, 9, 3, 3),   # outside: counts include 3
]


def _ref_extract(
    img: torch.Tensor, ph: int, pw: int, sh: int, sw: int
) -> torch.Tensor:
    """Slice every patch out pixel region by pixel region; stack row-major."""
    _, h, w = img.shape
    patches = [
        img[:, y : y + ph, x : x + pw].clone()
        for y in range(0, h - ph + 1, sh)
        for x in range(0, w - pw + 1, sw)
    ]
    return torch.stack(patches)


def _ref_reconstruct(
    patches: torch.Tensor, c: int, h: int, w: int, ph: int, pw: int,
    sh: int, sw: int,
) -> torch.Tensor:
    """Accumulate sum and count per pixel in the input dtype, then divide.
    Row-major ascending patch order, one slice-add per patch."""
    acc = torch.zeros(c, h, w, dtype=patches.dtype)
    cnt = torch.zeros(c, h, w, dtype=patches.dtype)
    k = 0
    for y in range(0, h - ph + 1, sh):
        for x in range(0, w - pw + 1, sw):
            acc[:, y : y + ph, x : x + pw] += patches[k]
            cnt[:, y : y + ph, x : x + pw] += 1
            k += 1
    return acc / cnt


@pytest.mark.parametrize("g", _GRID, ids=repr)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_reference_matches_library(g: Geometry, dtype: torch.dtype) -> None:
    h, w, ph, pw, sh, sw = g
    img = rand_image(3, h, w, dtype, seed=17)

    ref_patches = _ref_extract(img, ph, pw, sh, sw)
    lib_patches = extract(img, patch_size=(ph, pw), stride=(sh, sw))
    # extract is a pure gather: bits must survive identically.
    assert bit_equal(lib_patches, ref_patches)

    ref = _ref_reconstruct(ref_patches, 3, h, w, ph, pw, sh, sw)
    lib = reconstruct(lib_patches, image_shape=img.shape, stride=(sh, sw))
    if count_map_pow2(*g):
        # Any summation order of k identical values with k a power of two is
        # exact, so the two implementations agree bit for bit.
        assert bit_equal(lib, ref)
    else:
        counts = coverage_counts(h, w, ph, pw, sh, sw)
        assert within_pixel_bound(lib, img, counts)
        assert within_pixel_bound(ref, img, counts)
