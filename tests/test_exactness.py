"""Falsification suite for the bit-exactness predicate (FOCO §5, ADR 0003).

Predicate: the ``extract``/``reconstruct`` round trip is bit-exact iff every
value of the overlap count map is a power of two. Outside it, the per-pixel
error is bounded by ``(k+1)*eps*|v|`` with ``k`` the pixel's coverage count
(Amendment A of docs/superpowers/specs/2026-09-01-fase3-g1-predicado-design.md
(the frozen "1 ULP" wording was measured false).

Strategy: enumerate the legal geometry space *independently* of the
predicate (H, W in 4..24; ph, pw in 2..9; strides with exact coverage,
126,736 geometries), draw a seeded 256-geometry sample (67 inside the
predicate, 189 outside), and try to break both halves:

- inside  -> bit-exact for every seed of a fixed set;
- outside -> at least one of 50 seeds comes back inexact (exactness outside
  the predicate is a property of the data, not of the geometry: measured
  63/300 exact seeds in float32 at the FOCO anchor geometry, so a single
  execution proves nothing) AND every seed stays within the pixel bound.

``PATCHCRAFT_SWEEP_FULL=1`` replaces the sample with the full space, and the
local gate before merging (recipe vs count map over all 126,736, plus one
bit-exact round trip per inside-predicate geometry; ~1-2 min).
"""
from __future__ import annotations

import os
import random

import pytest
import torch

from patchcraft import extract, reconstruct
from tests._rng import (
    bit_equal,
    count_map_pow2,
    coverage_counts,
    exact_axes_pow2,
    rand_image,
    within_pixel_bound,
)

Geometry = tuple[int, int, int, int, int, int]  # (h, w, ph, pw, sh, sw)

_SAMPLE_SIZE = 256
_SAMPLE_SEED = 20260901
_POSITIVE_SEEDS = range(5)
_NEGATIVE_SEEDS = range(50)
_DTYPES = [torch.float32, torch.float64]


def _legal_geometries() -> list[Geometry]:
    """Every (h, w, ph, pw, sh, sw) with exact coverage: H, W in 4..24,
    ph, pw in 2..9 (and <= the axis), 1 <= s <= p with (n - p) % s == 0 on
    each axis independently (rectangular included). No hand-picked lists."""
    out: list[Geometry] = []
    for h in range(4, 25):
        for w in range(4, 25):
            for ph in range(2, min(9, h) + 1):
                for pw in range(2, min(9, w) + 1):
                    for sh in range(1, ph + 1):
                        if (h - ph) % sh != 0:
                            continue
                        for sw in range(1, pw + 1):
                            if (w - pw) % sw == 0:
                                out.append((h, w, ph, pw, sh, sw))
    return out


_SPACE = _legal_geometries()
_SAMPLE = random.Random(_SAMPLE_SEED).sample(_SPACE, _SAMPLE_SIZE)
_POSITIVE = [g for g in _SAMPLE if count_map_pow2(*g)]
_NEGATIVE = [g for g in _SAMPLE if not count_map_pow2(*g)]


def _roundtrip(
    g: Geometry, dtype: torch.dtype, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    h, w, ph, pw, sh, sw = g
    img = rand_image(1, h, w, dtype, seed)
    out = reconstruct(
        extract(img, patch_size=(ph, pw), stride=(sh, sw)),
        image_shape=img.shape,
        stride=(sh, sw),
    )
    return out, img


class TestSampleShape:
    def test_space_size(self) -> None:
        assert len(_SPACE) == 126_736

    def test_sample_has_both_halves(self) -> None:
        # Seeded, so stable: 67 inside the predicate, 189 outside.
        assert (len(_POSITIVE), len(_NEGATIVE)) == (67, 189)


class TestRecipeMatchesPredicate:
    @pytest.mark.parametrize("g", _SAMPLE, ids=repr)
    def test_axes_recipe_equals_count_map(self, g: Geometry) -> None:
        assert exact_axes_pow2(*g) == count_map_pow2(*g)


class TestPositiveHalf:
    @pytest.mark.parametrize("g", _POSITIVE, ids=repr)
    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_bit_exact_every_seed(self, g: Geometry, dtype: torch.dtype) -> None:
        for seed in _POSITIVE_SEEDS:
            out, img = _roundtrip(g, dtype, seed)
            assert bit_equal(out, img), (g, dtype, seed)


class TestNegativeHalf:
    @pytest.mark.parametrize("g", _NEGATIVE, ids=repr)
    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_some_seed_inexact_and_all_within_bound(
        self, g: Geometry, dtype: torch.dtype
    ) -> None:
        h, w, ph, pw, sh, sw = g
        counts = coverage_counts(h, w, ph, pw, sh, sw)
        inexact = 0
        for seed in _NEGATIVE_SEEDS:
            out, img = _roundtrip(g, dtype, seed)
            assert within_pixel_bound(out, img, counts), (g, dtype, seed)
            if not bit_equal(out, img):
                inexact += 1
        assert inexact >= 1, (
            f"{g} {dtype}: exact on all 50 seeds, so either the predicate grew "
            "or the fixed sample got lucky; investigate before touching this"
        )


class TestNanInsidePredicate:
    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_nan_roundtrips_bit_exact(self, dtype: torch.dtype) -> None:
        """D5 in code: inside the predicate the bits come back even when the
        value is NaN, where torch.equal is not reflexive."""
        img = rand_image(1, 16, 16, dtype, seed=99)
        img[0, 3, 5] = float("nan")
        img[0, 10, 11] = float("nan")
        out = reconstruct(
            extract(img, patch_size=(4, 4), stride=(2, 2)),
            image_shape=img.shape,
            stride=(2, 2),
        )
        assert not torch.equal(out, img)  # NaN != NaN
        assert bit_equal(out, img)


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("PATCHCRAFT_SWEEP_FULL") != "1",
    reason="full 126,736-geometry sweep is a local gate: PATCHCRAFT_SWEEP_FULL=1",
)
class TestFullSweep:
    def test_recipe_matches_count_map_everywhere(self) -> None:
        for g in _SPACE:
            assert exact_axes_pow2(*g) == count_map_pow2(*g), g

    def test_no_counterexample_inside_predicate(self) -> None:
        for g in _SPACE:
            if not count_map_pow2(*g):
                continue
            out, img = _roundtrip(g, torch.float32, seed=1)
            assert bit_equal(out, img), f"counterexample: {g}"
