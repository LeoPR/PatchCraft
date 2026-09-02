"""Self-tests for the audited helpers in tests/_rng.py (G1, FOCO §4)."""
from __future__ import annotations

import numpy as np
import torch

from tests._rng import (
    bit_equal,
    count_map_pow2,
    coverage_counts,
    exact_axes_pow2,
    rand_image,
    within_pixel_bound,
)


class TestRandImage:
    def test_seeded_reproducible(self) -> None:
        a = rand_image(1, 8, 8, torch.float32, seed=7)
        b = rand_image(1, 8, 8, torch.float32, seed=7)
        assert bit_equal(a, b)

    def test_different_seed_differs(self) -> None:
        a = rand_image(1, 8, 8, torch.float32, seed=7)
        b = rand_image(1, 8, 8, torch.float32, seed=8)
        assert not bit_equal(a, b)

    def test_generated_in_target_dtype_not_widened(self) -> None:
        """The banned shortcut: `.double()` of float32 leaves the low 29
        mantissa bits zero; true float64 noise does not (measured, FOCO §0)."""
        direct = rand_image(1, 16, 16, torch.float64, seed=3)
        widened = direct.float().double()
        assert not bit_equal(direct, widened)
        low29 = (1 << 29) - 1
        assert bool((widened.view(torch.int64) & low29 == 0).all())
        assert bool((direct.view(torch.int64) & low29 != 0).any())


class TestBitEqual:
    def test_nan_safe(self) -> None:
        img = rand_image(1, 4, 4, torch.float32, seed=1)
        img[0, 0, 0] = float("nan")
        clone = img.clone()
        assert not torch.equal(img, clone)  # NaN != NaN
        assert bit_equal(img, clone)  # bits are identical

    def test_one_ulp_differs(self) -> None:
        a = rand_image(1, 4, 4, torch.float32, seed=1)
        b = a.clone()
        b[0, 0, 0] = torch.nextafter(b[0, 0, 0], torch.tensor(2.0))
        assert not bit_equal(a, b)

    def test_dtype_mismatch(self) -> None:
        a = rand_image(1, 4, 4, torch.float32, seed=1)
        assert not bit_equal(a, a.double())


class TestPredicates:
    def test_stride_equals_patch_always_true(self) -> None:
        assert exact_axes_pow2(28, 28, 7, 7, 7, 7)
        assert count_map_pow2(28, 28, 7, 7, 7, 7)

    def test_half_stride_true(self) -> None:
        # counts are 1, 2, 4
        assert exact_axes_pow2(16, 16, 4, 4, 2, 2)
        assert count_map_pow2(16, 16, 4, 4, 2, 2)

    def test_foco_anchor_false(self) -> None:
        # FOCO anchor geometry (1, 4, 14) p=(4, 4) s=(1, 1): counts include 3
        assert not exact_axes_pow2(4, 14, 4, 4, 1, 1)
        assert not count_map_pow2(4, 14, 4, 4, 1, 1)

    def test_recipe_matches_count_map_on_fixed_grid(self) -> None:
        cases = [
            (28, 28, 7, 7, 7, 7),
            (16, 16, 4, 4, 2, 2),
            (4, 14, 4, 4, 1, 1),
            (13, 13, 4, 4, 3, 3),
            (12, 18, 4, 6, 2, 3),
            (9, 9, 4, 4, 1, 1),
            (24, 24, 9, 9, 3, 3),
            (10, 10, 4, 4, 2, 2),
            (7, 7, 3, 3, 2, 2),
        ]
        for g in cases:
            assert exact_axes_pow2(*g) == count_map_pow2(*g), g

    def test_coverage_counts_known_geometry(self) -> None:
        counts = coverage_counts(8, 8, 4, 4, 2, 2)
        assert counts[0, 0] == 1 and counts[7, 7] == 1
        assert counts[3, 3] == 4
        assert set(np.unique(counts).tolist()) == {1, 2, 4}


class TestWithinPixelBound:
    def test_exact_result_passes(self) -> None:
        img = rand_image(1, 9, 9, torch.float32, seed=5)
        counts = coverage_counts(9, 9, 4, 4, 1, 1)
        assert within_pixel_bound(img, img, counts)

    def test_detects_large_error(self) -> None:
        img = rand_image(1, 9, 9, torch.float32, seed=5)
        counts = coverage_counts(9, 9, 4, 4, 1, 1)
        assert not within_pixel_bound(img + 0.5, img, counts)

    def test_zero_image_bound_is_zero_and_exact(self) -> None:
        img = torch.zeros(1, 9, 9)
        counts = coverage_counts(9, 9, 4, 4, 1, 1)
        assert within_pixel_bound(img, img, counts)
