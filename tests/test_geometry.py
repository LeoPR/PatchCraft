"""Tests for `patchkit.num_patches` and `patchkit.tilings`.

Contract: docs/THEORY.md §1.5 and §9.6.
"""
from __future__ import annotations

import pytest
import torch

from patchkit import (
    TilingSpec,
    extract,
    num_patches,
    reconstruct,
    tilings,
)

# ---------------------------------------------------------------- num_patches --

class TestNumPatches:
    def test_matches_extract_count(self) -> None:
        """num_patches must equal extract(...).shape[0] when geometry fits."""
        img = torch.zeros(1, 20, 30)
        out = extract(img, patch_size=(4, 6), stride=(2, 3))
        nh, nw = num_patches((20, 30), patch_size=(4, 6), stride=(2, 3))
        assert nh * nw == out.shape[0]

    def test_accepts_chw_shape(self) -> None:
        nh_chw, nw_chw = num_patches((3, 16, 16), 4, 4)
        nh_hw, nw_hw = num_patches((16, 16), 4, 4)
        assert (nh_chw, nw_chw) == (nh_hw, nw_hw)

    def test_28x28_classic_geometries(self) -> None:
        """Spot-check classic MNIST patch geometries."""
        assert num_patches((28, 28), 7, 7) == (4, 4)
        assert num_patches((28, 28), 4, 4) == (7, 7)
        assert num_patches((28, 28), 4, 2) == (13, 13)
        assert num_patches((28, 28), 3, 1) == (26, 26)
        assert num_patches((28, 28), 28, 28) == (1, 1)

    def test_dilation_counted_correctly(self) -> None:
        # eff_h = 2 * (3 - 1) + 1 = 5; (16 - 5) // 1 + 1 = 12
        assert num_patches((16, 16), patch_size=3, stride=1, dilation=2) == (12, 12)

    def test_patch_larger_than_image_returns_zero(self) -> None:
        assert num_patches((4, 4), 8, 1) == (0, 0)
        assert num_patches((4, 16), patch_size=(8, 4), stride=1) == (0, 13)

    def test_rejects_bad_shape(self) -> None:
        with pytest.raises(ValueError, match="image_shape"):
            num_patches((28,), 4, 4)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="image_shape"):
            num_patches((1, 1, 28, 28), 4, 4)  # type: ignore[arg-type]

    def test_rejects_nonpositive_dims(self) -> None:
        with pytest.raises(ValueError, match="positive int"):
            num_patches((0, 28), 4, 4)
        with pytest.raises(ValueError, match="positive int"):
            num_patches((28, -4), 4, 4)


# -------------------------------------------------------------------- tilings --

class TestTilingsExact:
    def test_28x28_divisors(self) -> None:
        """28x28: divisors >= 2 are {2, 4, 7, 14, 28} -> 5 exact tilings."""
        specs = tilings((28, 28))
        ps = [t.patch_size[0] for t in specs]
        assert ps == [2, 4, 7, 14, 28]
        assert all(t.overlap is False for t in specs)
        assert all(t.dilation == (1, 1) for t in specs)
        assert all(t.patch_size == t.stride for t in specs)

    def test_28x28_total_counts(self) -> None:
        specs = {t.patch_size[0]: t.total_patches for t in tilings((28, 28))}
        assert specs == {2: 196, 4: 49, 7: 16, 14: 4, 28: 1}

    def test_includes_p_1_when_min_patch_size_is_1(self) -> None:
        specs = tilings((28, 28), min_patch_size=1)
        assert any(t.patch_size == (1, 1) for t in specs)
        assert specs[0].total_patches == 28 * 28

    def test_max_patch_size_caps_results(self) -> None:
        specs = tilings((28, 28), max_patch_size=10)
        assert [t.patch_size[0] for t in specs] == [2, 4, 7]

    def test_non_square_image(self) -> None:
        """20x30: ph must divide BOTH 20 and 30. gcd(20,30)=10; divisors of 10
        >= 2 that also divide 20 and 30 are {2, 5, 10}."""
        specs = tilings((20, 30))
        ps = [t.patch_size[0] for t in specs]
        assert ps == [2, 5, 10]

    def test_chw_accepted(self) -> None:
        specs_hw = tilings((28, 28))
        specs_chw = tilings((3, 28, 28))
        assert [t.patch_size for t in specs_hw] == [t.patch_size for t in specs_chw]


class TestTilingsOverlap:
    def test_28x28_includes_classic_overlap_specs(self) -> None:
        """All half-overlap (s = p/2) geometries on 28x28."""
        specs = tilings((28, 28), allow_overlap=True)
        overlap_specs = [t for t in specs if t.overlap]
        # patch=4 stride=2 is a classic; should appear
        assert any(t.patch_size == (4, 4) and t.stride == (2, 2) for t in overlap_specs)
        # patch=14 stride=7 (s = p/2)
        assert any(t.patch_size == (14, 14) and t.stride == (7, 7) for t in overlap_specs)

    def test_overlap_includes_clean_edge_only(self) -> None:
        """Every emitted overlap spec must have (H - p) % s == 0 on both axes."""
        for t in tilings((28, 28), allow_overlap=True):
            if t.overlap:
                p, s = t.patch_size[0], t.stride[0]
                assert (28 - p) % s == 0
                assert (28 - p) % s == 0  # W is also 28

    def test_no_overlap_subset_when_flag_false(self) -> None:
        without = tilings((28, 28), allow_overlap=False)
        with_overlap = tilings((28, 28), allow_overlap=True)
        without_keys = {(t.patch_size, t.stride) for t in without}
        with_keys = {(t.patch_size, t.stride) for t in with_overlap}
        assert without_keys.issubset(with_keys)


class TestTilingsRejects:
    @pytest.mark.parametrize("bad", [(28,), (28, 28, 28, 28), 28, [28, 28]])
    def test_bad_shape(self, bad: object) -> None:
        with pytest.raises(ValueError, match="image_shape"):
            tilings(bad)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [0, -1])
    def test_nonpositive_min(self, bad: int) -> None:
        with pytest.raises(ValueError, match="min_patch_size"):
            tilings((28, 28), min_patch_size=bad)

    def test_nonpositive_max(self) -> None:
        with pytest.raises(ValueError, match="max_patch_size"):
            tilings((28, 28), max_patch_size=0)

    def test_min_greater_than_max(self) -> None:
        with pytest.raises(ValueError, match=r"min_patch_size.*max_patch_size"):
            tilings((28, 28), min_patch_size=10, max_patch_size=5)


class TestTilingsRoundtripGuarantee:
    """Every spec from tilings() must produce bit-exact extract+reconstruct."""

    def test_all_exact_tilings_28x28_roundtrip(self) -> None:
        img = torch.arange(28 * 28, dtype=torch.float64).reshape(1, 28, 28)
        for spec in tilings((28, 28)):
            patches = extract(img, patch_size=spec.patch_size, stride=spec.stride)
            assert patches.shape[0] == spec.total_patches
            recon = reconstruct(patches, image_shape=img.shape, stride=spec.stride)
            assert torch.equal(recon, img), (
                f"spec {spec} broke bit-exact round-trip"
            )

    def test_overlap_tilings_28x28_close_roundtrip(self) -> None:
        """Overlap geometries: weighted reconstruction within float64 tolerance."""
        img = torch.arange(28 * 28, dtype=torch.float64).reshape(1, 28, 28)
        for spec in tilings((28, 28), allow_overlap=True):
            patches = extract(img, patch_size=spec.patch_size, stride=spec.stride)
            recon = reconstruct(patches, image_shape=img.shape, stride=spec.stride)
            assert torch.allclose(recon, img, rtol=1e-12, atol=1e-12), (
                f"spec {spec} broke overlap round-trip"
            )


class TestTilingSpecShape:
    def test_is_namedtuple(self) -> None:
        spec = tilings((28, 28))[0]
        assert isinstance(spec, tuple)
        assert isinstance(spec, TilingSpec)
        assert spec._fields == (
            "patch_size", "stride", "dilation",
            "num_patches", "total_patches", "overlap",
        )

    def test_iterable_and_indexable(self) -> None:
        spec = tilings((28, 28))[0]
        # NamedTuple destructures positionally just like a regular tuple
        p, s, d, n, total, overlap = spec
        assert p == spec.patch_size
        assert s == spec.stride
        assert d == spec.dilation
        assert n == spec.num_patches
        assert total == spec.total_patches
        assert overlap == spec.overlap
