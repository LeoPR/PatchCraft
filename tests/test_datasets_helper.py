"""Tests for the label_subset helper in tests/_datasets.py.

mnist_subset is not unit-tested here (it downloads ~12 MB on first call;
exercised by lab/ scripts and any future slow-marked integration test).
"""
from __future__ import annotations

import pytest

from tests._datasets import label_subset


class TestLabelSubsetAccepts:
    def test_balanced_pick(self) -> None:
        labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        picks = label_subset(labels, n_per_label=2, seed=0)
        assert len(picks) == 6
        picked_labels = sorted(labels[i] for i in picks)
        assert picked_labels == [0, 0, 1, 1, 2, 2]

    def test_takes_all_when_class_underfilled(self) -> None:
        labels = [0, 0, 1]  # class 1 only has 1 sample
        picks = label_subset(labels, n_per_label=5, seed=0)
        assert sorted(labels[i] for i in picks) == [0, 0, 1]

    def test_deterministic_under_same_seed(self) -> None:
        labels = list(range(100)) * 2  # 100 classes of 2 each
        picks_a = label_subset(labels, n_per_label=1, seed=42)
        picks_b = label_subset(labels, n_per_label=1, seed=42)
        assert picks_a == picks_b

    def test_different_seeds_change_picks(self) -> None:
        labels = [0] * 10
        a = label_subset(labels, n_per_label=3, seed=1)
        b = label_subset(labels, n_per_label=3, seed=2)
        assert a != b

    def test_empty_labels(self) -> None:
        assert label_subset([], n_per_label=5, seed=0) == []

    def test_picks_are_valid_indices(self) -> None:
        labels = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
        picks = label_subset(labels, n_per_label=2, seed=0)
        assert all(0 <= i < len(labels) for i in picks)
        assert len(set(picks)) == len(picks)  # no duplicates


class TestLabelSubsetRejects:
    @pytest.mark.parametrize("bad", [0, -1, -100])
    def test_n_per_label_nonpositive(self, bad: int) -> None:
        with pytest.raises(ValueError, match="n_per_label must be a positive int"):
            label_subset([0, 1, 2], n_per_label=bad, seed=0)

    def test_n_per_label_float(self) -> None:
        with pytest.raises(ValueError, match="n_per_label must be"):
            label_subset([0, 1, 2], n_per_label=1.5, seed=0)  # type: ignore[arg-type]

    def test_n_per_label_bool_rejected(self) -> None:
        with pytest.raises(ValueError):
            label_subset([0, 1, 2], n_per_label=True, seed=0)  # type: ignore[arg-type]

    def test_seed_not_int(self) -> None:
        with pytest.raises(ValueError, match="seed must be int"):
            label_subset([0, 1, 2], n_per_label=1, seed="zero")  # type: ignore[arg-type]
