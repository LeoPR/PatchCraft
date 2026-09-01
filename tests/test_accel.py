"""Tests for the optional native accelerator bridge (`patchcraft._accel`).

Two layers: import/ABI/env behavior with fake modules (always runs), and
numerical checks against `F.fold` (require the real `patchcraft-accel`
installed — `maturin develop` — and are skipped otherwise).
"""
from __future__ import annotations

import sys
import types

import pytest
import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)

from patchcraft import _accel

real_accel = pytest.mark.skipif(
    not _accel.accel_available(), reason="patchcraft-accel not installed"
)


@pytest.fixture
def fresh_accel(monkeypatch: pytest.MonkeyPatch):
    """Reset the import-once cache so a test fully controls detection."""
    monkeypatch.setattr(_accel, "_module", None)
    monkeypatch.setattr(_accel, "_checked", False)
    return _accel


def test_accel_available_returns_bool() -> None:
    assert isinstance(_accel.accel_available(), bool)


def test_absent_package_means_unavailable(fresh_accel, monkeypatch: pytest.MonkeyPatch) -> None:
    # `None` in sys.modules makes importlib raise ImportError.
    monkeypatch.setitem(sys.modules, "patchcraft_accel", None)
    assert _accel.accel_available() is False


def test_wrong_abi_version_means_unavailable(
    fresh_accel, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = types.SimpleNamespace(_ABI_VERSION=2)
    monkeypatch.setitem(sys.modules, "patchcraft_accel", fake)
    assert _accel.accel_available() is False


def test_matching_abi_version_means_available(
    fresh_accel, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = types.SimpleNamespace(_ABI_VERSION=1)
    monkeypatch.setitem(sys.modules, "patchcraft_accel", fake)
    assert _accel.accel_available() is True


def test_env_override_disables(fresh_accel, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.SimpleNamespace(_ABI_VERSION=1)
    monkeypatch.setitem(sys.modules, "patchcraft_accel", fake)
    monkeypatch.setenv("PATCHCRAFT_ACCEL", "0")
    assert _accel.accel_available() is False


def test_fold_weighted_none_when_unavailable(
    fresh_accel, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "patchcraft_accel", None)
    patches = torch.rand(49, 3, 4, 4)
    assert _accel.fold_weighted(patches, (3, 16, 16), (2, 2), None) is None


def test_fold_weighted_none_when_env_disabled(
    fresh_accel, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = types.SimpleNamespace(_ABI_VERSION=1, fold_add=lambda *a: None)
    monkeypatch.setitem(sys.modules, "patchcraft_accel", fake)
    monkeypatch.setenv("PATCHCRAFT_ACCEL", "0")
    patches = torch.rand(49, 3, 4, 4)
    assert _accel.fold_weighted(patches, (3, 16, 16), (2, 2), None) is None


def test_fold_weighted_rejects_ineligible_dtype(
    fresh_accel, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = types.SimpleNamespace(_ABI_VERSION=1, fold_add=lambda *a: None)
    monkeypatch.setitem(sys.modules, "patchcraft_accel", fake)
    patches = torch.rand(49, 3, 4, 4).half()
    assert _accel.fold_weighted(patches, (3, 16, 16), (2, 2), None) is None


@real_accel
def test_fold_weighted_matches_manual_fold() -> None:
    """The native numerator equals the F.fold composition, both kernel modes."""
    torch.manual_seed(0)
    patches = torch.rand(49, 3, 4, 4)  # (3,16,16), p=4, s=2 -> grid 7x7
    flat = patches.permute(1, 2, 3, 0).reshape(3 * 4 * 4, 49).unsqueeze(0)
    ref = F.fold(flat, (16, 16), (4, 4), stride=(2, 2))[0]

    out = _accel.fold_weighted(patches, (3, 16, 16), (2, 2), None)
    assert out is not None
    assert torch.equal(out, ref)

    kernel = torch.rand(4, 4)
    flat_w = (patches * kernel).permute(1, 2, 3, 0).reshape(3 * 4 * 4, 49).unsqueeze(0)
    ref_w = F.fold(flat_w, (16, 16), (4, 4), stride=(2, 2))[0]
    out_w = _accel.fold_weighted(patches, (3, 16, 16), (2, 2), kernel)
    assert out_w is not None
    assert torch.equal(out_w, ref_w)
