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

from patchcraft import _accel, reconstruct, stitch

real_accel = pytest.mark.skipif(
    not _accel.accel_available(), reason="patchcraft-accel not installed"
)


@pytest.fixture
def fresh_accel(monkeypatch: pytest.MonkeyPatch):
    """Reset the import-once cache so a test fully controls detection."""
    monkeypatch.setattr(_accel, "_module", None)
    monkeypatch.setattr(_accel, "_checked", False)
    # Neutralize the ambient override so detection tests are mode-independent;
    # tests that exercise the override set it explicitly afterwards.
    monkeypatch.delenv("PATCHCRAFT_ACCEL", raising=False)
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


# --- Equivalence through the public API, both modes -------------------------

_OVERLAP_GEOMETRIES = [
    # (ph, pw, sh, sw, h, w) — all with exact coverage
    (4, 4, 2, 2, 16, 16),  # square, half overlap
    (5, 4, 3, 2, 14, 14),  # stride does not divide patch, both axes
    (4, 6, 3, 5, 16, 21),  # rectangular everything
    (32, 32, 16, 16, 96, 96),  # realistic scale
]


def _patches_for(
    c: int, geom: tuple[int, int, int, int, int, int], dtype: torch.dtype
) -> torch.Tensor:
    from patchcraft import extract

    ph, pw, sh, sw, h, w = geom
    image = torch.rand(c, h, w, dtype=dtype)
    return extract(image, (ph, pw), (sh, sw))


@real_accel
@pytest.mark.parametrize("geom", _OVERLAP_GEOMETRIES)
@pytest.mark.parametrize("c", [1, 3, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_reconstruct_accel_matches_pure(
    geom: tuple[int, int, int, int, int, int],
    c: int,
    dtype: torch.dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ph, _pw, sh, sw, h, w = geom
    torch.manual_seed(0)
    patches = _patches_for(c, geom, dtype)
    monkeypatch.setenv("PATCHCRAFT_ACCEL", "0")
    pure = reconstruct(patches, (c, h, w), (sh, sw))
    monkeypatch.setenv("PATCHCRAFT_ACCEL", "1")
    fast = reconstruct(patches, (c, h, w), (sh, sw))
    assert fast.dtype == pure.dtype
    assert torch.equal(pure, fast)


@real_accel
@pytest.mark.parametrize("weight", ["uniform", "hann", "gaussian"])
@pytest.mark.parametrize("geom", _OVERLAP_GEOMETRIES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_stitch_accel_matches_pure(
    weight: str,
    geom: tuple[int, int, int, int, int, int],
    dtype: torch.dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ph, _pw, sh, sw, h, w = geom
    torch.manual_seed(0)
    patches = _patches_for(3, geom, dtype)
    monkeypatch.setenv("PATCHCRAFT_ACCEL", "0")
    pure = stitch(patches, (3, h, w), (sh, sw), weight=weight)  # type: ignore[arg-type]
    monkeypatch.setenv("PATCHCRAFT_ACCEL", "1")
    fast = stitch(patches, (3, h, w), (sh, sw), weight=weight)  # type: ignore[arg-type]
    assert fast.dtype == pure.dtype
    assert torch.equal(pure, fast)


@real_accel
def test_accel_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two accelerated runs are bit-identical regardless of thread scheduling."""
    monkeypatch.setenv("PATCHCRAFT_ACCEL", "1")
    torch.manual_seed(0)
    patches = torch.rand(961, 3, 32, 32)
    a = reconstruct(patches, (3, 512, 512), 16)
    b = reconstruct(patches, (3, 512, 512), 16)
    assert torch.equal(a, b)


@real_accel
def test_reconstruct_noncontiguous_matches_pure(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(0)
    patches = torch.rand(49, 3, 4, 8)[..., :4]  # non-contiguous (49,3,4,4)
    assert not patches.is_contiguous()
    monkeypatch.setenv("PATCHCRAFT_ACCEL", "0")
    pure = reconstruct(patches, (3, 16, 16), 2)
    monkeypatch.setenv("PATCHCRAFT_ACCEL", "1")
    fast = reconstruct(patches, (3, 16, 16), 2)
    assert torch.equal(pure, fast)
