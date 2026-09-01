"""Bridge to the optional Rust accelerator (``patchcraft-accel``).

The accelerator is a separate PyPI package (a pyo3/Rust extension module).
This module detects it at runtime and exposes a single primitive,
:func:`fold_weighted`, for the overlap paths of ``reconstruct`` and
``stitch``. Any failure — package absent, ABI mismatch, ``PATCHCRAFT_ACCEL=0``
in the environment, or an ineligible tensor — yields ``None``/``False`` and
the caller falls back to the pure-torch path. Nothing here raises for an
unavailable accelerator.
"""
from __future__ import annotations

import importlib
import os
from types import ModuleType

import torch

__all__ = ["accel_available", "fold_weighted"]

_ABI_VERSION_REQUIRED = 1

_module: ModuleType | None = None
_checked = False


def _load() -> ModuleType | None:
    """Import ``patchcraft_accel`` once, cache the outcome, return it or None."""
    global _checked, _module
    if not _checked:
        _checked = True
        try:
            mod = importlib.import_module("patchcraft_accel")
        except ImportError:
            pass
        else:
            if getattr(mod, "_ABI_VERSION", None) == _ABI_VERSION_REQUIRED:
                _module = mod
    return _module


def _env_disabled() -> bool:
    """Evaluated per call so tests/debugging can toggle without cache resets."""
    return os.environ.get("PATCHCRAFT_ACCEL", "1") == "0"


def accel_available() -> bool:
    """Whether the overlap paths will use the native accelerator."""
    return not _env_disabled() and _load() is not None


def fold_weighted(
    patches: torch.Tensor,
    image_shape: tuple[int, int, int],
    stride: tuple[int, int],
    kernel: torch.Tensor | None,
) -> torch.Tensor | None:
    """Sum ``patches`` into a ``(C, H, W)`` image, optionally kernel-weighted.

    ``out[c, y, x] = sum_p patches[p, c, y - row_p, x - col_p] * kernel[...]``
    over the patches covering ``(y, x)`` — the numerator of the overlap fold,
    before the count/denominator division the caller performs. Summation per
    pixel is in descending patch index (= ascending kernel offset), matching
    ATen col2im's per-pixel order — the order that is bit-exact against
    ``F.fold``.

    Returns ``None`` when the accelerator is unavailable or the input is
    ineligible (non-CPU device, dtype outside float32/float64, mismatched
    kernel); the caller then uses its pure-torch path. Geometry is NOT
    validated here — callers validate with ``check_fold_geometry`` first.
    """
    mod = None if _env_disabled() else _load()
    if mod is None:
        return None
    if patches.device.type != "cpu":
        return None
    if patches.dtype not in (torch.float32, torch.float64):
        return None
    if patches.requires_grad:
        # The native kernel writes a leaf tensor through raw pointers — no
        # autograd graph. Fall back to the differentiable torch path.
        return None
    if kernel is not None and (
        kernel.dtype != patches.dtype or kernel.device != patches.device
    ):
        return None

    c, h, w = image_shape
    n, _c, ph, pw = patches.shape
    sh, sw = stride

    work = patches.contiguous()
    kern = kernel.contiguous() if kernel is not None else None
    out = torch.empty((c, h, w), dtype=work.dtype, device=work.device)
    if work.numel() == 0 or out.numel() == 0:
        return None

    mod.fold_add(
        work.data_ptr(),
        out.data_ptr(),
        n,
        c,
        ph,
        pw,
        h,
        w,
        sh,
        sw,
        kern.data_ptr() if kern is not None else None,
        "f32" if work.dtype == torch.float32 else "f64",
    )
    # `work` and `kern` must stay alive until the native call returns; it is
    # synchronous and both are still local references here, so this holds.
    return out
