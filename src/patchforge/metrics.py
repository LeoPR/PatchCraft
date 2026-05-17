"""Pixel-level metrics between patches (or between any same-shape tensors).

Three pure functions, no state, no allocation beyond the diff itself. Lives
in PatchForge because every consumer that uses ``extract`` + ``reconstruct``
or ``pair`` ends up reinventing the same MSE/PSNR per patch — bundling them
here saves consumers from inventing slightly-different reductions and gives
the test suite a stable comparison surface.

Out of scope: SSIM, MS-SSIM, LPIPS, perceptual losses. Those depend on
windowing schemes or pre-trained networks; pytorch-msssim and lpips are
mature standalone packages.

Contract: docs/THEORY.md §1.6 and §9.7.
"""
from __future__ import annotations

import math

import torch

__all__ = ["patch_metrics", "per_patch_mse", "per_patch_psnr"]


def _check_pair(a: torch.Tensor, b: torch.Tensor) -> None:
    if not isinstance(a, torch.Tensor):
        raise TypeError(f"a must be torch.Tensor, got {type(a).__name__}")
    if not isinstance(b, torch.Tensor):
        raise TypeError(f"b must be torch.Tensor, got {type(b).__name__}")
    if a.shape != b.shape:
        raise ValueError(
            f"shape mismatch: a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}"
        )
    if a.dtype != b.dtype:
        raise ValueError(
            f"dtype mismatch: a.dtype={a.dtype}, b.dtype={b.dtype}"
        )
    if a.device != b.device:
        raise ValueError(
            f"device mismatch: a.device={a.device}, b.device={b.device}"
        )


def _check_max_value(max_value: float) -> float:
    if not isinstance(max_value, (int, float)) or isinstance(max_value, bool):
        raise ValueError(
            f"max_value must be a positive number, got {max_value!r}"
        )
    if not math.isfinite(max_value) or max_value <= 0:
        raise ValueError(
            f"max_value must be finite and positive, got {max_value!r}"
        )
    return float(max_value)


def patch_metrics(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    max_value: float = 1.0,
) -> dict[str, float]:
    """Pixel-level metrics between two same-shape tensors.

    Computes over the full tensor (reduces every axis). Works on single
    patches ``(C, h, w)``, patch stacks ``(L, C, h, w)``, paired patches
    of either shape — anything as long as ``a.shape == b.shape``.

    Internal accumulation promotes to ``float64`` for stability, regardless
    of input dtype. Returns plain Python floats so the dict can be
    JSON-serialized.

    Parameters
    ----------
    a, b
        Same shape, same dtype, same device.
    max_value
        Dynamic range of the signal (``1.0`` for normalized
        ``float`` in ``[0, 1]``, ``255`` for byte-scaled). Used only for PSNR.

    Returns
    -------
    dict
        ``{"mae", "mse", "max_abs", "psnr_db"}``. ``psnr_db`` is
        ``+inf`` when ``a == b`` exactly.

    Raises
    ------
    TypeError, ValueError
        On non-tensor input, shape/dtype/device mismatch, or non-positive
        ``max_value``.
    """
    _check_pair(a, b)
    mv = _check_max_value(max_value)

    a64 = a.to(torch.float64) if a.dtype != torch.float64 else a
    b64 = b.to(torch.float64) if b.dtype != torch.float64 else b
    diff = a64 - b64
    abs_diff = diff.abs()
    mse = (diff * diff).mean().item()
    psnr_db = float("inf") if mse == 0.0 else 10.0 * math.log10(mv * mv / mse)
    return {
        "mae": abs_diff.mean().item(),
        "mse": mse,
        "max_abs": abs_diff.max().item(),
        "psnr_db": psnr_db,
    }


def per_patch_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return a ``(L,)`` tensor of MSE values, one per patch in ``(L, C, h, w)``.

    Useful for ranking patches by reconstruction error after a model pass
    or after a lossy round-trip. Reduction is over ``C, h, w``; the leading
    axis is preserved.

    Raises ``ValueError`` if either input is not 4-D or shapes differ.
    """
    _check_pair(a, b)
    if a.ndim != 4:
        raise ValueError(
            f"per_patch_mse expects 4-D tensors (L, C, h, w), got ndim={a.ndim}"
        )
    diff = a - b
    return (diff * diff).mean(dim=(1, 2, 3))


def per_patch_psnr(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    max_value: float = 1.0,
) -> torch.Tensor:
    """Return a ``(L,)`` tensor of PSNR values (dB), one per patch.

    Identical patches yield ``+inf`` (no clamp tricks — the result is
    mathematically infinite and the caller should treat it as such).

    Parameters
    ----------
    a, b
        ``(L, C, h, w)`` tensors with identical shape, dtype, device.
    max_value
        Signal dynamic range; see :func:`patch_metrics`.
    """
    mv = _check_max_value(max_value)
    mse = per_patch_mse(a, b)
    finfo = torch.finfo(mse.dtype) if mse.is_floating_point() else None
    tiny = finfo.tiny if finfo is not None else 1e-12
    mse_safe = mse.clamp_min(tiny)
    psnr = 10.0 * torch.log10((mv * mv) / mse_safe)
    inf = torch.full_like(mse, float("inf"))
    return torch.where(mse == 0, inf, psnr)
