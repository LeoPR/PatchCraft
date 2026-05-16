"""Patch extraction via torch.nn.functional.unfold.

Contract: docs/THEORY.md §1 and §10.1, docs/ADR/0001-patch-extraction-api.md.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)

__all__ = ["extract"]


def _as_pair(value: int | tuple[int, int], name: str) -> tuple[int, int]:
    if isinstance(value, int) and not isinstance(value, bool):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        return (value, value)
    if isinstance(value, tuple) and len(value) == 2:
        h, w = value
        h_ok = isinstance(h, int) and not isinstance(h, bool)
        w_ok = isinstance(w, int) and not isinstance(w, bool)
        if not (h_ok and w_ok):
            raise ValueError(f"{name} must contain ints, got {value!r}")
        if h <= 0 or w <= 0:
            raise ValueError(f"{name} must be positive, got {value!r}")
        return (h, w)
    raise ValueError(f"{name} must be int or (int, int), got {value!r}")


def extract(
    image: torch.Tensor,
    patch_size: int | tuple[int, int],
    stride: int | tuple[int, int],
    dilation: int | tuple[int, int] = 1,
) -> torch.Tensor:
    """Extract rectangular patches from a `(C, H, W)` image.

    Returns `Tensor[L, C, ph, pw]` in row-major order. Patch `k` has its
    top-left at `(k // num_w * sh, k % num_w * sw)`. Truncation is the only
    boundary policy: if the geometry fits no patch, returns `Tensor[0, C, ph, pw]`.
    Dtype and device of `image` are preserved.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"image must be torch.Tensor, got {type(image).__name__}")
    if image.ndim != 3:
        raise ValueError(f"image must have ndim==3 (C, H, W), got ndim={image.ndim}")

    ph, pw = _as_pair(patch_size, "patch_size")
    sh, sw = _as_pair(stride, "stride")
    dh, dw = _as_pair(dilation, "dilation")

    c, h, w = image.shape
    eff_h = dh * (ph - 1) + 1
    eff_w = dw * (pw - 1) + 1

    if h < eff_h or w < eff_w:
        return torch.empty(0, c, ph, pw, dtype=image.dtype, device=image.device)

    unfolded = F.unfold(
        image.unsqueeze(0),
        kernel_size=(ph, pw),
        dilation=(dh, dw),
        stride=(sh, sw),
    )
    return unfolded[0].view(c, ph, pw, -1).permute(3, 0, 1, 2).contiguous()
