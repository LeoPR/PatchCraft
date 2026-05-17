"""Stitch patches back into an image with configurable weighting kernels.

Where :func:`patchforge.reconstruct` is a bit-exact inverse of ``extract``,
``stitch`` is intended for *modified* patches — patches that have been
denoised, super-resolved, or otherwise altered — where overlap seams are
visible if patches are averaged uniformly. Weighting by a window kernel
(Hann, Gaussian) emphasizes patch centers and reduces those seams.

With ``weight="uniform"``, ``stitch`` is mathematically equivalent to
``reconstruct`` (down to floating-point ordering).

Contract: docs/THEORY.md §2.5 and §9.9.
"""
from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)

from patchforge.extract import _as_pair

__all__ = ["stitch"]


WeightKind = Literal["uniform", "hann", "gaussian"]
_WEIGHT_KINDS: tuple[WeightKind, ...] = ("uniform", "hann", "gaussian")


def _hann_1d(n: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Symmetric Hann window in ``[0, 1]``. ``n == 1`` is degenerate → ``[1.0]``."""
    if n == 1:
        return torch.ones(1, dtype=dtype, device=device)
    i = torch.arange(n, dtype=dtype, device=device)
    return 0.5 * (1.0 - torch.cos(2.0 * math.pi * i / (n - 1)))


def _gaussian_1d(n: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Gaussian centered at ``(n-1)/2`` with ``sigma = max(1, n/4)``."""
    if n == 1:
        return torch.ones(1, dtype=dtype, device=device)
    sigma = max(1.0, n / 4.0)
    center = (n - 1) / 2.0
    i = torch.arange(n, dtype=dtype, device=device)
    return torch.exp(-((i - center) ** 2) / (2.0 * sigma * sigma))


def _window_kernel(
    kind: WeightKind,
    ph: int,
    pw: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build a ``(ph, pw)`` window as the outer product of two 1-D windows."""
    if kind == "uniform":
        return torch.ones(ph, pw, dtype=dtype, device=device)
    if kind == "hann":
        wh = _hann_1d(ph, dtype, device)
        ww = _hann_1d(pw, dtype, device)
        return wh.unsqueeze(1) * ww.unsqueeze(0)
    if kind == "gaussian":
        wh = _gaussian_1d(ph, dtype, device)
        ww = _gaussian_1d(pw, dtype, device)
        return wh.unsqueeze(1) * ww.unsqueeze(0)
    raise ValueError(
        f"weight must be one of {_WEIGHT_KINDS!r}, got {kind!r}"
    )


def stitch(
    patches: torch.Tensor,
    image_shape: tuple[int, int, int],
    stride: int | tuple[int, int],
    *,
    weight: WeightKind = "uniform",
    dilation: int | tuple[int, int] = 1,
) -> torch.Tensor:
    """Reassemble a ``(C, H, W)`` image from ``(L, C, ph, pw)`` with blendable weights.

    Use ``stitch`` when patches have been modified (model output, denoised,
    super-resolved). Use :func:`patchforge.reconstruct` when patches came
    straight from ``extract`` and you want a bit-exact inverse with no
    extra arithmetic.

    ``weight`` controls how overlapping patches are blended:

    - ``"uniform"`` — each covering patch contributes equally. Mathematically
      equivalent to ``reconstruct`` (no seam attenuation).
    - ``"hann"`` — Hann window: full weight at patch center, zero at patch
      edges. Strong seam suppression. **Caveat:** image-corner pixels that
      are covered only by patches whose edge-weight at that location is zero
      will be zero in the output. Document this for callers.
    - ``"gaussian"`` — Gaussian centered on the patch with
      ``sigma = max(1.0, min(ph, pw) / 4)``. Smooth seam suppression without
      the strict zero at the edge (no corner-zero artifact).

    Internally: each patch is multiplied by the 2-D weight kernel, the
    weighted patches are folded into the numerator, the weight kernel itself
    is folded over the same geometry into the denominator, and
    ``numerator / denominator.clamp(min=1e-6)`` gives the output. The clamp
    absorbs float noise on covered pixels; geometry validation guarantees
    no uncovered pixels.

    Rejects (per §9.9): ``dilation != 1``; ``stride > patch_size`` in any
    axis; ``patches.ndim != 4``; non-floating-point patches (kernel
    multiplication breaks integer semantics for non-uniform weights —
    callers convert to ``float`` first); ``image_shape`` inconsistent with
    the patch grid; unknown ``weight``.

    Dtype and device of ``patches`` are preserved.
    """
    if not isinstance(patches, torch.Tensor):
        raise TypeError(
            f"patches must be torch.Tensor, got {type(patches).__name__}"
        )
    if patches.ndim != 4:
        raise ValueError(
            f"patches must have ndim==4 (L, C, ph, pw), got ndim={patches.ndim}"
        )
    if not patches.is_floating_point():
        raise ValueError(
            f"stitch requires floating-point patches, got dtype={patches.dtype}. "
            "Convert with patches.float() — weight kernels are float-valued and "
            "integer semantics would silently quantize the result."
        )

    if weight not in _WEIGHT_KINDS:
        raise ValueError(
            f"weight must be one of {_WEIGHT_KINDS!r}, got {weight!r}"
        )

    n_patches, c, ph, pw = patches.shape

    if not (isinstance(image_shape, tuple) and len(image_shape) == 3):
        raise ValueError(
            f"image_shape must be a 3-tuple (C, H, W), got {image_shape!r}"
        )
    for axis_name, val in zip(("C", "H", "W"), image_shape, strict=True):
        if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
            raise ValueError(
                f"image_shape[{axis_name}] must be a positive int, got {val!r}"
            )
    target_c, h, w = image_shape
    if target_c != c:
        raise ValueError(
            f"image_shape channels={target_c} does not match patches channel count {c}"
        )

    sh, sw = _as_pair(stride, "stride")
    dh, dw = _as_pair(dilation, "dilation")

    if dh != 1 or dw != 1:
        raise ValueError(
            f"stitch requires dilation==1, got dilation=({dh}, {dw}). "
            "Patches extracted with dilation > 1 cannot round-trip — consume them as features."
        )
    if sh > ph or sw > pw:
        raise ValueError(
            f"stitch forbids stride > patch_size (partial coverage forbidden), "
            f"got stride=({sh}, {sw}) and patch_size=({ph}, {pw})."
        )

    num_h = (h - ph) // sh + 1
    num_w = (w - pw) // sw + 1
    if num_h <= 0 or num_w <= 0:
        raise ValueError(
            f"image_shape={image_shape} too small for patch_size=({ph}, {pw}) "
            f"and stride=({sh}, {sw})"
        )
    expected_n_patches = num_h * num_w
    if n_patches != expected_n_patches:
        raise ValueError(
            f"patches.shape[0]={n_patches} inconsistent with grid implied by "
            f"image_shape={image_shape}, patch_size=({ph}, {pw}), "
            f"stride=({sh}, {sw}): expected L={expected_n_patches} "
            f"(num_h={num_h}, num_w={num_w})."
        )

    kernel = _window_kernel(weight, ph, pw, patches.dtype, patches.device)

    # Weighted patches: broadcast kernel (ph, pw) across (L, C, ph, pw).
    weighted = patches * kernel

    # Numerator fold: (L, C, ph, pw) -> (1, C*ph*pw, L) for F.fold.
    num_flat = (
        weighted.permute(1, 2, 3, 0)
        .reshape(c * ph * pw, n_patches)
        .unsqueeze(0)
    )
    folded_num = F.fold(
        num_flat,
        output_size=(h, w),
        kernel_size=(ph, pw),
        stride=(sh, sw),
    )

    # Denominator fold: replicate kernel across L patches; one "channel"
    # broadcasts across image C in the division.
    kernel_flat = (
        kernel.flatten().unsqueeze(1).repeat(1, n_patches).unsqueeze(0)
    )
    folded_den = F.fold(
        kernel_flat,
        output_size=(h, w),
        kernel_size=(ph, pw),
        stride=(sh, sw),
    )

    # clamp(min=1e-6): for "uniform" this matches reconstruct's count-map
    # clamp. For "hann", corner pixels covered only by edge-weight-zero
    # positions have ~0 numerator AND ~0 denominator — output is dominated
    # by the clamp (i.e., zero). Documented artifact (§9.9).
    return (folded_num / folded_den.clamp(min=1e-6))[0]
