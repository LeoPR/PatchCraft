"""Stitch patches back into an image with configurable weighting kernels.

Where :func:`patchcraft.reconstruct` inverts ``extract`` exactly under the
count-map rule (every coverage count a power of two, always true at
``stride == patch_size``), ``stitch`` is intended for *modified* patches,
meaning patches that have been
denoised, super-resolved, or otherwise altered, where overlap seams are
visible if patches are averaged uniformly. Weighting by a window kernel
(Hann, Gaussian) emphasizes patch centers and reduces those seams.

With ``weight="uniform"``, ``stitch`` is mathematically equivalent to
``reconstruct`` (down to floating-point ordering).

Contract: docs/THEORY.md §2.5 and §9.9.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)

from patchcraft._accel import fold_weighted
from patchcraft._foldgeom import check_fold_geometry

__all__ = ["WeightKind", "stitch"]


WeightKind = Literal["uniform", "hann", "gaussian"]
_WEIGHT_KINDS: tuple[WeightKind, ...] = ("uniform", "hann", "gaussian")


def _hann_1d(n: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Hann window, strictly positive on every sample. ``n == 1`` → ``[1.0]``.

    Uses the interior of a longer symmetric Hann window,
    ``hann_window(n + 2, periodic=False)[1:-1]``, instead of the plain symmetric
    window, which is exactly zero at both endpoints and zeroed every pixel whose
    only covering patches placed it on a patch edge (0.2.0 defect, THEORY §2.5).
    """
    if n == 1:
        return torch.ones(1, dtype=dtype, device=device)
    w = torch.hann_window(n + 2, periodic=False, dtype=dtype, device=device)
    return w[1:-1]


def _gaussian_1d(n: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Gaussian centered at ``(n-1)/2`` with ``sigma = max(1, n/4)``."""
    if n == 1:
        return torch.ones(1, dtype=dtype, device=device)
    sigma = max(1.0, n / 4.0)
    center = (n - 1) / 2.0
    i = torch.arange(n, dtype=dtype, device=device)
    return torch.exp(-((i - center) ** 2) / (2.0 * sigma * sigma))


def _window_1d(
    kind: WeightKind,
    n: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build the 1-D window of ``kind`` and length ``n``."""
    if kind == "uniform":
        return torch.ones(n, dtype=dtype, device=device)
    if kind == "hann":
        return _hann_1d(n, dtype, device)
    if kind == "gaussian":
        return _gaussian_1d(n, dtype, device)
    raise ValueError(
        f"weight must be one of {_WEIGHT_KINDS!r}, got {kind!r}"
    )


def _fold_window_1d(
    w1d: torch.Tensor,
    length: int,
    num: int,
    step: int,
) -> torch.Tensor:
    """``S[y] = sum of w1d[y - i*step]`` over ``i`` in ``[0, num)`` with
    ``0 <= y - i*step < len(w1d)``, the 1-D analog of folding the kernel.

    Within each residue class modulo ``step`` the sum is a sliding window
    over the strided kernel, computed with a cumsum: O(length + len(w1d))
    instead of a 2-D F.fold of the replicated kernel.
    """
    out = w1d.new_zeros(length)
    for r in range(step):
        sub = w1d[r::step]
        n_sub = sub.numel()
        cs = torch.cat([w1d.new_zeros(1), sub.cumsum(0)])
        ks = torch.arange((length - r + step - 1) // step, device=w1d.device)
        hi = torch.clamp(ks + 1, max=n_sub)
        lo = torch.clamp(ks + 1 - num, min=0)
        out[r::step] = cs[hi] - cs[lo]
    return out


def _window_kernel(
    kind: WeightKind,
    ph: int,
    pw: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build a ``(ph, pw)`` window as the outer product of two 1-D windows."""
    wh = _window_1d(kind, ph, dtype, device)
    ww = _window_1d(kind, pw, dtype, device)
    return wh.unsqueeze(1) * ww.unsqueeze(0)


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
    super-resolved). Use :func:`patchcraft.reconstruct` when patches came
    straight from ``extract`` and you want the exact inverse, bit-exact
    under the count-map rule, with no extra arithmetic.

    ``weight`` controls how overlapping patches are blended:

    - ``"uniform"``: each covering patch contributes equally. Mathematically
      equivalent to ``reconstruct`` (no seam attenuation).
    - ``"hann"``: Hann window with full weight at patch center and low weight
      at patch edges. Strong seam suppression. The window is the interior of a
      longer symmetric Hann window (``hann_window(n + 2)[1:-1]``), so it is
      strictly positive on every sample and no output pixel is zeroed by the
      window itself.
    - ``"gaussian"``: Gaussian centered on the patch with per-axis
      ``sigma = max(1.0, ph / 4)`` and ``sigma = max(1.0, pw / 4)``. Smooth
      seam suppression; the 1-D profile stays above ``exp(-2)`` at the edges,
      so the 2-D kernel stays above ``exp(-4)`` at the corners.

    Internally: each patch is multiplied by the 2-D weight kernel, the
    weighted patches are folded into the numerator, and the denominator is
    built from two 1-D window folds (the kernel is separable, so the 2-D
    denominator is their outer product). ``numerator / denominator`` gives
    the output; geometry validation guarantees no uncovered pixels and all
    three windows are strictly positive.

    Rejects (per §9.9): ``dilation != 1``; ``stride > patch_size`` in any
    axis; grids that do not cover the image exactly (same coverage guard as
    :func:`patchcraft.reconstruct`); ``patches.ndim != 4``; non-floating-point
    patches (kernel multiplication breaks integer semantics for non-uniform
    weights, callers convert to ``float`` first); ``image_shape`` inconsistent
    with the patch grid; unknown ``weight``.

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
            "Convert with patches.float(); weight kernels are float-valued and "
            "integer semantics would silently quantize the result."
        )

    if weight not in _WEIGHT_KINDS:
        raise ValueError(
            f"weight must be one of {_WEIGHT_KINDS!r}, got {weight!r}"
        )

    n_patches, c, ph, pw = patches.shape
    h, w, num_h, num_w = check_fold_geometry(
        patches, image_shape, stride, dilation, op="stitch"
    )
    # stride was validated inside the helper; normalize it for the F.fold calls.
    sh, sw = (stride, stride) if isinstance(stride, int) else stride

    # Half-precision accumulates in float32 for the two different reasons
    # reconstruct.py records: float16 overflows the fold, bfloat16 cannot and
    # is promoted for precision instead. Build the kernel and accumulate in
    # float32, cast back at the end (§9.9).
    accum_dtype = (
        torch.float32
        if patches.dtype in (torch.float16, torch.bfloat16)
        else patches.dtype
    )
    wh = _window_1d(weight, ph, accum_dtype, patches.device)
    ww = _window_1d(weight, pw, accum_dtype, patches.device)
    kernel = wh.unsqueeze(1) * ww.unsqueeze(0)

    work = patches.to(accum_dtype)

    # Numerator of the overlap fold. The native path multiplies by the kernel
    # during the gather, skipping the full (L, C, ph, pw) pre-multiply pass and
    # its temporary; the torch path folds the pre-weighted patches.
    numerator = fold_weighted(work, (c, h, w), (sh, sw), kernel)
    if numerator is None:
        # Weighted patches: broadcast kernel (ph, pw) across (L, C, ph, pw).
        weighted = work * kernel
        # (L, C, ph, pw) -> (1, C*ph*pw, L) for F.fold.
        num_flat = (
            weighted.permute(1, 2, 3, 0)
            .reshape(c * ph * pw, n_patches)
            .unsqueeze(0)
        )
        numerator = F.fold(
            num_flat,
            output_size=(h, w),
            kernel_size=(ph, pw),
            stride=(sh, sw),
        )[0]

    # Separable denominator: because the kernel is an outer product,
    # den[y, x] = (sum_i wh[y - i*sh]) * (sum_j ww[x - j*sw]), two 1-D
    # folds + outer product instead of a second 2-D F.fold.
    den_h = _fold_window_1d(wh, h, num_h, sh)
    den_w = _fold_window_1d(ww, w, num_w, sw)
    den = den_h.unsqueeze(1) * den_w.unsqueeze(0)

    # The floor is the dtype's own smallest normal, not an absolute constant.
    # Geometry validation guarantees coverage and all three windows are
    # strictly positive, so the denominator is genuinely positive and this
    # only guards a true zero. An absolute 1e-6 was not the no-op its comment
    # claimed: the 2-D hann corner weight is (pi/(n+1))**4, which falls below
    # 1e-6 at patch 99, so every larger hann stitch had its corner band
    # divided by 1e-6 instead of by the real weight. Measured at 640x640,
    # patch 256, stride 128: max error 0.94 on data in [0, 1], 960 pixels
    # wrong, identical in float64 because it was never a precision problem.
    return (numerator / den.clamp_min(torch.finfo(den.dtype).tiny)).to(patches.dtype)
