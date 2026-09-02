"""Reconstruction of an image from its patches.

Non-overlapping grids are a pure rearrangement; overlapping grids sum the
patches with a closed-form O(H+W) count map — via the optional native
accelerator (patchcraft-accel) when available, otherwise F.fold.

Contract: docs/THEORY.md §2 and §9.2.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)

from patchcraft._accel import fold_weighted
from patchcraft._foldgeom import check_fold_geometry

__all__ = ["reconstruct"]


def reconstruct(
    patches: torch.Tensor,
    image_shape: tuple[int, int, int],
    stride: int | tuple[int, int],
    dilation: int | tuple[int, int] = 1,
) -> torch.Tensor:
    """Inverse of `extract`: rebuild a ``(C, H, W)`` image from ``(L, C, ph, pw)``.

    On non-overlapping grids (``stride == patch_size``) this is a pure
    rearrangement with no arithmetic; on overlapping grids it folds the
    patches (via the optional native accelerator when available, otherwise
    ``F.fold``) plus a closed-form overlap count map, and each pixel's
    reconstructed value is the average of every patch covering it.

    The round trip is bit-exact when every value in that count map is a power of
    two, because dividing a float by a power of two is the one division that
    never rounds. ``stride == patch_size`` always satisfies this, since each
    pixel is then covered exactly once. Overlap satisfies it only sometimes:
    with ``stride == patch_size / 2`` the counts are 1, 2 and 4, so the round
    trip stays exact, while a geometry that puts a 3 or a 9 in the map does not.
    Outside the rule the per-pixel error is bounded by ``(k + 1) * eps * |v|``,
    where ``k`` is that pixel's coverage count — the error grows with the
    overlap, so there is no fixed ULP figure (measured up to 19 ULP at k=81 in
    float32). Widening the dtype does not help, because the deciding axis is
    the geometry rather than the precision.

    Rejects (per §9.2): ``dilation != 1``; ``stride > patch_size`` in any axis
    (partial coverage would synthesize pixel values, which PatchCraft refuses);
    grids that do not cover the image exactly (the last patch must end on the
    image edge on both axes, otherwise pixels would come back zeroed);
    ``image_shape`` inconsistent with the patch grid (channels mismatch or
    ``L`` does not match the geometry); ``patches.ndim != 4``.

    Dtype and device of ``patches`` are preserved. Integer dtypes are
    rejected (``F.fold`` is not implemented for them). Half-precision inputs
    (``float16``, ``bfloat16``) accumulate internally in ``float32`` on
    overlapping grids to avoid overflow inside ``F.fold`` and are cast back on
    return.
    """
    if not isinstance(patches, torch.Tensor):
        raise TypeError(
            f"patches must be torch.Tensor, got {type(patches).__name__}"
        )
    if patches.ndim != 4:
        raise ValueError(
            f"patches must have ndim==4 (L, C, ph, pw), got ndim={patches.ndim}"
        )

    n_patches, c, ph, pw = patches.shape

    if not patches.is_floating_point():
        raise ValueError(
            f"reconstruct requires floating-point patches, got dtype={patches.dtype}. "
            "F.fold is not implemented for integer dtypes; convert with "
            "patches.float() first."
        )

    h, w, num_h, num_w = check_fold_geometry(
        patches, image_shape, stride, dilation, op="reconstruct"
    )
    # stride was validated inside the helper; normalize it for the F.fold calls.
    sh, sw = (stride, stride) if isinstance(stride, int) else stride

    if sh == ph and sw == pw:
        # Non-overlapping grid: every pixel is covered exactly once, so
        # reconstruction is a pure rearrangement -- no fold, no count map, and
        # no widening for half precision (nothing accumulates).
        grid = patches.reshape(num_h, num_w, c, ph, pw)
        return grid.permute(2, 0, 3, 1, 4).reshape(c, h, w)

    # Half-precision inputs overflow inside F.fold, which accumulates the sum
    # of all overlapping patches before the count-map division (fp16 max is
    # 65504). Accumulate in float32 and cast back at the end (§9.2).
    accum_dtype = (
        torch.float32
        if patches.dtype in (torch.float16, torch.bfloat16)
        else patches.dtype
    )
    work = patches.to(accum_dtype)

    # Numerator of the overlap fold: native accelerator when available,
    # otherwise F.fold of the (1, C*ph*pw, L) flattening. Both produce the
    # (C, H, W) sum over covering patches in descending patch order
    # (= ascending kernel offset, ATen col2im's per-pixel order — bit-exact
    # against each other).
    numerator = fold_weighted(work, (c, h, w), (sh, sw), None)
    if numerator is None:
        # (L, C, ph, pw) -> (1, C*ph*pw, L), the layout F.fold expects.
        patches_flat = (
            work.permute(1, 2, 3, 0).reshape(c * ph * pw, n_patches).unsqueeze(0)
        )
        numerator = F.fold(
            patches_flat,
            output_size=(h, w),
            kernel_size=(ph, pw),
            stride=(sh, sw),
        )[0]

    # Closed-form count map: on a full-coverage regular grid the number of
    # patches covering row y is
    #   min(y//sh + 1, num_h) + min((h-1-y)//sh + 1, num_h) - num_h
    # (prefix ramp + suffix ramp - total; inclusion-exclusion), same along W,
    # and the 2-D map is the outer product. O(H+W) integer math instead of a
    # second F.fold of ones; the contents are identical integers, so the
    # division is bit-exact vs the fold.
    ys = torch.arange(h, device=patches.device)
    num_h_t = torch.full_like(ys, num_h)
    count_h = torch.minimum(ys // sh + 1, num_h_t)
    count_h = count_h + torch.minimum((h - 1 - ys) // sh + 1, num_h_t) - num_h
    xs = torch.arange(w, device=patches.device)
    num_w_t = torch.full_like(xs, num_w)
    count_w = torch.minimum(xs // sw + 1, num_w_t)
    count_w = count_w + torch.minimum((w - 1 - xs) // sw + 1, num_w_t) - num_w
    count = (count_h.unsqueeze(1) * count_w.unsqueeze(0)).to(accum_dtype)

    # Every count is an exact integer >= 1 (coverage is validated), so no
    # clamp is needed -- unlike the folded ones, there is no float noise.
    return (numerator / count).to(patches.dtype)
