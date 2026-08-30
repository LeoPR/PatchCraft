"""Reconstruction of an image from its patches via torch.nn.functional.fold.

Contract: docs/THEORY.md §2 and §9.2.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)

from patchcraft.extract import _as_pair

__all__ = ["reconstruct"]


def reconstruct(
    patches: torch.Tensor,
    image_shape: tuple[int, int, int],
    stride: int | tuple[int, int],
    dilation: int | tuple[int, int] = 1,
) -> torch.Tensor:
    """Inverse of `extract`: rebuild a ``(C, H, W)`` image from ``(L, C, ph, pw)``.

    Uses ``F.fold`` plus an overlap count map. Each pixel's reconstructed value
    is the average of every patch covering it.

    The round trip is bit-exact when every value in that count map is a power of
    two, because dividing a float by a power of two is the one division that
    never rounds. ``stride == patch_size`` always satisfies this, since each
    pixel is then covered exactly once. Overlap satisfies it only sometimes:
    with ``stride == patch_size / 2`` the counts are 1, 2 and 4, so the round
    trip stays exact, while a geometry that puts a 3 or a 9 in the map does not.
    Outside the rule the error is about 1 ULP, measured at 2.4e-07 in float32 on
    a 16x16 image with ``patch_size=4, stride=1``. Widening the dtype does not
    help, because the deciding axis is the geometry rather than the precision.

    Rejects (per §9.2): ``dilation != 1``; ``stride > patch_size`` in any axis
    (partial coverage would synthesize pixel values, which PatchCraft refuses);
    grids that do not cover the image exactly (the last patch must end on the
    image edge on both axes, otherwise pixels would come back zeroed);
    ``image_shape`` inconsistent with the patch grid (channels mismatch or
    ``L`` does not match the geometry); ``patches.ndim != 4``.

    Dtype and device of ``patches`` are preserved. Integer dtypes are
    rejected (``F.fold`` is not implemented for them). Half-precision inputs
    (``float16``, ``bfloat16``) accumulate internally in ``float32`` to avoid
    overflow inside ``F.fold`` and are cast back on return.
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
            f"reconstruct requires dilation==1, got dilation=({dh}, {dw}). "
            "Patches extracted with dilation > 1 cannot round-trip; consume them as features."
        )
    if sh > ph or sw > pw:
        raise ValueError(
            f"reconstruct forbids stride > patch_size (partial coverage forbidden), "
            f"got stride=({sh}, {sw}) and patch_size=({ph}, {pw})."
        )

    num_h = (h - ph) // sh + 1
    num_w = (w - pw) // sw + 1
    if num_h <= 0 or num_w <= 0:
        raise ValueError(
            f"image_shape={image_shape} too small for patch_size=({ph}, {pw}) "
            f"and stride=({sh}, {sw})"
        )
    covered_h = (num_h - 1) * sh + ph
    covered_w = (num_w - 1) * sw + pw
    if covered_h != h or covered_w != w:
        raise ValueError(
            f"patch grid leaves pixels uncovered (partial coverage forbidden): "
            f"image_shape={image_shape}, patch_size=({ph}, {pw}), "
            f"stride=({sh}, {sw}) covers ({covered_h}, {covered_w}) of "
            f"({h}, {w}). Choose a geometry with exact coverage "
            f"(see patchcraft.tilings)."
        )
    expected_n_patches = num_h * num_w
    if n_patches != expected_n_patches:
        raise ValueError(
            f"patches.shape[0]={n_patches} inconsistent with grid implied by "
            f"image_shape={image_shape}, patch_size=({ph}, {pw}), "
            f"stride=({sh}, {sw}): expected L={expected_n_patches} "
            f"(num_h={num_h}, num_w={num_w})."
        )

    # Half-precision inputs overflow inside F.fold, which accumulates the sum
    # of all overlapping patches before the count-map division (fp16 max is
    # 65504). Accumulate in float32 and cast back at the end (§9.2).
    accum_dtype = (
        torch.float32
        if patches.dtype in (torch.float16, torch.bfloat16)
        else patches.dtype
    )
    work = patches.to(accum_dtype)

    # (L, C, ph, pw) -> (1, C*ph*pw, L), the layout F.fold expects.
    patches_flat = (
        work.permute(1, 2, 3, 0).reshape(c * ph * pw, n_patches).unsqueeze(0)
    )
    folded = F.fold(
        patches_flat,
        output_size=(h, w),
        kernel_size=(ph, pw),
        stride=(sh, sw),
    )

    # Count map: same fold geometry but 1 "channel", which broadcasts across C in division.
    ones = torch.ones(
        1, ph * pw, n_patches, dtype=accum_dtype, device=patches.device
    )
    count = F.fold(
        ones,
        output_size=(h, w),
        kernel_size=(ph, pw),
        stride=(sh, sw),
    )

    # clamp(min=1e-6) absorbs float noise on covered pixels; geometry validation
    # above guarantees there are no uncovered pixels (count > 0 everywhere).
    return (folded / count.clamp(min=1e-6))[0].to(patches.dtype)
