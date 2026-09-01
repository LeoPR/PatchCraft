"""Shared fold-geometry validation for reconstruct and stitch (THEORY §9.2/§9.9).

The two functions enforced identical rules on the image/patch-grid
relationship with ~60 lines of verbatim-duplicated validation; this module
centralizes them. Error messages are byte-identical to the pre-refactor
ones, with ``op`` filling the function-name slot.
"""
from __future__ import annotations

import torch

from patchcraft.extract import _as_pair

__all__ = ["check_fold_geometry"]


def check_fold_geometry(
    patches: torch.Tensor,
    image_shape: tuple[int, int, int],
    stride: int | tuple[int, int],
    dilation: int | tuple[int, int],
    *,
    op: str,
) -> tuple[int, int, int, int]:
    """Validate the fold geometry; return ``(h, w, num_h, num_w)``.

    Caller owns the dtype/ndim checks (their messages differ between
    ``reconstruct`` and ``stitch``); this owns everything from
    ``image_shape`` shape-checking through grid/L consistency.
    """
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
            f"{op} requires dilation==1, got dilation=({dh}, {dw}). "
            "Patches extracted with dilation > 1 cannot round-trip; consume them as features."
        )
    if sh > ph or sw > pw:
        raise ValueError(
            f"{op} forbids stride > patch_size (partial coverage forbidden), "
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
    return (h, w, num_h, num_w)
