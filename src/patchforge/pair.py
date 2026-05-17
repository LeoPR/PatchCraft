"""LR ↔ HR patch pairing.

Given a low-resolution image, a high-resolution image, and an integer scale
factor, produce patches on both sides that correspond pixel-region for
pixel-region (patch ``k`` on each side covers the same image area, at
different resolutions).

Contract: docs/THEORY.md §3 and §9.3.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from patchforge.extract import _as_pair, extract

__all__ = ["PatchMeta", "PatchPair", "pair"]


@dataclass(frozen=True, slots=True)
class PatchMeta:
    """Metadata for a single LR/HR patch correspondence.

    Lives on CPU (never moves to GPU regardless of where the patches live).
    Identifies *which* patch in the grid — coordinates are in LR pixel space;
    multiply ``row`` and ``col`` by ``scale_factor`` to get HR coordinates.
    """

    patch_index: int
    row: int
    col: int
    lr_patch_size: tuple[int, int]
    hr_patch_size: tuple[int, int]
    image_id: str | None = None


@dataclass(frozen=True, slots=True)
class PatchPair:
    """Result of ``pair()``: LR and HR patch tensors plus per-patch metadata."""

    lr_patches: torch.Tensor  # (L, C, ph_lr, pw_lr)
    hr_patches: torch.Tensor  # (L, C, ph_hr, pw_hr)
    metas: tuple[PatchMeta, ...]

    def __len__(self) -> int:
        return int(self.lr_patches.shape[0])


def pair(
    lr_image: torch.Tensor,
    hr_image: torch.Tensor,
    lr_patch_size: int | tuple[int, int],
    scale_factor: int,
    stride: int | tuple[int, int],
    *,
    image_id: str | None = None,
) -> PatchPair:
    """Extract aligned LR/HR patch pairs.

    Both images are ``(C, H, W)``. ``hr_image.shape`` must equal
    ``(C, scale_factor * H_lr, scale_factor * W_lr)``. HR patch size and HR
    stride are derived as ``scale_factor * lr_*``; dilation is fixed at 1.

    Patch ``k`` on the LR side has its top-left at LR pixel
    ``(row * sh_lr, col * sw_lr)``; the corresponding HR patch covers the
    same image region at ``scale_factor`` times the resolution.

    Returns a ``PatchPair`` with:
    - ``lr_patches``: ``Tensor[L, C, ph_lr, pw_lr]`` (from `extract`).
    - ``hr_patches``: ``Tensor[L, C, ph_hr, pw_hr]``.
    - ``metas``: tuple of ``L`` :class:`PatchMeta` (CPU only).

    Raises ``ValueError`` on any of the conditions listed in §9.3: non-int
    or non-positive ``scale_factor``; HR shape that does not match
    ``scale_factor * LR shape``; channel mismatch between LR and HR; LR or
    HR not 3D; non-positive ``lr_patch_size`` or ``stride``.

    LR and HR are expected to share the same dtype and device; mismatch is
    rejected (caller normalizes upstream).
    """
    if not isinstance(lr_image, torch.Tensor):
        raise TypeError(
            f"lr_image must be torch.Tensor, got {type(lr_image).__name__}"
        )
    if not isinstance(hr_image, torch.Tensor):
        raise TypeError(
            f"hr_image must be torch.Tensor, got {type(hr_image).__name__}"
        )
    if lr_image.ndim != 3:
        raise ValueError(f"lr_image must have ndim==3, got ndim={lr_image.ndim}")
    if hr_image.ndim != 3:
        raise ValueError(f"hr_image must have ndim==3, got ndim={hr_image.ndim}")

    if (
        not isinstance(scale_factor, int)
        or isinstance(scale_factor, bool)
        or scale_factor <= 0
    ):
        raise ValueError(
            f"scale_factor must be a positive int, got {scale_factor!r}"
        )

    c_lr, h_lr, w_lr = lr_image.shape
    c_hr, h_hr, w_hr = hr_image.shape

    if c_lr != c_hr:
        raise ValueError(
            f"channel mismatch: lr_image has C={c_lr}, hr_image has C={c_hr}"
        )
    if lr_image.dtype != hr_image.dtype:
        raise ValueError(
            f"dtype mismatch: lr_image is {lr_image.dtype}, hr_image is {hr_image.dtype}"
        )
    if lr_image.device != hr_image.device:
        raise ValueError(
            f"device mismatch: lr_image on {lr_image.device}, hr_image on {hr_image.device}"
        )

    if h_hr != scale_factor * h_lr or w_hr != scale_factor * w_lr:
        raise ValueError(
            f"hr_image shape {hr_image.shape} does not match "
            f"scale_factor={scale_factor} times lr_image shape {lr_image.shape}; "
            f"expected hr shape ({c_lr}, {scale_factor * h_lr}, {scale_factor * w_lr})"
        )

    ph_lr, pw_lr = _as_pair(lr_patch_size, "lr_patch_size")
    sh_lr, sw_lr = _as_pair(stride, "stride")

    ph_hr, pw_hr = ph_lr * scale_factor, pw_lr * scale_factor
    sh_hr, sw_hr = sh_lr * scale_factor, sw_lr * scale_factor

    lr_patches = extract(
        lr_image, patch_size=(ph_lr, pw_lr), stride=(sh_lr, sw_lr)
    )
    hr_patches = extract(
        hr_image, patch_size=(ph_hr, pw_hr), stride=(sh_hr, sw_hr)
    )

    # Geometry is identical by construction (integer scale), so counts match.
    n_patches = int(lr_patches.shape[0])
    if hr_patches.shape[0] != n_patches:  # defensive — shouldn't happen
        raise RuntimeError(
            f"internal: lr and hr patch counts diverged "
            f"({n_patches} vs {hr_patches.shape[0]}); please file a bug"
        )

    num_w_lr = (w_lr - pw_lr) // sw_lr + 1 if n_patches > 0 else 0
    metas = tuple(
        PatchMeta(
            patch_index=k,
            row=(k // num_w_lr) * sh_lr if num_w_lr else 0,
            col=(k % num_w_lr) * sw_lr if num_w_lr else 0,
            lr_patch_size=(ph_lr, pw_lr),
            hr_patch_size=(ph_hr, pw_hr),
            image_id=image_id,
        )
        for k in range(n_patches)
    )

    return PatchPair(lr_patches=lr_patches, hr_patches=hr_patches, metas=metas)
