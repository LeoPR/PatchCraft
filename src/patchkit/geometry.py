"""Pre-flight geometry helpers: enumerate valid patch tilings, count patches.

Pure number-only API — does not touch tensors or images. Useful for:
- asking the lib "what patch sizes fit my image cleanly?" before extracting;
- precomputing patch counts for memory planning;
- driving parametrized tests over the full space of valid geometries.

Contract: docs/THEORY.md §1.5 and §9.6.
"""
from __future__ import annotations

from typing import NamedTuple

from patchkit.extract import _as_pair

__all__ = ["TilingSpec", "num_patches", "tilings"]


class TilingSpec(NamedTuple):
    """One valid patch geometry for an image.

    ``overlap=False`` means an *exact tile*: ``patch_size == stride`` and the
    image is divided into a clean grid with no overlap and no waste.
    ``overlap=True`` means ``stride < patch_size`` and full coverage is still
    achieved — adjacent patches share pixels.
    """

    patch_size: tuple[int, int]
    stride: tuple[int, int]
    dilation: tuple[int, int]
    num_patches: tuple[int, int]
    total_patches: int
    overlap: bool


def num_patches(
    image_shape: tuple[int, ...],
    patch_size: int | tuple[int, int],
    stride: int | tuple[int, int],
    dilation: int | tuple[int, int] = 1,
) -> tuple[int, int]:
    """Return ``(num_h, num_w)`` — how many patches `extract` would produce.

    Accepts ``(H, W)`` or ``(C, H, W)`` for ``image_shape``. Returns ``(0, 0)``
    in either axis when the effective patch does not fit (mirroring `extract`'s
    empty-tensor behavior). Does not allocate or touch any tensor.
    """
    if not (isinstance(image_shape, tuple) and len(image_shape) in (2, 3)):
        raise ValueError(
            f"image_shape must be (H, W) or (C, H, W), got {image_shape!r}"
        )
    h, w = (image_shape[-2], image_shape[-1])
    for axis_name, val in zip(("H", "W"), (h, w), strict=True):
        if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
            raise ValueError(
                f"image_shape[{axis_name}] must be a positive int, got {val!r}"
            )

    ph, pw = _as_pair(patch_size, "patch_size")
    sh, sw = _as_pair(stride, "stride")
    dh, dw = _as_pair(dilation, "dilation")

    eff_h = dh * (ph - 1) + 1
    eff_w = dw * (pw - 1) + 1
    nh = (h - eff_h) // sh + 1 if h >= eff_h else 0
    nw = (w - eff_w) // sw + 1 if w >= eff_w else 0
    return (nh, nw)


def tilings(
    image_shape: tuple[int, int] | tuple[int, int, int],
    *,
    allow_overlap: bool = False,
    min_patch_size: int = 2,
    max_patch_size: int | None = None,
) -> list[TilingSpec]:
    """Enumerate square geometries that fully cover an image (``dilation==1``).

    Always emits geometries with **full coverage** (no truncated rows/cols).
    Patches are square: ``ph == pw``, ``sh == sw``.

    Parameters
    ----------
    image_shape
        ``(H, W)`` or ``(C, H, W)``. Only H, W matter.
    allow_overlap
        If False (default), emit only exact tilings (``stride == patch_size``
        and ``H % p == 0`` and ``W % p == 0``). If True, also emit
        ``stride < patch_size`` geometries with ``(H - p) % s == 0`` and
        ``(W - p) % s == 0`` — overlap-with-clean-edges.
    min_patch_size
        Smallest patch to consider. Defaults to 2 (skips the trivial pixel-wise
        tiling at ``p == 1``).
    max_patch_size
        Largest patch to consider. Defaults to ``min(H, W)``.

    Returns
    -------
    list[TilingSpec]
        Sorted by ``(patch_size[0], stride[0])`` ascending. Always at least
        one entry: ``(p=min(H, W), s=p)`` when image is square.

    Raises
    ------
    ValueError
        On malformed ``image_shape``, non-positive bounds, or
        ``min_patch_size > max_patch_size``.
    """
    if not (isinstance(image_shape, tuple) and len(image_shape) in (2, 3)):
        raise ValueError(
            f"image_shape must be (H, W) or (C, H, W), got {image_shape!r}"
        )
    h, w = (image_shape[-2], image_shape[-1])
    for axis_name, val in zip(("H", "W"), (h, w), strict=True):
        if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
            raise ValueError(
                f"image_shape[{axis_name}] must be a positive int, got {val!r}"
            )

    if (
        not isinstance(min_patch_size, int)
        or isinstance(min_patch_size, bool)
        or min_patch_size <= 0
    ):
        raise ValueError(
            f"min_patch_size must be a positive int, got {min_patch_size!r}"
        )
    if max_patch_size is None:
        max_patch_size = min(h, w)
    if (
        not isinstance(max_patch_size, int)
        or isinstance(max_patch_size, bool)
        or max_patch_size <= 0
    ):
        raise ValueError(
            f"max_patch_size must be a positive int or None, got {max_patch_size!r}"
        )
    if min_patch_size > max_patch_size:
        raise ValueError(
            f"min_patch_size ({min_patch_size}) > max_patch_size ({max_patch_size})"
        )

    upper = min(max_patch_size, h, w)
    results: list[TilingSpec] = []
    for p in range(min_patch_size, upper + 1):
        # Exact tile: stride == patch_size; requires divisibility on both axes.
        if h % p == 0 and w % p == 0:
            nh, nw = h // p, w // p
            results.append(TilingSpec(
                patch_size=(p, p),
                stride=(p, p),
                dilation=(1, 1),
                num_patches=(nh, nw),
                total_patches=nh * nw,
                overlap=False,
            ))
        if allow_overlap:
            for s in range(1, p):
                if (h - p) % s == 0 and (w - p) % s == 0:
                    nh = (h - p) // s + 1
                    nw = (w - p) // s + 1
                    results.append(TilingSpec(
                        patch_size=(p, p),
                        stride=(s, s),
                        dilation=(1, 1),
                        num_patches=(nh, nw),
                        total_patches=nh * nw,
                        overlap=True,
                    ))
    return results
