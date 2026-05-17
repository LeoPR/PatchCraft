"""Pre-flight geometry helpers: enumerate valid patch tilings, count patches.

Pure number-only API — does not touch tensors or images. Useful for:
- asking the lib "what patch sizes fit my image cleanly?" before extracting;
- precomputing patch counts for memory planning;
- driving parametrized tests over the full space of valid geometries.

Contract: docs/THEORY.md §1.5 and §9.6.
"""
from __future__ import annotations

from typing import NamedTuple

from patchcraft.extract import _as_pair

__all__ = [
    "PairedTilingSpec",
    "TilingSpec",
    "num_patches",
    "paired_tilings",
    "scale_factor",
    "tilings",
]


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


class PairedTilingSpec(NamedTuple):
    """A pair of tilings that align across two resolutions of the same image.

    ``hr.patch_size == scale_factor * lr.patch_size`` and same for stride;
    ``lr.total_patches == hr.total_patches`` by construction. Patch ``k`` on
    the LR side and patch ``k`` on the HR side cover the same image region
    at different resolutions.
    """

    lr: TilingSpec
    hr: TilingSpec
    scale_factor: int


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


def scale_factor(
    lr_shape: tuple[int, int] | tuple[int, int, int],
    hr_shape: tuple[int, int] | tuple[int, int, int],
) -> int | None:
    """Return the integer scale factor between two image shapes, or ``None``.

    Accepts ``(H, W)`` or ``(C, H, W)`` for either argument (channels are
    ignored). Returns ``k`` such that
    ``hr_shape[-2:] == (k * lr_shape[-2], k * lr_shape[-1])``, or ``None``
    when no such integer ``k >= 1`` exists (non-divisible, anisotropic, or
    LR larger than HR).

    Pure shape math; no tensor, no allocation. Use it before calling
    :func:`patchcraft.pair` to discover the scale factor from data instead
    of hard-coding it.
    """
    for name, shape in (("lr_shape", lr_shape), ("hr_shape", hr_shape)):
        if not (isinstance(shape, tuple) and len(shape) in (2, 3)):
            raise ValueError(
                f"{name} must be (H, W) or (C, H, W), got {shape!r}"
            )
        h, w = shape[-2], shape[-1]
        for axis, val in (("H", h), ("W", w)):
            if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
                raise ValueError(
                    f"{name}[{axis}] must be a positive int, got {val!r}"
                )

    h_lr, w_lr = lr_shape[-2], lr_shape[-1]
    h_hr, w_hr = hr_shape[-2], hr_shape[-1]
    if h_hr % h_lr != 0 or w_hr % w_lr != 0:
        return None
    sf_h = h_hr // h_lr
    sf_w = w_hr // w_lr
    if sf_h != sf_w or sf_h < 1:
        return None
    return sf_h


def paired_tilings(
    lr_shape: tuple[int, int] | tuple[int, int, int],
    hr_shape: tuple[int, int] | tuple[int, int, int],
    *,
    allow_overlap: bool = False,
    min_patch_size: int = 2,
    max_patch_size: int | None = None,
) -> list[PairedTilingSpec]:
    """Enumerate aligned tiling pairs between two resolutions of the same image.

    Requires ``hr_shape`` to be an integer multiple of ``lr_shape`` (see
    :func:`scale_factor`). For each LR tiling emitted by :func:`tilings`,
    derives the matching HR tiling by multiplying patch size and stride by
    the scale factor. Both sides have identical ``total_patches`` and patch
    ``k`` covers the same image region on both sides.

    Use the result to drive :func:`patchcraft.pair` with confidence that the
    parameters produce sound, aligned LR/HR patch sets.

    Raises
    ------
    ValueError
        If ``lr_shape`` and ``hr_shape`` are not related by an integer
        scale factor, or on the same input validation cases as
        :func:`tilings`.
    """
    sf = scale_factor(lr_shape, hr_shape)
    if sf is None:
        raise ValueError(
            f"lr_shape={lr_shape} and hr_shape={hr_shape} are not related "
            "by a positive integer scale factor on both spatial axes"
        )
    h_hr, w_hr = hr_shape[-2], hr_shape[-1]
    lr_specs = tilings(
        lr_shape,
        allow_overlap=allow_overlap,
        min_patch_size=min_patch_size,
        max_patch_size=max_patch_size,
    )

    pairs: list[PairedTilingSpec] = []
    for lr in lr_specs:
        ph_hr = lr.patch_size[0] * sf
        pw_hr = lr.patch_size[1] * sf
        sh_hr = lr.stride[0] * sf
        sw_hr = lr.stride[1] * sf
        nh_hr = (h_hr - ph_hr) // sh_hr + 1
        nw_hr = (w_hr - pw_hr) // sw_hr + 1
        hr = TilingSpec(
            patch_size=(ph_hr, pw_hr),
            stride=(sh_hr, sw_hr),
            dilation=(1, 1),
            num_patches=(nh_hr, nw_hr),
            total_patches=nh_hr * nw_hr,
            overlap=lr.overlap,
        )
        pairs.append(PairedTilingSpec(lr=lr, hr=hr, scale_factor=sf))
    return pairs
