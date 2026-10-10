"""Reconstruction of an image from its patches.

Non-overlapping grids are a pure rearrangement; overlapping grids sum the
patches with a closed-form O(H+W) count map, or via the optional native
accelerator (built into this wheel where available) or F.fold. Integer patches
take an exact integer mean instead.

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
    two, which in a legal geometry means every count is 1, 2 or 4: adding two or
    four copies of a value never rounds, and neither does dividing by 2 or 4.
    ``stride == patch_size`` always satisfies this, since each pixel is then
    covered exactly once. Overlap satisfies it only sometimes: with
    ``stride == patch_size / 2`` the counts are 1, 2 and 4, so the round trip
    stays exact, while a geometry that puts a 3 or a 9 in the map does not.
    Exact means equal as values, which is what ``torch.equal`` checks: the fold
    sums into ``+0.0``, so on an overlapping grid a ``-0.0`` pixel comes back as
    ``+0.0`` and a NaN may come back with different bits.
    Outside the rule the per-pixel error is bounded by ``(k + 1) * eps * |v|``,
    where ``k`` is that pixel's coverage count, so the error grows with the
    overlap, so there is no fixed ULP figure (measured up to 19 ULP at k=81 in
    float32). Widening the *input* dtype does not help, because the deciding
    axis is the geometry rather than the precision; widening the accumulator
    does (a float64 accumulator is exact on every legal geometry measured, a
    float32 one only on the power-of-two ones), and the accumulator is not a
    parameter today. See THEORY section 9.2 and ADR 0004.

    Rejects (per §9.2): ``dilation != 1``; ``stride > patch_size`` in any axis
    (partial coverage would synthesize pixel values, which PatchCraft refuses);
    grids that do not cover the image exactly (the last patch must end on the
    image edge on both axes, otherwise pixels would come back zeroed);
    ``image_shape`` inconsistent with the patch grid (channels mismatch or
    ``L`` does not match the geometry); ``patches.ndim != 4``.

    Integer patches (``uint8``, ``int8``, ``int16``, ``int32``, ``int64``) are
    exact in every legal geometry, with no count-map rule: the covering values
    of an unmodified pixel are all ``v``, their integer sum is ``k * v``, and
    ``k * v`` divided by ``k`` is ``v``. Patches that disagree get the exact
    mean rounded half to even, the rounding ``resize`` already applies to
    integers, and no intermediate overflows, ``int64`` included. ``bool`` and
    complex patches are rejected: a mean of booleans is a vote, which ADR 0004
    leaves to a function of its own.

    Dtype and device of ``patches`` are preserved. Half-precision inputs
    (``float16``, ``bfloat16``) accumulate internally in ``float32`` on
    overlapping grids, ``float16`` because its sum would overflow inside the
    fold and ``bfloat16`` for precision, and are cast back on return.
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

    if patches.dtype == torch.bool or patches.is_complex():
        raise ValueError(
            f"reconstruct takes floating-point or integer patches, got "
            f"dtype={patches.dtype}. A mean of booleans is a vote rather than an "
            "average, and complex patches have no rounding rule here; convert "
            "with patches.float() first."
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
        # Exactly one copy, as in `extract`: on a single-patch grid the permuted
        # view is already contiguous and a reshape would alias the caller's
        # patches, so a later write to them would change an image this
        # function already returned. Cloning unconditionally avoids asking the
        # tensor for its storage, which a torch.vmap batched tensor lacks.
        grid = patches.reshape(num_h, num_w, c, ph, pw)
        return (
            grid.permute(2, 0, 3, 1, 4)
            .clone(memory_format=torch.contiguous_format)
            .view(c, h, w)
        )

    if not patches.is_floating_point():
        return _integer_mean(patches, c, h, w, ph, pw, sh, sw, num_h, num_w)

    # Half-precision accumulates in float32, for one reason in float16 and a
    # different one in bfloat16. float16 genuinely overflows: F.fold sums every
    # overlapping patch before the count-map division, and a constant 10000.0
    # image at ph=3, stride=1 reached inf in 144 of 256 pixels against a finite
    # max of 65504. bfloat16 carries float32's exponent and cannot overflow
    # here; its numerator peaked at 9.0e+04 against a finite max of 3.4e+38.
    # What the promotion buys bfloat16 is exactness, since it has 8 mantissa
    # bits to float16's 11: at 1x16x16, ph=3, stride=1, on data drawn in
    # bfloat16, the native fold misses by 7.8e-03 and the promoted round trip
    # returns the input bit for bit. Cast back at the end (§9.2).
    accum_dtype = (
        torch.float32
        if patches.dtype in (torch.float16, torch.bfloat16)
        else patches.dtype
    )
    work = patches.to(accum_dtype)

    # Numerator of the overlap fold: native accelerator when available,
    # otherwise F.fold of the (1, C*ph*pw, L) flattening. Both produce the
    # (C, H, W) sum over covering patches in descending patch order
    # (= ascending kernel offset, ATen col2im's per-pixel order, bit-exact
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
    #
    # The suffix ramp is the prefix ramp reversed, and a square grid reuses
    # the H axis for W. The outer product is taken in the accumulation dtype:
    # both factors are small exact integers, so the multiply rounds their
    # exact product once, exactly as casting the int64 product did.
    count_h = _axis_counts(h, sh, num_h, accum_dtype, patches.device)
    count_w = (
        count_h
        if (w, sw, num_w) == (h, sh, num_h)
        else _axis_counts(w, sw, num_w, accum_dtype, patches.device)
    )
    count = count_h.unsqueeze(1) * count_w.unsqueeze(0)

    # Every count is an exact integer >= 1 (coverage is validated), so no
    # clamp is needed -- unlike the folded ones, there is no float noise.
    # The numerator is a fresh tensor this function owns, so it is divided
    # in place rather than allocating a second (C, H, W) buffer.
    return numerator.div_(count).to(patches.dtype)


def _integer_mean(
    patches: torch.Tensor,
    c: int,
    h: int,
    w: int,
    ph: int,
    pw: int,
    sh: int,
    sw: int,
    num_h: int,
    num_w: int,
) -> torch.Tensor:
    """Per-pixel mean of integer patches, rounded half to even, never overflowing.

    Up to ``int32`` the covering values are summed in ``int64``, where any
    number of 32-bit values a legal geometry can stack cannot overflow. An
    ``int64`` sum could, so ``int64`` patches are divided first: each value is
    split by its own pixel's count ``k`` into a quotient and a remainder, the
    quotients sum to at most the largest input, the remainders to less than
    ``k * k``, and the mean is rebuilt from the two. Either way the result is
    ``floor(S / k)`` plus the half-to-even step, computed exactly in integers.
    Out-of-place ``index_add`` keeps it usable under ``torch.vmap``.
    """
    device = patches.device
    count = (
        _axis_counts(h, sh, num_h, torch.int64, device).unsqueeze(1)
        * _axis_counts(w, sw, num_w, torch.int64, device).unsqueeze(0)
    ).view(1, h * w)
    # The pixel each patch entry lands on, shared by every channel; the patch
    # order is row-major over the grid, as extract produces it.
    top = (torch.arange(num_h, device=device) * sh).view(num_h, 1, 1, 1)
    left = (torch.arange(num_w, device=device) * sw).view(1, num_w, 1, 1)
    rows = top + torch.arange(ph, device=device).view(1, 1, ph, 1)
    cols = left + torch.arange(pw, device=device).view(1, 1, 1, pw)
    index = (rows * w + cols).reshape(-1)
    values = patches.permute(1, 0, 2, 3).reshape(c, -1).to(torch.int64)
    zeros = torch.zeros(c, h * w, dtype=torch.int64, device=device)

    if patches.dtype == torch.int64:
        k = count.view(-1)[index].view(1, -1)
        quotient = torch.div(values, k, rounding_mode="floor")
        base = zeros.index_add(1, index, quotient)
        rest = zeros.index_add(1, index, values - quotient * k)
    else:
        base = zeros
        rest = zeros.index_add(1, index, values)

    carry = torch.div(rest, count, rounding_mode="floor")
    remainder = rest - carry * count                          # 0 <= remainder < k
    mean = base + carry                                       # floor(S / k)
    up = (2 * remainder > count) | ((2 * remainder == count) & (mean % 2 == 1))
    return (mean + up.to(torch.int64)).view(c, h, w).to(patches.dtype)


def _axis_counts(
    n: int, step: int, num: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Patches covering each position along one axis of a full-coverage grid."""
    prefix = (torch.arange(n, device=device) // step + 1).clamp_(max=num)
    return (prefix + prefix.flip(0) - num).to(dtype)
