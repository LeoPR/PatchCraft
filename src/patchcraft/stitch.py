"""Stitch patches back into an image with configurable weighting kernels.

Where :func:`patchcraft.reconstruct` inverts ``extract`` exactly under the
count-map rule (every coverage count a power of two, always true at
``stride == patch_size``), ``stitch`` is intended for *modified* patches,
meaning patches that have been
denoised, super-resolved, or otherwise altered, where overlap seams are
visible if patches are averaged uniformly. Weighting by a window kernel
(Hann, Gaussian) emphasizes patch centers and reduces those seams.

With ``weight="uniform"``, ``stitch`` returns what ``reconstruct`` returns:
on an overlapping grid it is the same computation, bit for bit. On a
non-overlapping one ``reconstruct`` only copies, while ``stitch`` returns what
the fold it stands in for would: ``+0.0`` for a ``-0.0``, and a NaN that stays
a NaN but may change its bits.

Contract: docs/THEORY.md §2.5 and §9.9.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)

from patchcraft._accel import fold_weighted
from patchcraft._foldgeom import check_fold_geometry
from patchcraft.reconstruct import reconstruct

__all__ = ["WeightKind", "stitch"]


WeightKind = Literal["uniform", "hann", "gaussian"]
_WEIGHT_KINDS: tuple[WeightKind, ...] = ("uniform", "hann", "gaussian")

# Devices on which the 1-D window sums accumulate in float64: each has it, and
# a meta tensor has no data to sum. Any other device accumulates in the
# window's own dtype, because float64 may be missing there (MPS has none).
# The sums never leave the window's device, so no call waits on a copy.
_FLOAT64_DEVICES = ("cpu", "cuda", "meta")


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

    Every output is a sum of its at most ``ceil(len(w1d) / step)`` terms with
    nothing subtracted. On the CPU, CUDA and meta it is accumulated in float64
    and rounded once to ``w1d``'s dtype: for a float32 window float64 holds
    every partial sum exactly, so the result is the correctly rounded sum,
    and a float64 window is rounded at each addition, as any float64 sum is.
    On any other device it is accumulated in ``w1d``'s own dtype, rounded at
    each addition too. The subtraction was the defect:
    until 0.6.1 each output was the difference of two cumulative sums of its
    stride residue, and on the last ``len(w1d) - step`` positions, where every
    covering patch sees the pixel near its far edge, the two partial sums are
    large and their difference is a few small edge weights. Measured in
    float32 hann, all-ones patches came back 2.4e-03 above 1 at patch 256 and
    stride 32, and the difference reached exactly zero first at patch 616 and
    stride 1, so the floor in ``stitch`` divided covered pixels by the
    smallest normal.

    Cut the window into ``step``-long blocks; output row ``q`` (positions
    ``q*step`` to ``q*step + step - 1``) sums the ``num`` consecutive blocks
    ``t`` with ``0 <= q - t < num``, fewer where the run passes an end of the
    window. The first ``num`` rows are prefix sums of the blocks, the same
    cumulative sum the old code returned there on the CPU, bit for bit (on
    CUDA a float32 window's rows now equal the CPU's, where the old ones
    differed), and the last ``num`` rows are suffix sums. With at least as many patches as blocks,
    the rows between hold the whole window. With fewer, they are runs that
    touch neither end, and each is a suffix sum of one ``num``-long chunk of
    the blocks plus a prefix sum of the next (van Herk and Gil-Werman's
    sliding-window method), so every row costs one addition at most and the
    work grows with the axis, not with the number of patches.

    Everything runs on the window's device, so a CUDA call does not wait for
    the device and a meta call returns a meta tensor.
    """
    # Out of place throughout: a recorder such as torch.func.linearize replays
    # the ops it saw, and writes into views of a fresh buffer did not survive
    # the replay. Every tensor below derives from ``w``, so all of it stays on
    # the window's device whatever the default device is.
    n = w1d.numel()
    acc = torch.float64 if w1d.device.type in _FLOAT64_DEVICES else w1d.dtype
    w = w1d.detach().to(acc)
    blocks = -(-n // step)
    b = F.pad(w, (0, blocks * step - n)).view(blocks, step)
    prefix = b.cumsum(0)
    suffix = b.flip(0).cumsum(0).flip(0)
    if num >= blocks:
        middle = prefix[-1:].expand(num - blocks, step)
        rows = torch.cat([prefix, middle, suffix[1:]])
    else:
        # Rows num to blocks - 2 sum the runs that start at blocks 1 to
        # blocks - 1 - num. In chunks of num blocks, a run starting at a is
        # the suffix of a's chunk from a, plus the prefix of the next chunk
        # that stops short of a + num, which is empty when a starts a chunk.
        chunks = -(-blocks // num)
        c = F.pad(b, (0, 0, 0, chunks * num - blocks)).view(chunks, num, step)
        run_suffix = c.flip(1).cumsum(1).flip(1).reshape(-1, step)
        run_prefix = F.pad(c.cumsum(1)[:, :-1], (0, 0, 1, 0)).reshape(-1, step)
        inner = blocks - 1 - num
        middle = run_suffix[1 : 1 + inner] + run_prefix[1 + num : 1 + num + inner]
        rows = torch.cat([prefix[:num], middle, suffix[blocks - num :]])
    out = rows.reshape(-1)[:length]
    return out.to(w1d.dtype)


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
    three windows are strictly positive. Under ``"uniform"`` every weight is
    1, so an overlapping grid goes to :func:`patchcraft.reconstruct`, which is
    the same arithmetic without the multiply and the window folds, and a
    non-overlapping grid is only rearranged.

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

    if weight == "uniform" and sh == ph and sw == pw:
        # Non-overlapping grid under the uniform window. The fold would add
        # each value, times 1, into a zero-filled buffer and divide by a
        # count of 1, so the whole computation is 0.0 + x: x itself, except
        # that -0.0 comes out as +0.0 and a NaN comes out quiet. Adding 0.0
        # to the rearranged patches does exactly that to the returned values
        # in eager mode, without the fold, which at 3x2048x2048 is most of the
        # call. Derivatives pass through unchanged, and torch.compile folds the
        # addition away, so there a -0.0 and a signaling NaN come out as given.
        grid = patches.to(accum_dtype).reshape(num_h, num_w, c, ph, pw)
        out = grid.permute(2, 0, 3, 1, 4).reshape(c, h, w)
        return (out + 0.0).to(patches.dtype)

    if weight == "uniform":
        # Overlapping grid under the uniform window. Every weight is 1, so the
        # weighted fold is the plain fold, in the same order, and the
        # separable denominator is the count map: exactly the arithmetic
        # reconstruct performs, without the multiply by ones and the two
        # window folds. Bit-identical to the general path below, values,
        # signed zeros, NaN payloads and gradients included; the speedup is
        # measured in the 0.6.0 entry of CHANGELOG.md.
        return reconstruct(patches, image_shape, stride)

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
    # Both operands are fresh tensors this function owns, so the clamp and the
    # division run in place instead of allocating an (H, W) copy of the
    # denominator and a (C, H, W) quotient.
    return numerator.div_(den.clamp_min_(torch.finfo(den.dtype).tiny)).to(patches.dtype)
