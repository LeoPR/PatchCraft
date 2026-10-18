"""Bridge to the optional Rust accelerator (``patchcraft._accel_native``).

The accelerator is a pyo3/Rust extension module compiled into this wheel on
the platforms that have one. Where it is absent the wheel is pure Python and
every path here reports unavailable. This module detects it at runtime and
exposes a single primitive, :func:`fold_weighted`, for the overlap paths of
``reconstruct`` and ``stitch``. Any failure at all, whether the extension was
never built, the ABI mismatches, ``PATCHCRAFT_ACCEL=0`` is set in the
environment or the tensor is ineligible, yields ``None``/``False`` and the
caller falls back to the pure-torch path. Nothing here raises for an
unavailable accelerator.
"""
from __future__ import annotations

import importlib.util
import os
from collections.abc import Callable
from types import ModuleType
from typing import Any

import torch
from torch.autograd import forward_ad

__all__ = ["accel_available", "fold_weighted"]

_ABI_VERSION_REQUIRED = 1

_module: ModuleType | None = None
_checked = False

# torch exposes the depth of its two dispatch-mode stacks only under private
# names, read with getattr: both are present from the declared floor (2.6) to
# the newest tested, and a torch without one leaves the recorders that sit on
# it unguarded. make_fx in real mode sits on the first stack, and make_fx with
# pre_dispatch=True on the second only.
_dispatch_stack_lens: tuple[Callable[[], int], ...] = tuple(
    stack_len
    for stack_len in (
        getattr(torch._C, "_len_torch_dispatch_stack", None),
        getattr(torch._ops, "_len_torch_dispatch_stack_pre_dispatch", None),
    )
    if stack_len is not None
)


def _load() -> ModuleType | None:
    """Import the native module once, cache the outcome, return it or None."""
    global _checked, _module
    if not _checked:
        _checked = True
        try:
            mod = importlib.import_module("patchcraft._accel_native")
        except Exception:
            # Any failure, not only a missing file: since this runs at import,
            # an extension that raises while it loads would otherwise make
            # `import patchcraft` itself fail.
            pass
        else:
            if getattr(mod, "_ABI_VERSION", None) == _ABI_VERSION_REQUIRED:
                _module = mod
    return _module


def _env_disabled() -> bool:
    """Evaluated per call so tests/debugging can toggle without cache resets."""
    return os.environ.get("PATCHCRAFT_ACCEL", "1") == "0"


def accel_available() -> bool:
    """Whether the overlap paths will use the native accelerator."""
    return not _env_disabled() and _load() is not None


def _declined(t: torch.Tensor) -> bool:
    """Whether ``t``'s metadata alone rules the kernel out.

    The kernel writes its sum through a raw pointer into a fresh tensor, so
    whatever torch would attach to the result is lost: the autograd graph of a
    tensor that requires grad, and the class of a subclass. A tensor with the
    negative bit set keeps its un-negated values in memory, which the kernel
    would read as they are.
    """
    return type(t) is not torch.Tensor or t.requires_grad or t.is_neg()


def _recorded() -> bool:
    """Whether something records or intercepts the torch ops of this call.

    A recorder sees ops, and the kernel's writes are not ops: under
    ``torch.jit.trace`` and ``make_fx`` the recorded program allocates the
    output and divides it without ever filling it, so a replay returns
    whatever the allocator left there. ``make_fx`` and every other dispatch
    mode sit on one of the two dispatch stacks.
    """
    # torch.jit exports is_tracing without listing it, and without annotations.
    if torch.jit.is_tracing():  # type: ignore[attr-defined,no-untyped-call]
        return True
    return any(stack_len() > 0 for stack_len in _dispatch_stack_lens)


def _has_tangent(t: torch.Tensor) -> bool:
    """Whether ``t`` is a forward-mode dual at the current level.

    torch allows one forward-AD level at a time, and ``torch.func.jvp`` nests
    through wrappers without storage, which are declined before this is asked.
    """
    return forward_ad.unpack_dual(t).tangent is not None  # type: ignore[no-untyped-call]


def _fold_weighted_eager(
    patches: torch.Tensor,
    image_shape: tuple[int, int, int],
    stride: tuple[int, int],
    kernel: torch.Tensor | None,
) -> torch.Tensor | None:
    """The body of :func:`fold_weighted`, run on real tensors."""
    mod = None if _env_disabled() else _load()
    if mod is None:
        return None
    if patches.device.type != "cpu":
        return None
    if patches.dtype not in (torch.float32, torch.float64):
        return None
    if kernel is not None and (
        kernel.dtype != patches.dtype or kernel.device != patches.device
    ):
        return None
    if _declined(patches) or (kernel is not None and _declined(kernel)):
        return None
    if _recorded():
        return None

    c, h, w = image_shape
    n, _c, ph, pw = patches.shape
    sh, sw = stride

    work = patches.contiguous()
    kern = kernel.contiguous() if kernel is not None else None
    out = torch.empty((c, h, w), dtype=work.dtype, device=work.device)
    if work.numel() == 0 or out.numel() == 0:
        return None
    try:
        work_ptr = work.data_ptr()
        out_ptr = out.data_ptr()
        kern_ptr = kern.data_ptr() if kern is not None else None
    except RuntimeError:
        # A tensor without storage, which is what torch.vmap and most of the
        # torch.func transforms pass in. It is ineligible like any other, so
        # the caller's pure-torch path handles it.
        return None
    # torch.func.functionalize, FakeTensorMode and wrapper subclasses answer
    # data_ptr() with a null pointer instead of raising, and a tensor viewed
    # from a byte buffer at an offset, such as torch.frombuffer(..., offset=1)
    # makes, can start off its dtype's alignment. The kernel may read through
    # neither.
    size = work.element_size()
    pointers = (work_ptr, out_ptr) if kern_ptr is None else (work_ptr, out_ptr, kern_ptr)
    if any(ptr == 0 or ptr % size for ptr in pointers):
        return None
    # Asked only now, of a tensor known to have storage: inside a dual level,
    # unpack_dual raises on a torch.vmap tensor, which has no batching rule for
    # it, and the pointer read above has already sent those to torch.
    if _has_tangent(patches) or (kernel is not None and _has_tangent(kernel)):
        return None

    mod.fold_add(
        work_ptr,
        out_ptr,
        n,
        c,
        ph,
        pw,
        h,
        w,
        sh,
        sw,
        kern_ptr,
        "f32" if work.dtype == torch.float32 else "f64",
    )
    # `work` and `kern` must stay alive until the native call returns; it is
    # synchronous and both are still local references here, so this holds.
    return out


# Every check above asks the real tensor a question, and Dynamo answers some of
# them from its fake tensors: at torch 2.6 a dual's tangent reads as None, and
# under vmap the data_ptr error escapes the try. Under torch.compile the body
# therefore runs outside the trace, where the answers are the eager ones.
# torch.compiler.disable imports Dynamo when it wraps, which would cost every
# process that imports patchcraft seconds; torch._disable_dynamo wraps the same
# way and imports Dynamo at the first call, which only a compiled call makes.
# The private name is read with getattr, like the stacks above, and a torch
# without it gets the public one.
_disable: Any = getattr(torch, "_disable_dynamo", None) or torch.compiler.disable
_fold_weighted_outside_trace = _disable(_fold_weighted_eager)


def fold_weighted(
    patches: torch.Tensor,
    image_shape: tuple[int, int, int],
    stride: tuple[int, int],
    kernel: torch.Tensor | None,
) -> torch.Tensor | None:
    """Sum ``patches`` into a ``(C, H, W)`` image, optionally kernel-weighted.

    ``out[c, y, x] = sum_p patches[p, c, y - row_p, x - col_p] * kernel[...]``
    over the patches covering ``(y, x)``, which is the numerator of the overlap fold,
    before the count/denominator division the caller performs. Summation per
    pixel is in descending patch index (= ascending kernel offset), matching
    ATen col2im's per-pixel order, the one order that is bit-exact against
    ``F.fold``.

    Only a plain CPU tensor reaches the kernel; for anything else this returns
    ``None`` and the caller takes its pure-torch path, which handles it as
    torch does. Declined: the accelerator absent or disabled, a device other
    than the CPU, a dtype outside float32/float64, a mismatched kernel, a
    tensor that requires grad or is a forward-mode dual, a tensor subclass, a
    tensor with the negative bit set, a call under ``torch.jit.trace`` or a
    dispatch mode such as ``make_fx``, and a tensor whose storage cannot be
    read through an aligned pointer (``torch.vmap`` and most ``torch.func``
    transforms pass in tensors that raise for it; ``torch.func.functionalize``,
    ``FakeTensorMode`` and wrapper subclasses answer with a null one; a view of
    a byte buffer at an offset can be misaligned). Geometry is NOT validated
    here, because callers validate with ``check_fold_geometry`` first.
    """
    # Without the accelerator nothing leaves the trace, so the pure path stays
    # traceable end to end (strict torch.export, fullgraph=True); the lookup is
    # settled at import, by _lookup_at_import below, so this holds from the
    # first call.
    if _env_disabled() or _load() is None:
        return None
    if torch.compiler.is_compiling():
        result: torch.Tensor | None = _fold_weighted_outside_trace(
            patches, image_shape, stride, kernel
        )
        return result
    return _fold_weighted_eager(patches, image_shape, stride, kernel)


def _lookup_at_import() -> None:
    """Settle the lookup while ``patchcraft`` is imported, outside any trace.

    Dynamo cannot trace an import, so a lookup made by a compiled first call
    made strict torch.export and fullgraph=True fail on the pure path. Under
    PATCHCRAFT_ACCEL=0 the extension stays unloaded, as it always has, but its
    file is looked for without loading it: an install without one records the
    absence now, so a process that turns the variable on later still finds the
    answer cached. A file that is there but fails to load is found out only at
    the first call after the switch, as before.
    """
    global _checked
    if not _env_disabled():
        _load()
        return
    try:
        absent = importlib.util.find_spec("patchcraft._accel_native") is None
    except Exception:
        absent = True
    if absent:
        _checked = True


_lookup_at_import()
