"""Which tensors the native kernel may take (THEORY §9.2, §9.9; I6).

The kernel reads and writes through raw pointers, so it is correct only for a
plain eager CPU tensor, and everything else must reach the torch path, which
handles it as torch does. 0.6.0 checked only ``requires_grad`` and a
``data_ptr()`` that raises, and each test below pins an input that slipped
through, measured on 0.6.0 at torch 2.6.0 and 2.14.1:

- ``torch.func.functionalize``, ``__torch_dispatch__`` wrapper subclasses and
  the fake tensors of a non-strict ``torch.export`` at torch 2.6 answer
  ``data_ptr()`` with a null pointer instead of raising, and the kernel read
  and wrote through it: the interpreter died with an access violation. Those
  run in a child process, so a crash is reported instead of ending the run.
- ``torch.jit.trace`` and ``make_fx``, with and without ``pre_dispatch``,
  record the ops they see, and the kernel's writes are not ops: the recorded
  program allocated the output and divided it without filling it, so a replay
  returned stale memory.
- A ``__torch_function__`` subclass came back as a plain Tensor with the
  kernel and as the subclass without it.
- A tensor with the negative bit set keeps its un-negated values in memory,
  and the kernel folded those, with the wrong sign.
- A tensor viewed from a byte buffer at an odd offset is plain and contiguous
  but misaligned, and the kernel read through its pointer, which Rust leaves
  undefined.
- Under ``torch.compile``, ``torch.vmap`` raised with the kernel: Dynamo let
  the data_ptr error escape the check that should have declined it.

The last tests guard the other way: a plain tensor must still reach the
kernel, and without the accelerator, absent or failing to load, patchcraft
must import and its pure path must compile as one graph from the first call
of a process.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from collections.abc import Callable

import pytest
import torch

from patchcraft import _accel, extract, reconstruct, stitch
from tests._rng import bit_equal, rand_image

real_accel = pytest.mark.skipif(
    not _accel.accel_available(), reason="native accelerator not built into this install"
)

_OPS = ["reconstruct", "uniform", "hann", "gaussian"]


@pytest.fixture(params=["1", "0"], ids=["accel", "pure"])
def accel(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("PATCHCRAFT_ACCEL", request.param)
    return str(request.param)


def _fold(op: str, shape: tuple[int, int, int] = (3, 16, 16), stride: int = 2
          ) -> Callable[[torch.Tensor], torch.Tensor]:
    if op == "reconstruct":
        return lambda p: reconstruct(p, shape, stride)
    return lambda p: stitch(p, shape, stride, weight=op)  # type: ignore[arg-type]


def _patches(seed: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(49, 3, 4, 4, generator=gen, dtype=dtype)


def _pure(monkeypatch: pytest.MonkeyPatch, accel: str,
          fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    monkeypatch.setenv("PATCHCRAFT_ACCEL", "0")
    try:
        return fn(x)
    finally:
        monkeypatch.setenv("PATCHCRAFT_ACCEL", accel)


def _count_kernel_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Route the native module through a proxy that records each fold_add."""
    native = _accel._load()
    assert native is not None
    calls: list[int] = []

    class Counting:
        _ABI_VERSION = native._ABI_VERSION

        def fold_add(self, *args: object) -> None:
            calls.append(1)
            native.fold_add(*args)

    monkeypatch.setattr(_accel, "_module", Counting())
    return calls


def _child(code: str, accel: str | None = None) -> list[str]:
    """Run ``code`` in a fresh interpreter, with PATCHCRAFT_ACCEL set to
    ``accel`` or unset, and return its result lines. Each case prints a RUN
    line before it starts, so a crash names the case that killed the child."""
    env = {k: v for k, v in os.environ.items() if k != "PATCHCRAFT_ACCEL"}
    if accel is not None:
        env["PATCHCRAFT_ACCEL"] = accel
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600, env=env
    )
    last = [line for line in proc.stdout.splitlines() if line.startswith("RUN")][-1:]
    assert proc.returncode == 0, (
        f"the child died with exit code {proc.returncode} during {last}: {proc.stderr[-600:]}"
    )
    return [line for line in proc.stdout.splitlines() if not line.startswith("RUN")]


# Importing patchcraft must not import Dynamo, which takes seconds; the check
# compares with the state after importing torch, so a torch that imports it
# itself does not fail the test.
_IMPORT_CHECK = """
import sys
import torch
dynamo_before = "torch._dynamo" in sys.modules
import patchcraft
print("import", "EQUAL" if ("torch._dynamo" in sys.modules) == dynamo_before else "DIFFERENT")
"""

# Every null-pointer and misaligned case runs in one child process, so the
# suite pays for one interpreter start, which importing torch makes the slow
# part. It starts with PATCHCRAFT_ACCEL=0, under which importing patchcraft
# must leave the extension unloaded, as SECURITY.md promises.
_CHILD = _IMPORT_CHECK + textwrap.dedent(
    """
    import os
    print("unloaded", "EQUAL" if "patchcraft._accel_native" not in sys.modules else "DIFFERENT")
    from torch.utils._pytree import tree_map
    import patchcraft._accel_native as native
    from patchcraft import reconstruct, stitch

    class Wrapper(torch.Tensor):
        # The documented shape of a __torch_dispatch__ wrapper subclass.
        @staticmethod
        def __new__(cls, elem):
            return torch.Tensor._make_wrapper_subclass(
                cls, elem.shape, dtype=elem.dtype, device=elem.device, strides=elem.stride())

        def __init__(self, elem):
            self.elem = elem

        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            unwrap = lambda t: t.elem if isinstance(t, Wrapper) else t
            wrap = lambda t: Wrapper(t) if isinstance(t, torch.Tensor) else t
            return tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs or {})))

    # The entry point refuses a pointer it may not build a slice from before
    # reading anything; without the check each call reads unmapped memory.
    # One bad pointer per case, so each of the six conditions is pinned alone.
    null, misaligned = "null data pointer", "misaligned data pointer"
    native_cases = [
        ("native-null-patches", (0, 4, 1, 1, 1, 1, 1, 1, 1, 1, None, "f32"), null),
        ("native-null-out", (4, 0, 1, 1, 1, 1, 1, 1, 1, 1, None, "f32"), null),
        ("native-null-kernel", (4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 0, "f32"), null),
        ("native-misaligned-patches", (1, 4, 1, 1, 1, 1, 1, 1, 1, 1, None, "f32"), misaligned),
        ("native-misaligned-out", (4, 2, 1, 1, 1, 1, 1, 1, 1, 1, None, "f32"), misaligned),
        ("native-misaligned-kernel", (4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 2, "f32"), misaligned),
        ("native-misaligned-f64", (4, 8, 1, 1, 1, 1, 1, 1, 1, 1, None, "f64"), misaligned),
    ]
    for name, args, message in native_cases:
        print("RUN", name, flush=True)
        try:
            native.fold_add(*args)
            print(name, "DIFFERENT (no error)", flush=True)
        except ValueError as e:
            print(name, "EQUAL" if str(e) == message else f"DIFFERENT ({e})", flush=True)

    cases = [(family, op, s)
             for family in ("functionalize", "wrapper-subclass")
             for op in ("reconstruct", "uniform", "hann", "gaussian") for s in (2, 4)]
    cases.append(("export", "hann", 2))
    for family, op, stride in cases:
        print("RUN", family, op, stride, flush=True)
        n = ((16 - 4) // stride + 1) ** 2
        p = torch.rand(n, 3, 4, 4, generator=torch.Generator().manual_seed(5))
        fresh = torch.rand(n, 3, 4, 4, generator=torch.Generator().manual_seed(6))

        def fold(x, op=op, stride=stride):
            if op == "reconstruct":
                return reconstruct(x, (3, 16, 16), stride)
            return stitch(x, (3, 16, 16), stride, weight=op)

        class Fold(torch.nn.Module):
            def forward(self, x):
                return fold(x)

        os.environ["PATCHCRAFT_ACCEL"] = "0"
        want = fold(fresh if family == "export" else p)
        os.environ["PATCHCRAFT_ACCEL"] = "1"
        if family == "functionalize":
            got = torch.func.functionalize(fold)(p)
        elif family == "export":
            got = torch.export.export(Fold(), (p,), strict=False).module()(fresh)
        else:
            got = fold(Wrapper(p)).elem
        print(family, op, stride, "EQUAL" if torch.equal(got, want) else "DIFFERENT", flush=True)
    """
)


@real_accel
def test_no_null_or_misaligned_pointer_reaches_the_kernel() -> None:
    """functionalize, a wrapper subclass and a non-strict export hand out null
    pointers, and the kernel's own entry point refuses a null or misaligned one
    with ValueError."""
    results = _child(_CHILD, accel="0")
    assert len(results) == 26 and all(line.endswith("EQUAL") for line in results), results


# An install whose accelerator cannot load: its import raises RuntimeError, as
# a broken build can, and patchcraft must still import and work without it.
# Its first patchcraft calls are traced ones. The lookup used to happen inside
# the first call, and Dynamo cannot trace an import, so fullgraph=True raised
# until an eager call had made it, and a strict torch.export, which traces with
# Dynamo too, raised the same way.
_BROKEN_EXTENSION = textwrap.dedent(
    """
    import importlib.abc
    import sys

    class Broken(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "patchcraft._accel_native":
                raise RuntimeError("simulated failure while loading the native module")
            return None

    sys.meta_path.insert(0, Broken())
    """
)
_PURE_CHILD = (
    _BROKEN_EXTENSION + _IMPORT_CHECK
    + textwrap.dedent(
        """
        from patchcraft import reconstruct, stitch

        # The compiled reconstruct is the first patchcraft call of the process;
        # nothing may look the accelerator up before it.
        p = torch.rand(49, 3, 4, 4, generator=torch.Generator().manual_seed(7))
        for op in ("reconstruct", "hann"):
            print("RUN", op, flush=True)

            def fold(x, op=op):
                if op == "reconstruct":
                    return reconstruct(x, (3, 16, 16), 2)
                return stitch(x, (3, 16, 16), 2, weight=op)

            got = torch.compile(fold, fullgraph=True, backend="eager")(p)
            print(op, "EQUAL" if torch.equal(got, fold(p)) else "DIFFERENT", flush=True)
        print("absent", "EQUAL" if not patchcraft.accel_available() else "DIFFERENT")
        """
    )
)


def test_without_the_accelerator_a_first_call_compiles_as_one_graph() -> None:
    results = _child(_PURE_CHILD)
    assert results == ["import EQUAL", "reconstruct EQUAL", "hann EQUAL", "absent EQUAL"], results


@pytest.mark.parametrize("pre_dispatch", [False, True], ids=["dispatch", "pre-dispatch"])
@pytest.mark.parametrize("op", _OPS)
def test_a_make_fx_program_replays_on_new_patches(
    accel: str, op: str, pre_dispatch: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """make_fx sits on the dispatch stack, and with pre_dispatch=True on the
    pre-dispatch stack only, which the kernel let through until both counted."""
    from torch.fx.experimental.proxy_tensor import make_fx

    fold = _fold(op)
    program = make_fx(fold, tracing_mode="real", pre_dispatch=pre_dispatch)(_patches(1))
    fresh = _patches(2)
    assert bit_equal(program(fresh), _pure(monkeypatch, accel, fold, fresh))


@pytest.mark.parametrize("op", _OPS)
def test_a_jit_trace_replays_on_new_patches(
    accel: str, op: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fold = _fold(op)
    traced = torch.jit.trace(fold, (_patches(1),), check_trace=False)
    fresh = _patches(2)
    assert bit_equal(traced(fresh), _pure(monkeypatch, accel, fold, fresh))


class _Tagged(torch.Tensor):
    """A __torch_function__ subclass with real storage, like most user tags."""


@pytest.mark.parametrize("op", _OPS)
def test_a_tensor_subclass_comes_back_as_itself(
    accel: str, op: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The round trip of a subclassed image keeps the class on both paths, and
    returns the image where every count is 1, 2 or 4."""
    x = rand_image(3, 16, 16, torch.float32, seed=91).as_subclass(_Tagged)
    patches = extract(x, 4, 2)
    fold = _fold(op)
    got = fold(patches)
    want = _pure(monkeypatch, accel, fold, patches)
    assert type(got) is type(want) is _Tagged
    plain_got = got.as_subclass(torch.Tensor)
    assert bit_equal(plain_got, want.as_subclass(torch.Tensor))
    if op in ("reconstruct", "uniform"):
        assert bit_equal(plain_got, x.as_subclass(torch.Tensor))


@pytest.mark.parametrize("op", _OPS)
def test_a_negative_view_folds_its_values(accel: str, op: str) -> None:
    base = _patches(83)
    view = torch._neg_view(base)
    assert view.is_neg() and view.is_contiguous()
    fold = _fold(op)
    assert bit_equal(fold(view), fold(view.resolve_neg()))


@pytest.mark.parametrize("weight", ["hann", "gaussian"])
def test_a_conjugate_imaginary_part_keeps_its_sign(accel: str, weight: str) -> None:
    """The same defect through public API alone: z.conj().imag has the negative bit."""
    z = torch.tensor([[[[0.25 + 0.75j]]]], dtype=torch.complex64)
    patches = z.conj().imag
    assert patches.is_neg() and patches.is_contiguous()
    assert stitch(patches, (1, 1, 1), 1, weight=weight).item() == -0.75  # type: ignore[arg-type]


@real_accel
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
@pytest.mark.parametrize("op", _OPS)
def test_a_misaligned_tensor_takes_the_torch_path(
    op: str, dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A view of a byte buffer at offset 1 is a plain contiguous tensor whose
    pointer sits off its dtype's alignment. It folds to the bits of an aligned
    copy, on the torch path, and only the aligned copy reaches the kernel."""
    monkeypatch.setenv("PATCHCRAFT_ACCEL", "1")
    want = _patches(97, dtype)
    size = want.element_size()
    buffer = bytearray(want.numel() * size + 1)
    patches = torch.frombuffer(buffer, dtype=dtype, offset=1).view(want.shape)
    patches.copy_(want)
    assert type(patches) is torch.Tensor and patches.is_contiguous()
    assert patches.data_ptr() % size == 1
    calls = _count_kernel_calls(monkeypatch)
    fold = _fold(op)
    got = fold(patches)
    assert calls == []
    assert bit_equal(got, fold(want))
    assert calls == [1]


@pytest.mark.parametrize("order", ["compile-of-vmap", "vmap-of-compile"])
@pytest.mark.parametrize("op", ["reconstruct", "hann"])
def test_vmap_under_compile_equals_a_loop(accel: str, op: str, order: str) -> None:
    gen = torch.Generator().manual_seed(85)
    batch = torch.rand(2, 49, 3, 4, 4, generator=gen)
    fold = _fold(op)
    want = torch.stack([fold(p) for p in batch])
    torch._dynamo.reset()
    if order == "compile-of-vmap":
        got = torch.compile(torch.vmap(fold), backend="eager")(batch)
    else:
        got = torch.vmap(torch.compile(fold, backend="eager"))(batch)
    assert bit_equal(got, want)


@real_accel
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
@pytest.mark.parametrize("op", _OPS)
def test_a_plain_tensor_still_reaches_the_kernel(
    op: str, dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every check above declines; this one makes sure a plain tensor is not
    declined too, which no output comparison could tell."""
    monkeypatch.setenv("PATCHCRAFT_ACCEL", "1")
    calls = _count_kernel_calls(monkeypatch)
    _fold(op)(_patches(87, dtype))
    assert calls == [1]


@pytest.mark.parametrize("op", ["reconstruct", "hann"])
def test_the_pure_path_compiles_as_one_graph(op: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """With the accelerator switched off nothing leaves the trace, so fullgraph holds."""
    monkeypatch.setenv("PATCHCRAFT_ACCEL", "0")
    fold = _fold(op)
    torch._dynamo.reset()
    compiled = torch.compile(fold, fullgraph=True, backend="eager")
    patches = _patches(89)
    assert bit_equal(compiled(patches), fold(patches))
