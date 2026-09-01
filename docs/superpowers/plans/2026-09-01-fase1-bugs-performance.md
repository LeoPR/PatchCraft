# Fase 1 — Bugs + Performance pura torch — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the small bugs found in the 0.2.2 audit and land the pure-torch performance fast paths (extract / reconstruct / stitch / metrics) with bit-exact equivalence, shipping as 0.3.0.

**Architecture:** Spec: `docs/superpowers/specs/2026-09-01-fase1-bugs-performance-design.md`. Approach A: internal fast paths + closed-form geometry maps, no API change, no cache, no new dependencies. The fold-geometry validation duplicated in `reconstruct`/`stitch` is first centralized in a private helper, then both functions get their fast paths on top.

**Tech Stack:** Python 3.12+, torch >= 2.6, pytest, ruff, mypy (strict). Windows dev machine: venv binaries live at `.venv/Scripts/python`.

## Global Constraints

- No public API change: `__all__` in `src/patchcraft/__init__.py` unchanged; no signature changes.
- Equivalence bar per spec §2: `extract` and `reconstruct` bit-exact vs current implementation; `stitch` uniform bit-exact; `stitch` hann/gaussian `torch.testing.assert_close` default tolerances; `metrics` bit-exact.
- All existing 346 tests must pass **unmodified** (proves non-regression).
- `ruff check src tests` and `python -m mypy src/patchcraft` clean after every task. Line length 100.
- Error messages of `reconstruct`/`stitch` validation must remain byte-identical (tests assert on them) except for the single-op-name slot.
- fp16/bf16 promotion to f32 accumulator preserved in the overlap paths.
- Docs: factual corrections only (no restructuring, no cuts).
- Commits: one per task, message style follows `git log` (lowercase conventional: `fix:`, `perf:`, `refactor:`, `docs:`, `chore:`).

---

### Task 1: resize — uint8 wrap fix + dtype-check tidy

**Files:**
- Modify: `src/patchcraft/resize.py` (`_tensor_to_pil_u8` lines 52-69; `_resize_torch` line 125; `resize` line 183)
- Test: `tests/test_resize.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `_tensor_to_pil_u8(tensor: torch.Tensor) -> PILImage` now clamps integer inputs to [0, 255] before the uint8 cast.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_resize.py`:

```python
def test_tensor_to_pil_int_tensor_clamps_instead_of_wrapping():
    """int16 values outside [0, 255] must clamp, not wrap (300 -> 255, not 44)."""
    t = torch.tensor([[[300, -5], [100, 255]]], dtype=torch.int16)
    out = resize(t, (2, 2), backend="pil")
    assert out.dtype == torch.int16
    assert out[0, 0, 0].item() == 255  # would be 44 with a bare astype(uint8)
    assert out[0, 0, 1].item() == 0    # would be 251 with a bare astype(uint8)
    assert out[0, 1, 0].item() == 100
    assert out[0, 1, 1].item() == 255
```

Note: `resize` with `backend="pil"` on an int tensor goes tensor → PIL uint8 → resize (2,2) is a no-op size-wise if the tensor is already (1, 2, 2) → back. Check imports at the top of `tests/test_resize.py`; if `resize` and `torch` are already imported, no new imports needed.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_resize.py::test_tensor_to_pil_int_tensor_clamps_instead_of_wrapping -v`
Expected: FAIL (`out[0,0,0]` is 44, the wrapped value).

- [ ] **Step 3: Implement the fix**

In `src/patchcraft/resize.py`, `_tensor_to_pil_u8`, replace:

```python
    if tensor.is_floating_point():
        arr = (tensor.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
    else:
        arr = tensor.cpu().numpy().astype(np.uint8)
```

with:

```python
    if tensor.is_floating_point():
        arr = (tensor.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
    else:
        # Integer tensors: clamp to the uint8 range before casting, otherwise
        # values outside [0, 255] wrap (300 -> 44) -- same defect class as the
        # 0.2.0 bicubic overshoot wrap on the torch backend.
        arr = tensor.clamp(0, 255).cpu().numpy().astype(np.uint8)
```

- [ ] **Step 4: dtype-check tidy (no behavior change)**

In `src/patchcraft/resize.py` line 125, replace:

```python
    if not torch.empty(0, dtype=original_dtype).is_floating_point():
```

with:

```python
    if not original_dtype.is_floating_point:
```

and line 183, replace:

```python
            if not torch.empty(0, dtype=original_dtype).is_floating_point():
```

with:

```python
            if not original_dtype.is_floating_point:
```

(`torch.dtype.is_floating_point` is a plain attribute; no empty-tensor allocation needed.)

- [ ] **Step 5: Run tests, ruff, mypy**

Run: `.venv/Scripts/python -m pytest tests/test_resize.py -q && .venv/Scripts/python -m ruff check src tests && .venv/Scripts/python -m mypy src/patchcraft`
Expected: all pass; the new test passes.

- [ ] **Step 6: Commit**

```bash
git add src/patchcraft/resize.py tests/test_resize.py
git commit -m "fix: clamp integer tensors to uint8 range in the PIL resize backend"
```

---

### Task 2: extract — `Tensor.unfold` fast path for dilation == 1

**Files:**
- Modify: `src/patchcraft/extract.py` (module docstring line 1; `extract` lines 53-66)
- Test: `tests/test_extract.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `extract(image, patch_size, stride, dilation=1) -> Tensor[L, C, ph, pw]`, unchanged contract (row-major, independent memory, dtype/device preserved).

**Note on TDD for this task:** the equivalence tests are characterization tests — they must pass on the *current* implementation too. Write them, run them against current code (they pass, pinning behavior), then swap the implementation and run again.

- [ ] **Step 1: Record the "before" benchmark**

Run: `.venv/Scripts/python lab/bench_quick.py > lab/bench_before.txt 2>&1`
Expected: file with the baseline table (extract non-overlap ~2.2 ms, overlap ~9.5 ms).

- [ ] **Step 2: Write the characterization + aliasing tests**

Append to `tests/test_extract.py` (check the file's existing imports first; `torch.nn.functional as F`, `torch`, `pytest`, `extract` are expected — add whichever is missing):

```python
import torch.nn.functional as F  # noqa: N812  (add only if missing)


def _unfold_reference(image, ph, pw, sh, sw, dh, dw):
    """The pre-0.3.0 extract implementation, as ground truth."""
    c = image.shape[0]
    unfolded = F.unfold(
        image.unsqueeze(0), kernel_size=(ph, pw), dilation=(dh, dw), stride=(sh, sw)
    )
    return unfolded[0].view(c, ph, pw, -1).permute(3, 0, 1, 2).contiguous()


@pytest.mark.parametrize("c,h,w", [(1, 7, 5), (3, 16, 16), (4, 9, 31), (3, 64, 48)])
@pytest.mark.parametrize(
    "ph,pw,sh,sw,dh,dw",
    [
        (2, 2, 2, 2, 1, 1),   # exact non-overlap
        (4, 4, 2, 2, 1, 1),   # overlap 50%
        (3, 5, 1, 2, 1, 1),   # rectangular, anisotropic stride
        (1, 1, 1, 1, 1, 1),   # degenerate contiguous window
        (2, 2, 1, 1, 2, 2),   # dilation -> slow path
        (4, 4, 3, 3, 1, 1),   # stride < patch, non-divisor
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
def test_extract_matches_unfold_reference(c, h, w, ph, pw, sh, sw, dh, dw, dtype):
    img = torch.randn(c, h, w, dtype=dtype)
    got = extract(img, (ph, pw), (sh, sw), (dh, dw))
    ref = _unfold_reference(img, ph, pw, sh, sw, dh, dw)
    assert got.shape == ref.shape
    assert torch.equal(got, ref)


def test_extract_output_does_not_alias_image():
    """Patch memory must be independent: mutating patches never touches the image.

    patch_size == stride == 1 is the degenerate case where a naive reshape
    over the unfold view returns a view onto the image itself."""
    img = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    patches = extract(img, 1, 1)
    patches[0, 0, 0, 0] = -1.0
    assert img[0, 0, 0].item() == 0.0
```

- [ ] **Step 3: Run the new tests against the current implementation**

Run: `.venv/Scripts/python -m pytest tests/test_extract.py -q -k "matches_unfold_reference or does_not_alias"`
Expected: all PASS (they pin current behavior).

- [ ] **Step 4: Implement the fast path**

In `src/patchcraft/extract.py`, replace the tail of `extract` (lines 53-66):

```python
    c, h, w = image.shape
    eff_h = dh * (ph - 1) + 1
    eff_w = dw * (pw - 1) + 1

    if h < eff_h or w < eff_w:
        return torch.empty(0, c, ph, pw, dtype=image.dtype, device=image.device)

    unfolded = F.unfold(
        image.unsqueeze(0),
        kernel_size=(ph, pw),
        dilation=(dh, dw),
        stride=(sh, sw),
    )
    return unfolded[0].view(c, ph, pw, -1).permute(3, 0, 1, 2).contiguous()
```

with:

```python
    c, h, w = image.shape
    eff_h = dh * (ph - 1) + 1
    eff_w = dw * (pw - 1) + 1

    if h < eff_h or w < eff_w:
        return torch.empty(0, c, ph, pw, dtype=image.dtype, device=image.device)

    if dh == 1 and dw == 1:
        # Fast path: strided window view + one copy. Measured 13-21x faster
        # than F.unfold (im2col) on CPU, bit-exact (same pixels, same order).
        nh = (h - ph) // sh + 1
        nw = (w - pw) // sw + 1
        windows = image.unfold(1, ph, sh).unfold(2, pw, sw)  # (C, nh, nw, ph, pw)
        out = windows.permute(1, 2, 0, 3, 4).reshape(nh * nw, c, ph, pw)
        if out.untyped_storage().data_ptr() == image.untyped_storage().data_ptr():
            # Degenerate geometry (e.g. ph == pw == 1): the permuted window is
            # contiguous, reshape returned a view aliasing the image, and the
            # contract promises independent memory.
            out = out.clone()
        return out

    unfolded = F.unfold(
        image.unsqueeze(0),
        kernel_size=(ph, pw),
        dilation=(dh, dw),
        stride=(sh, sw),
    )
    return unfolded[0].view(c, ph, pw, -1).permute(3, 0, 1, 2).contiguous()
```

Also update the module docstring first line from:

```python
"""Patch extraction via torch.nn.functional.unfold.
```

to:

```python
"""Patch extraction: strided-window fast path, F.unfold for dilation > 1.
```

- [ ] **Step 5: Run full extract tests + linters**

Run: `.venv/Scripts/python -m pytest tests/test_extract.py -q && .venv/Scripts/python -m ruff check src tests && .venv/Scripts/python -m mypy src/patchcraft`
Expected: all PASS (equivalence bit-exact on the whole grid, aliasing test passes).

- [ ] **Step 6: Commit**

```bash
git add src/patchcraft/extract.py tests/test_extract.py
git commit -m "perf: extract via strided-window view for dilation == 1 (13-21x on CPU)"
```

---

### Task 3: fold-geometry validation helper (pure refactor)

**Files:**
- Create: `src/patchcraft/_foldgeom.py`
- Modify: `src/patchcraft/reconstruct.py`, `src/patchcraft/stitch.py`
- Test: existing `tests/test_reconstruct.py`, `tests/test_stitch.py` (no changes)

**Interfaces:**
- Produces (consumed by Tasks 4-5):

```python
# src/patchcraft/_foldgeom.py
def check_fold_geometry(
    patches: torch.Tensor,
    image_shape: tuple[int, int, int],
    stride: int | tuple[int, int],
    dilation: int | tuple[int, int],
    *,
    op: str,
) -> tuple[int, int, int, int]:  # returns (h, w, num_h, num_w)
```

- [ ] **Step 1: Create the helper with the exact current validation**

Create `src/patchcraft/_foldgeom.py`:

```python
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
```

- [ ] **Step 2: Rewire `reconstruct` to the helper**

In `src/patchcraft/reconstruct.py`, delete lines 66-119 (the `image_shape` check through the `expected_n_patches` check, inclusive) and the now-unused `_as_pair` import if it becomes unused — check: `_as_pair` is no longer called in reconstruct after the move, so remove `from patchcraft.extract import _as_pair` and add `from patchcraft._foldgeom import check_fold_geometry`. After the dtype check (the `patches.is_floating_point()` raise, kept in place), insert:

```python
    h, w, num_h, num_w = check_fold_geometry(
        patches, image_shape, stride, dilation, op="reconstruct"
    )
```

- [ ] **Step 3: Rewire `stitch` to the helper**

In `src/patchcraft/stitch.py`, delete lines 141-196 (from `n_patches, c, ph, pw = patches.shape` through the `expected_n_patches` raise). Keep `from patchcraft.extract import _as_pair` only if still used — after the move it is not, so replace it with `from patchcraft._foldgeom import check_fold_geometry`. After the `weight not in _WEIGHT_KINDS` check, insert:

```python
    n_patches, c, ph, pw = patches.shape
    h, w, num_h, num_w = check_fold_geometry(
        patches, image_shape, stride, dilation, op="stitch"
    )
```

- [ ] **Step 4: Run the full suite + linters (behavior must be identical)**

Run: `.venv/Scripts/python -m pytest -q && .venv/Scripts/python -m ruff check src tests && .venv/Scripts/python -m mypy src/patchcraft`
Expected: 346 passed, 5 skipped; ruff/mypy clean. Any failure on error-message text means the helper drifted from the original — fix the helper, not the tests.

- [ ] **Step 5: Commit**

```bash
git add src/patchcraft/_foldgeom.py src/patchcraft/reconstruct.py src/patchcraft/stitch.py
git commit -m "refactor: centralize reconstruct/stitch fold-geometry validation"
```

---

### Task 4: reconstruct — non-overlap fast path + closed-form count map

**Files:**
- Modify: `src/patchcraft/reconstruct.py` (module docstring line 1; body from the accumulator comment onward)
- Test: `tests/test_reconstruct.py`

**Interfaces:**
- Consumes: `check_fold_geometry` from Task 3 (returns `(h, w, num_h, num_w)`).
- Produces: `reconstruct(patches, image_shape, stride, dilation=1) -> Tensor[C, H, W]`, unchanged contract.

Same TDD note as Task 2: characterization tests pass before and after; they pin behavior.

- [ ] **Step 1: Write the characterization tests**

Append to `tests/test_reconstruct.py` (add missing imports — `torch.nn.functional as F`, `pytest`):

```python
def _fold_reference(patches, image_shape, stride):
    """The pre-0.3.0 reconstruct implementation, as ground truth."""
    n_patches, c, ph, pw = patches.shape
    h, w = image_shape[1], image_shape[2]
    sh, sw = (stride, stride) if isinstance(stride, int) else stride
    accum_dtype = (
        torch.float32
        if patches.dtype in (torch.float16, torch.bfloat16)
        else patches.dtype
    )
    work = patches.to(accum_dtype)
    flat = work.permute(1, 2, 3, 0).reshape(c * ph * pw, n_patches).unsqueeze(0)
    folded = F.fold(flat, (h, w), (ph, pw), stride=(sh, sw))
    ones = torch.ones(1, ph * pw, n_patches, dtype=accum_dtype, device=patches.device)
    count = F.fold(ones, (h, w), (ph, pw), stride=(sh, sw))
    return (folded / count.clamp(min=1e-6))[0].to(patches.dtype)


@pytest.mark.parametrize("c,h,w", [(1, 8, 8), (3, 16, 16), (3, 32, 24), (4, 15, 15)])
@pytest.mark.parametrize(
    "ph,pw,sh,sw",
    [
        (2, 2, 2, 2),   # exact non-overlap -> fast path
        (4, 4, 4, 4),   # exact non-overlap, larger
        (4, 4, 2, 2),   # overlap 50%
        (3, 3, 1, 1),   # max overlap
        (4, 6, 2, 3),   # rectangular patch, anisotropic stride
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
def test_reconstruct_matches_fold_reference(c, h, w, ph, pw, sh, sw, dtype):
    num_h = (h - ph) // sh + 1
    num_w = (w - pw) // sw + 1
    if (num_h - 1) * sh + ph != h or (num_w - 1) * sw + pw != w:
        pytest.skip("geometry does not cover exactly")
    img = torch.randn(c, h, w, dtype=dtype)
    from patchcraft import extract
    patches = extract(img, (ph, pw), (sh, sw))
    got = reconstruct(patches, image_shape=(c, h, w), stride=(sh, sw))
    ref = _fold_reference(patches, (c, h, w), (sh, sw))
    assert got.shape == ref.shape
    assert got.dtype == ref.dtype
    assert torch.equal(got, ref)
```

- [ ] **Step 2: Run against current implementation**

Run: `.venv/Scripts/python -m pytest tests/test_reconstruct.py -q -k matches_fold_reference`
Expected: all PASS (pinning current behavior).

- [ ] **Step 3: Implement the fast path + closed-form count map**

In `src/patchcraft/reconstruct.py`, replace everything from the half-precision comment block through the final `return` with:

```python
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
    return (folded[0] / count).to(patches.dtype)
```

Update the module docstring first line from:

```python
"""Reconstruction of an image from its patches via torch.nn.functional.fold.
```

to:

```python
"""Reconstruction of an image from its patches.

Non-overlapping grids are a pure rearrangement; overlapping grids use
F.fold for the patches plus a closed-form O(H+W) count map.
"""
```

- [ ] **Step 4: Run tests + linters**

Run: `.venv/Scripts/python -m pytest -q && .venv/Scripts/python -m ruff check src tests && .venv/Scripts/python -m mypy src/patchcraft`
Expected: 346 + new pass; `torch.equal` holds on the whole grid.

- [ ] **Step 5: Commit**

```bash
git add src/patchcraft/reconstruct.py tests/test_reconstruct.py
git commit -m "perf: reconstruct without fold for non-overlap grids, closed-form count map"
```

---

### Task 5: stitch — separable denominator + gaussian docstring fix

**Files:**
- Modify: `src/patchcraft/stitch.py` (module docstring; `_window_kernel` area; body from the accumulator comment onward)
- Test: `tests/test_stitch.py`

**Interfaces:**
- Consumes: `check_fold_geometry` from Task 3.
- Produces: `stitch(patches, image_shape, stride, *, weight="uniform", dilation=1) -> Tensor[C, H, W]`, unchanged contract. New private helpers:

```python
def _window_1d(kind: WeightKind, n: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor
def _fold_window_1d(w1d: torch.Tensor, length: int, num: int, step: int) -> torch.Tensor
```

- [ ] **Step 1: Write the characterization + invariant tests**

Append to `tests/test_stitch.py` (add missing imports — `torch.nn.functional as F`, `pytest`, `math` if used):

```python
def _stitch_reference(patches, image_shape, stride, weight):
    """The pre-0.3.0 stitch implementation, as ground truth."""
    from patchcraft.stitch import _window_kernel
    n_patches, c, ph, pw = patches.shape
    h, w = image_shape[1], image_shape[2]
    sh, sw = (stride, stride) if isinstance(stride, int) else stride
    accum_dtype = (
        torch.float32
        if patches.dtype in (torch.float16, torch.bfloat16)
        else patches.dtype
    )
    kernel = _window_kernel(weight, ph, pw, accum_dtype, patches.device)
    weighted = patches.to(accum_dtype) * kernel
    num_flat = weighted.permute(1, 2, 3, 0).reshape(c * ph * pw, n_patches).unsqueeze(0)
    folded_num = F.fold(num_flat, (h, w), (ph, pw), stride=(sh, sw))
    kernel_flat = kernel.flatten().unsqueeze(1).repeat(1, n_patches).unsqueeze(0)
    folded_den = F.fold(kernel_flat, (h, w), (ph, pw), stride=(sh, sw))
    return (folded_num / folded_den.clamp(min=1e-6))[0].to(patches.dtype)


@pytest.mark.parametrize("c,h,w", [(1, 8, 8), (3, 16, 16), (3, 32, 24)])
@pytest.mark.parametrize(
    "ph,pw,sh,sw",
    [(2, 2, 2, 2), (4, 4, 2, 2), (3, 3, 1, 1), (4, 6, 2, 3)],
)
@pytest.mark.parametrize("weight", ["uniform", "hann", "gaussian"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_stitch_matches_reference(c, h, w, ph, pw, sh, sw, weight, dtype):
    num_h = (h - ph) // sh + 1
    num_w = (w - pw) // sw + 1
    if (num_h - 1) * sh + ph != h or (num_w - 1) * sw + pw != w:
        pytest.skip("geometry does not cover exactly")
    img = torch.randn(c, h, w, dtype=dtype)
    from patchcraft import extract
    patches = extract(img, (ph, pw), (sh, sw))
    got = stitch(patches, (c, h, w), (sh, sw), weight=weight)
    ref = _stitch_reference(patches, (c, h, w), (sh, sw), weight)
    assert got.shape == ref.shape and got.dtype == ref.dtype
    if weight == "uniform":
        assert torch.equal(got, ref)  # integer counts: bit-exact
    else:
        # denominator summation order differs from F.fold: ULP-level
        torch.testing.assert_close(got, ref)


@pytest.mark.parametrize("n", [2, 3, 8, 32, 128])
def test_gaussian_kernel_stays_above_exp_minus_4(n):
    """2-D gaussian kernel minimum is strictly above exp(-4): the 1-D profile
    exceeds exp(-2) at both edges (exponent -2*((n-1)/n)^2 for n >= 4, closer
    to 0 for n in {2, 3}), so the outer-product corner exceeds exp(-4)."""
    from patchcraft.stitch import _window_kernel
    k = _window_kernel("gaussian", n, n, torch.float64, torch.device("cpu"))
    assert k.min().item() > math.exp(-4)
```

- [ ] **Step 2: Run against current implementation**

Run: `.venv/Scripts/python -m pytest tests/test_stitch.py -q -k "matches_reference or exp_minus_4"`
Expected: all PASS.

- [ ] **Step 3: Implement the separable denominator**

In `src/patchcraft/stitch.py`:

(a) After the `_gaussian_1d` definition, add:

```python
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
    ``0 <= y - i*step < len(w1d)`` — the 1-D analog of folding the kernel.

    Within each residue class modulo ``step`` the sum is a sliding window
    over the strided kernel, computed with a cumsum: O(length + len(w1d))
    instead of a 2-D F.fold of the replicated kernel.
    """
    out = w1d.new_zeros(length)
    for r in range(step):
        sub = w1d[r::step]
        n_sub = sub.numel()
        cs = torch.cat([w1d.new_zeros(1), sub.cumsum(0)])
        ks = torch.arange((length - r + step - 1) // step, device=w1d.device)
        hi = torch.clamp(ks + 1, max=n_sub)
        lo = torch.clamp(ks + 1 - num, min=0)
        out[r::step] = cs[hi] - cs[lo]
    return out
```

(b) Rewrite `_window_kernel` to reuse `_window_1d`:

```python
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
```

(c) In `stitch`, replace the block from `kernel = _window_kernel(...)` through the final `return` with:

```python
    wh = _window_1d(weight, ph, accum_dtype, patches.device)
    ww = _window_1d(weight, pw, accum_dtype, patches.device)
    kernel = wh.unsqueeze(1) * ww.unsqueeze(0)

    # Weighted patches: broadcast kernel (ph, pw) across (L, C, ph, pw).
    weighted = patches.to(accum_dtype) * kernel

    # Numerator fold: (L, C, ph, pw) -> (1, C*ph*pw, L) for F.fold.
    num_flat = (
        weighted.permute(1, 2, 3, 0)
        .reshape(c * ph * pw, n_patches)
        .unsqueeze(0)
    )
    folded_num = F.fold(
        num_flat,
        output_size=(h, w),
        kernel_size=(ph, pw),
        stride=(sh, sw),
    )

    # Separable denominator: because the kernel is an outer product,
    # den[y, x] = (sum_i wh[y - i*sh]) * (sum_j ww[x - j*sw]) — two 1-D
    # folds + outer product instead of a second 2-D F.fold.
    den_h = _fold_window_1d(wh, h, num_h, sh)
    den_w = _fold_window_1d(ww, w, num_w, sw)
    den = den_h.unsqueeze(1) * den_w.unsqueeze(0)

    # clamp(min=1e-6): geometry validation guarantees coverage and all three
    # windows are strictly positive, so the denominator is genuinely
    # positive; the clamp is a defensive no-op kept from the fold version.
    return (folded_num[0] / den.clamp(min=1e-6)).to(patches.dtype)
```

(d) Fix the gaussian docstring claim in the `stitch` docstring — replace:

```
    - ``"gaussian"``: Gaussian centered on the patch with per-axis
      ``sigma = max(1.0, ph / 4)`` and ``sigma = max(1.0, pw / 4)``. Smooth
      seam suppression with weight strictly above ``exp(-2)`` everywhere.
```

with:

```
    - ``"gaussian"``: Gaussian centered on the patch with per-axis
      ``sigma = max(1.0, ph / 4)`` and ``sigma = max(1.0, pw / 4)``. Smooth
      seam suppression; the 1-D profile stays above ``exp(-2)`` at the edges,
      so the 2-D kernel stays above ``exp(-4)`` at the corners.
```

(e) Update the module docstring paragraph that describes the internals — replace:

```
    Internally: each patch is multiplied by the 2-D weight kernel, the
    weighted patches are folded into the numerator, the weight kernel itself
    is folded over the same geometry into the denominator, and
    ``numerator / denominator.clamp(min=1e-6)`` gives the output. The clamp
    absorbs float noise on covered pixels; geometry validation guarantees
    no uncovered pixels.
```

with:

```
    Internally: each patch is multiplied by the 2-D weight kernel, the
    weighted patches are folded into the numerator, and the denominator is
    built from two 1-D window folds (the kernel is separable, so the 2-D
    denominator is their outer product). ``numerator / denominator`` gives
    the output; geometry validation guarantees no uncovered pixels and all
    three windows are strictly positive.
```

- [ ] **Step 4: Run tests + linters**

Run: `.venv/Scripts/python -m pytest -q && .venv/Scripts/python -m ruff check src tests && .venv/Scripts/python -m mypy src/patchcraft`
Expected: full suite passes; uniform is `torch.equal`, hann/gaussian `assert_close`.

- [ ] **Step 5: Commit**

```bash
git add src/patchcraft/stitch.py tests/test_stitch.py
git commit -m "perf: stitch denominator via separable 1-D window folds; fix gaussian docstring"
```

---

### Task 6: metrics — single f64 materialization + one GPU sync

**Files:**
- Modify: `src/patchcraft/metrics.py` (`patch_metrics` body; `per_patch_mse` body; `per_patch_psnr` dead branch)
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing.
- Produces: same signatures; `patch_metrics` returns identical values bit-exactly.

- [ ] **Step 1: Write the characterization tests**

Append to `tests/test_metrics.py`:

```python
def _patch_metrics_reference(a, b, max_value=1.0):
    """The pre-0.3.0 patch_metrics computation, as ground truth."""
    import math
    a64 = a.to(torch.float64) if a.dtype != torch.float64 else a
    b64 = b.to(torch.float64) if b.dtype != torch.float64 else b
    diff = a64 - b64
    abs_diff = diff.abs()
    mse = (diff * diff).mean().item()
    psnr_db = float("inf") if mse == 0.0 else 10.0 * math.log10(max_value * max_value / mse)
    return {
        "mae": abs_diff.mean().item(),
        "mse": mse,
        "max_abs": abs_diff.max().item(),
        "psnr_db": psnr_db,
    }


@pytest.mark.parametrize("shape", [(3, 8, 8), (16, 3, 4, 5), (2, 1, 1, 1)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
def test_patch_metrics_matches_reference(shape, dtype):
    a = torch.randn(shape, dtype=dtype)
    b = torch.randn(shape, dtype=dtype)
    got = patch_metrics(a, b)
    ref = _patch_metrics_reference(a, b)
    assert got.keys() == ref.keys()
    for k in ref:
        assert got[k] == ref[k], k  # bit-exact, not approximate


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
def test_per_patch_mse_matches_reference(dtype):
    a = torch.randn(7, 3, 4, 5, dtype=dtype)
    b = torch.randn(7, 3, 4, 5, dtype=dtype)
    got = per_patch_mse(a, b)
    diff = a.to(torch.float64) - b.to(torch.float64)
    ref = (diff * diff).mean(dim=(1, 2, 3))
    assert torch.equal(got, ref)


def test_patch_metrics_does_not_mutate_inputs():
    a = torch.randn(4, 3, 4, 4, dtype=torch.float64)
    b = torch.randn(4, 3, 4, 4, dtype=torch.float64)
    a0, b0 = a.clone(), b.clone()
    patch_metrics(a, b)
    assert torch.equal(a, a0) and torch.equal(b, b0)
```

(Imports needed at top of `tests/test_metrics.py` if missing: `pytest`, `torch`, `patch_metrics`, `per_patch_mse` — check what is already there.)

- [ ] **Step 2: Run against current implementation**

Run: `.venv/Scripts/python -m pytest tests/test_metrics.py -q -k "matches_reference or does_not_mutate"`
Expected: all PASS.

- [ ] **Step 3: Implement**

In `src/patchcraft/metrics.py`, replace the body of `patch_metrics` after `_check_pair`/`_check_max_value`:

```python
    _check_pair(a, b)
    mv = _check_max_value(max_value)

    a64 = a.to(torch.float64) if a.dtype != torch.float64 else a
    b64 = b.to(torch.float64) if b.dtype != torch.float64 else b
    diff = a64 - b64
    abs_diff = diff.abs()
    mse = (diff * diff).mean().item()
    psnr_db = float("inf") if mse == 0.0 else 10.0 * math.log10(mv * mv / mse)
    return {
        "mae": abs_diff.mean().item(),
        "mse": mse,
        "max_abs": abs_diff.max().item(),
        "psnr_db": psnr_db,
    }
```

with:

```python
    _check_pair(a, b)
    mv = _check_max_value(max_value)

    # Single f64 materialization: in-place subtract casts b elementwise, so
    # no separate f64 copy of b (and no mutation of a when a is already f64).
    if a.dtype == torch.float64:
        diff = a - b
    else:
        diff = a.to(torch.float64)
        diff.sub_(b)
    abs_diff = diff.abs()
    # One device->host sync for all three scalars instead of three .item()s.
    mae_t, mse_t, max_abs_t = torch.stack(
        [abs_diff.mean(), (diff * diff).mean(), abs_diff.max()]
    ).tolist()
    mse = float(mse_t)
    psnr_db = float("inf") if mse == 0.0 else 10.0 * math.log10(mv * mv / mse)
    return {
        "mae": float(mae_t),
        "mse": mse,
        "max_abs": float(max_abs_t),
        "psnr_db": psnr_db,
    }
```

Replace the body of `per_patch_mse` after the ndim check:

```python
    a64 = a.to(torch.float64) if a.dtype != torch.float64 else a
    b64 = b.to(torch.float64) if b.dtype != torch.float64 else b
    diff = a64 - b64
    return (diff * diff).mean(dim=(1, 2, 3))
```

with:

```python
    if a.dtype == torch.float64:
        diff = a - b
    else:
        diff = a.to(torch.float64)
        diff.sub_(b)
    return (diff * diff).mean(dim=(1, 2, 3))
```

In `per_patch_psnr`, replace the dead branch:

```python
    mse = per_patch_mse(a, b)
    finfo = torch.finfo(mse.dtype) if mse.is_floating_point() else None
    tiny = finfo.tiny if finfo is not None else 1e-12
    mse_safe = mse.clamp_min(tiny)
```

with:

```python
    mse = per_patch_mse(a, b)  # always float64 per per_patch_mse's contract
    mse_safe = mse.clamp_min(torch.finfo(torch.float64).tiny)
```

- [ ] **Step 4: Run tests + linters**

Run: `.venv/Scripts/python -m pytest -q && .venv/Scripts/python -m ruff check src tests && .venv/Scripts/python -m mypy src/patchcraft`
Expected: full suite passes; bit-exact equality holds.

- [ ] **Step 5: Commit**

```bash
git add src/patchcraft/metrics.py tests/test_metrics.py
git commit -m "perf: metrics with a single f64 materialization and one host sync"
```

---

### Task 7: cache docstring + pyproject cache_dir removal

**Files:**
- Modify: `src/patchcraft/cache.py` (class docstring, lines 74-82)
- Modify: `pyproject.toml` (line 74)
- Modify (maybe): `.gitignore`

**Interfaces:** none.

- [ ] **Step 1: Document the prefix tradeoff**

In `src/patchcraft/cache.py`, extend the `Cache` class docstring after the paragraph about `version`:

```python
    """Single-namespace content-addressed cache on disk.

    ``root`` is created on construction if missing. ``namespace`` is used
    as a subdirectory and as part of the key (so two namespaces never
    collide even if a caller produces identical key parts). ``version``
    is the invalidation lever: bump it, and old entries become
    unreadable by construction without any delete.

    Filenames address entries by the first 16 hex chars (64 bits) of the
    full key. Two distinct keys sharing that prefix overwrite each other
    on ``put``; ``get`` detects the mismatch via the sidecar and reports
    a transparent miss, never wrong bytes. The birthday bound puts the
    collision probability at ~50% only around 5e9 entries, so the prefix
    stays at 16 chars; widen ``_paths`` if that ever stops being true.
    """
```

- [ ] **Step 2: Remove the machine-local pytest cache_dir**

In `pyproject.toml`, delete line 74 (`cache_dir = "Z:\\caches\\pytest"`). This was a Windows-local path that made Linux/macOS runs create a literal `Z:\caches\pytest` directory.

- [ ] **Step 3: Verify the default cache location is gitignored**

pytest now falls back to `<rootdir>/.pytest_cache`. Check `.gitignore` covers `.pytest_cache/`; if not, append it. Then run the suite and confirm `git status --short` shows nothing new.

Run: `.venv/Scripts/python -m pytest -q && git status --short`
Expected: 346+ passed; working tree clean apart from the two edited files.

- [ ] **Step 4: Commit**

```bash
git add src/patchcraft/cache.py pyproject.toml .gitignore
git commit -m "chore: drop machine-local pytest cache_dir; document cache key-prefix tradeoff"
```

---

### Task 8: docs — factual corrections only

**Files:**
- Modify: `README.md`, `README.pt-BR.md`, `README.pypi.md`, `docs/GUIDE.md`, `docs/SCOPE.md`, `docs/THEORY.md`, `docs/USAGE.md`, `docs/AUXILIARY.md`, `CHANGELOG.md`

**Interfaces:** none. No restructuring, no cuts — factual corrections at the exact spots listed.

- [ ] **Step 1: Version strings 0.2.1 → 0.3.0**

- `README.md:136` — `**0.2.1, pre-1.0.**` → `**0.3.0, pre-1.0.**`
- `README.pt-BR.md:136` — `**0.2.1, pré-1.0.**` → `**0.3.0, pré-1.0.**`
- `README.pypi.md:119` — "This page documents 0.2.1" → "This page documents 0.3.0"
- `docs/GUIDE.md:9` — "run against `patchcraft` 0.2.1" → 0.3.0
- `docs/GUIDE.md:726` — `assert patchcraft.__version__ == "0.2.1"` → `"0.3.0"`
- `docs/GUIDE.md:791` — "Version 0.2.1 is pre-1.0" → 0.3.0
- `docs/GUIDE.md:845` — BibTeX `version = {0.2.1}` → `version = {0.3.0}`

Verify with: `grep -rn "0\.2\.1" README.md README.pt-BR.md README.pypi.md docs/GUIDE.md` — expected: no hits afterward (except legitimate historical mentions, if any; inspect each).

- [ ] **Step 2: SCOPE.md §4.4 (lines ~241-248)**

The section claims `reconstruct` has no dtype guard and integer patches fail with a raw `NotImplementedError` from torch. That has been false since 0.2.1. Read the section, then replace the false sentences with the current behavior: since 0.2.1, `reconstruct` and `stitch` reject non-floating-point patches with a framed `ValueError` telling the caller to convert with `patches.float()` (cross-reference THEORY §9.2 and §9.9). Keep the section's structure and length roughly the same; do not touch other sections.

- [ ] **Step 3: Bit-exactness rule in THEORY/USAGE**

Canonical rule (from the `reconstruct` docstring): the round trip is bit-exact when every value in the overlap count map is a power of two; `stride == patch_size` always qualifies; `stride == patch_size / 2` qualifies (counts 1, 2, 4); a 3 or 9 in the map does not, and the error is ~1 ULP in float32.

- `docs/THEORY.md:100` — replace the "bit-exactly for fractional pixel values" claim with the canonical rule.
- `docs/THEORY.md:153` and `:157` — same correction wherever the old "float64 bit-exact / float32 ~1 ULP" rule is stated.
- `docs/USAGE.md:149-160` — replace the "Overlap: weighted, still exact… Bit-exact for float64; within ~1 ULP for float32" passage with the canonical rule. USAGE is banner-marked as stale; this fixes its worst factual error but does not regenerate the file.

- [ ] **Step 4: THEORY.md §9.2 line ~279**

Remove the bullet listing "Promoção automática float16 → float32" from the "Fora de escopo v0.1" list — it has been implemented since 0.2.1 (the same section's line ~269 already says so).

- [ ] **Step 5: CHANGELOG.md fixes**

- Line ~65: "It is 137 lines now" — the README is 156 lines; correct the number (or reword to drop the count).
- Lines ~235-236: the gaussian kernel description `sigma = max(1, min(ph, pw) / 4)` → per-axis `sigma = max(1.0, n / 4.0)` (matches `stitch.py` and THEORY/USAGE).
- Add the missing link targets at the bottom, following the existing pattern of `[0.2.0]`/`[0.1.0]`:

```markdown
[Unreleased]: https://github.com/LeoPR/PatchCraft/compare/v0.2.2...HEAD
[0.2.2]: https://github.com/LeoPR/PatchCraft/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/LeoPR/PatchCraft/compare/v0.2.0...v0.2.1
```

(Check the exact URL pattern used by the existing `[0.2.0]` target first and match it, including tag naming.)

- [ ] **Step 6: AUXILIARY.md line ~114**

The markdown links `[Z:\caches](../../../caches)` / `[Z:\venvs](../../../venvs)` point nowhere on GitHub. Replace with inline code spans (`` `Z:\caches` `` / `` `Z:\venvs` ``), keeping the sentence.

- [ ] **Step 7: Commit**

```bash
git add README.md README.pt-BR.md README.pypi.md docs/GUIDE.md docs/SCOPE.md docs/THEORY.md docs/USAGE.md docs/AUXILIARY.md CHANGELOG.md
git commit -m "docs: correct stale version strings and factual errors from the 0.2.2 audit"
```

---

### Task 9: version bump 0.3.0 + CHANGELOG entry + benchmarks + full verification

**Files:**
- Modify: `src/patchcraft/__init__.py` (line 19)
- Modify: `CHANGELOG.md` (new entry at top)
- Modify: `lab/bench_quick.py` (extend to print the after table; lab/ is gitignored scratch, do not commit)

**Interfaces:** none.

- [ ] **Step 1: Bump the version**

`src/patchcraft/__init__.py:19`: `__version__ = "0.2.2"` → `__version__ = "0.3.0"`.

- [ ] **Step 2: CHANGELOG entry**

Add at the top of `CHANGELOG.md`, following the file's existing entry style:

```markdown
## [0.3.0] - 2026-09-01

### Performance

- `extract` uses a strided-window view (`Tensor.unfold`) for `dilation == 1`,
  one copy instead of im2col + permute-contiguous: 13-21x faster on CPU,
  bit-exact. `dilation > 1` still uses `F.unfold`.
- `reconstruct` skips `F.fold` entirely on non-overlapping grids (a pure
  rearrangement, 27x faster on CPU) and computes the overlap count map in
  closed form O(H+W) instead of folding a tensor of ones.
- `stitch` builds its denominator from two 1-D window folds (the kernels are
  separable) instead of a second 2-D `F.fold`. `uniform` stays bit-exact vs
  `reconstruct`; `hann`/`gaussian` may differ by ULPs (summation order).
- `patch_metrics`/`per_patch_mse` materialize one f64 tensor instead of
  three and sync the device once instead of three times.

### Fixed

- `resize` with `backend="pil"` clamped integer tensors to the uint8 range
  before casting; values outside [0, 255] used to wrap (300 became 44).
- Removed the machine-local `cache_dir = "Z:\caches\pytest"` from
  `pyproject.toml` (created a literal `Z:\caches\...` directory on
  Linux/macOS).

### Documentation

- Corrected stale 0.2.1 version strings on all cover pages and in GUIDE.
- Corrected the gaussian-kernel floor claim in `stitch` (exp(-4) at corners,
  not exp(-2) everywhere).
- Corrected the bit-exactness rule in THEORY/USAGE (power-of-two count map).
- SCOPE §4.4 no longer claims `reconstruct` lacks a dtype guard.
- CHANGELOG: link targets for [Unreleased], [0.2.2], [0.2.1].

### Internal

- `reconstruct`/`stitch` share one fold-geometry validator
  (`patchcraft._foldgeom.check_fold_geometry`); error messages unchanged.
```

And add the `[0.3.0]` link target at the bottom (`.../compare/v0.2.2...v0.3.0`), plus update `[Unreleased]` to compare `v0.3.0...HEAD` if Task 8 already added it.

- [ ] **Step 3: Re-run benchmarks and compare**

Run: `.venv/Scripts/python lab/bench_quick.py > lab/bench_after.txt 2>&1 && .venv/Scripts/python -c "print(open('lab/bench_after.txt').read())"`
Expected vs `lab/bench_before.txt`: extract non-overlap ~0.16 ms (was ~2.2), extract overlap ~0.45 ms (was ~9.5), reconstruct non-overlap ~0.16 ms (was ~4.3), reconstruct overlap drops by the ones-fold share (~4 ms of ~26), stitch down ~20%, metrics ~1.5x.

- [ ] **Step 4: Full verification**

Run: `.venv/Scripts/python -m pytest -q && .venv/Scripts/python -m ruff check src tests && .venv/Scripts/python -m mypy src/patchcraft && git status --short`
Expected: 346+new tests pass, 5 skipped (CUDA); ruff/mypy clean; working tree shows only the intended files.

- [ ] **Step 5: Commit**

```bash
git add src/patchcraft/__init__.py CHANGELOG.md
git commit -m "release: v0.3.0, pure-torch performance fast paths and small fixes"
```

---

## Self-review notes

- Spec §1 bugs 1-7 → Tasks 1 (1), 5 (2), 7 (3, 6), 6 (4), 1 (5), 3 (7). All covered.
- Spec §2.1/2.2/2.3/2.4 → Tasks 2, 4, 5, 6. All covered.
- Spec §3 tests → characterization grids in Tasks 2/4/5/6, aliasing test Task 2, uint8 test Task 1, exp(-4) invariant Task 5, benchmark Tasks 2/9.
- Spec §4 docs → Task 8 (all listed spots).
- Spec §5 version → Task 9.
- Equivalence levels match spec §2: `torch.equal` for extract/reconstruct/uniform-stitch/metrics; `assert_close` for hann/gaussian stitch.
- Not in plan (spec §6, out of scope): Rust layer, doc consolidation, `index_add_` prototype.
