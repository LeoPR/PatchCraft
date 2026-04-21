# PatchKit — Theory Notes

Working document. The goal is to distill the useful theory from the reference implementations in [`../archive/`](../archive/) into a single place **before** writing new code. Keep this document as the source-of-truth for design decisions.

> Every section should end with a paragraph titled **Design decision** stating what PatchKit will actually implement and why.

## 1. Patch extraction

**Topic.** Given an image of shape `(C, H, W)`, a patch size `(ph, pw)`, a stride `(sh, sw)` and an optional dilation `d`, define:
- `num_patches_h`, `num_patches_w` as functions of the inputs.
- The mapping `(patch_index) → (row, col)` and its inverse.
- The extraction operator as a single call to `torch.nn.functional.unfold`.

To write:
- Formula for `num_patches_h = (H − d·(ph − 1) − 1) / sh + 1`.
- Boundary conditions (truncation vs. padding).
- Memory layout of `unfold` output and how to reshape back to `(L, C, ph, pw)`.

**Design decision.** *TBD.*

## 2. Reconstruction

**Topic.** Inverse of extraction via `torch.nn.functional.fold`. Two regimes:
- **Exact** — when `sh == ph` and `sw == pw` and `d == 1`: patches tile the image; reconstruction is a cheap copy (weights are all ones).
- **Weighted overlap** — when `sh < ph` or `sw < pw`: patches overlap; reconstruction must divide by the overlap count map (from `fold` of an all-ones tensor with the same geometry).

To write:
- Worked example for `ph = 4, sh = 2` → 2× overlap in the middle rows/cols.
- Why dilation > 1 cannot reconstruct via `fold` directly.

**Design decision.** *TBD.*

## 3. LR ↔ HR pairing

**Topic.** Given a scale factor `r` and LR/HR shapes related by `HR = r · LR`, define patch sizes and strides on both sides so that the *k*-th LR patch corresponds to the *k*-th HR patch (same `k`). Invariant:
- `ph_hr = r · ph_lr` and `sh_hr = r · sh_lr` (similar for width).
- Then `num_patches_lr == num_patches_hr`, and the coordinates of patch *k* on both sides map to the same image region in pixel space.

To write:
- Proof of the coverage invariant.
- What happens when `(H − d·(ph − 1)) % sh != 0` (truncated patches lost on both sides).
- Recommended defaults for "no-overlap" pairing.

**Design decision.** *TBD.*

## 4. Cache semantics

**Topic.** Processed datasets (resized / quantized / paired) are expensive to build; cache them on disk keyed by a hash of the configuration. Reference implementations:
- `archive/PatchHub/src/patchhub/cache.py` — mmap-friendly on-disk cache with zstd.
- `archive/QSVM_patchkit/patchkit/processed.py` — cache bundle with labels.

To write:
- What fields go into the cache key (all configuration that affects content).
- Invalidation: when does a cache hit become stale?
- Serialization format (torch.save + zstd vs raw numpy memmap).
- Concurrency expectations (single-writer).

**Design decision.** *TBD.*

## 5. Resize backends

**Topic.** Image resize is surprisingly opinionated:
- **PIL** — battle-tested, supports BICUBIC/LANCZOS/etc., CPU-only, convenient.
- **torch** — `F.interpolate`, runs on GPU, different bicubic filter than PIL.

To write:
- Whether PatchKit exposes a single `resize(backend=...)` API or two functions.
- Documented differences (PIL vs torch bicubic are *not* pixel-identical).
- When to prefer each.

**Design decision.** *TBD.*

## 6. Quantization (optional)

**Topic.** Reducing color depth (binary, k-level) before extraction. Reference: `archive/QSVM_patchkit/patchkit/quantize.py` — uniform, k-means, Otsu, Floyd–Steinberg dither.

To write:
- Whether quantization belongs in PatchKit at all, or in a downstream consumer.
- If yes, which methods are worth carrying.

**Design decision.** *TBD.*

## 7. Label-stratified subsets

**Topic.** Given a labeled dataset, pick `n` samples per label with a deterministic seed. Reference: `archive/PatchHub/src/patchhub/subset.py` — `LabelSubset`.

To write:
- API (functional vs class).
- Deterministic seeding.
- Integration with `torch.utils.data.Subset`.

**Design decision.** *TBD.*

## 8. Open questions

- Do we want a `PatchPairDataset` that yields `(lr_patch, hr_patch, meta)` directly, or keep extraction and pairing decoupled?
- Should `meta` be a Python dict or a structured tensor for GPU-friendly batching?
- What is the minimum torch version we target? Currently `>=2.6` in `pyproject.toml`.
