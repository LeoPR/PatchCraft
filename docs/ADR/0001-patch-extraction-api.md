# ADR 0001 — Patch extraction API

- **Status:** Accepted
- **Date:** 2026-04-21
- **Deciders:** Leonardo Marques de Souza
- **Supersedes:** n/a (first ADR)

## Context

PatchCraft's central primitive is patch extraction: given an image, produce a tensor of rectangular sub-regions. Every downstream piece (reconstruction, LR↔HR pairing, dataset caching) depends on the shape of this API. Two reference implementations exist in the archive:

1. `archive/QSVM_patchkit/patchkit/patches.py` — class-based `OptimizedPatchExtractor`. State includes: patch size, stride, dilation, image size (fixed at construction), an in-memory LRU cache, a disk cache path, and config-hashed cache filenames. `process(image, index)` does extraction and caching in one call. Supports reconstruction via `reconstruct_image(patches)`.
2. `archive/PatchHub/src/patchhub/` — no extraction primitive. PatchHub is post-processing only (cache, quantize, resize, subset). Shows the project's preference for small, composable, stateless utilities.

These implementations disagree on the fundamentals:
- **Shape of state**: QSVM_patchkit bundles extraction + caching + reconstruction into one stateful object. PatchHub prefers free functions with a separate `Cache`.
- **Error handling**: QSVM_patchkit validates `dilation > 1` at reconstruction time (runtime). PatchHub has no precedent because it doesn't extract.
- **Input shape**: QSVM_patchkit accepts an opaque image (unclear whether PIL, numpy, or tensor) and internally normalizes. PatchHub's resize utilities take explicit types.

A new consumer (QPatchSR, possibly others) will depend on whatever API v0.1 ships. Changing later is a breaking change that forces lock-step releases.

## Decision

Patch extraction is a **pure function** with the signature:

```python
def extract(
    image: torch.Tensor,          # (C, H, W), any dtype, any device
    patch_size: int | tuple[int, int],
    stride: int | tuple[int, int],
    dilation: int | tuple[int, int] = 1,
) -> torch.Tensor:                # (L, C, ph, pw), dtype/device preserved
```

### Specified behaviour

- **Shape formula:** `num_h = (H − d_h · (ph − 1) − 1) // sh + 1`, analogous for width. `L = num_h · num_w`.
- **Ordering:** row-major. Patch `k` has its top-left at `(row · sh, col · sw)` where `row = k // num_w`, `col = k % num_w`.
- **Empty grid:** when the patch geometry does not fit, return `torch.empty(0, C, ph, pw, dtype=image.dtype, device=image.device)`. Do not raise.
- **Backend:** `torch.nn.functional.unfold`, a single call, no Python loops over patches.
- **Dilation:** supported. The sibling `reconstruct` function rejects `dilation != 1`; extraction does not, because feature-style consumers (kernels, classifiers) may want dilated patches and never round-trip.
- **Input validation:** `image` must be a 3-D tensor; 2-D and 4-D raise `ValueError`. `patch_size`, `stride`, `dilation` must be positive.
- **No side effects:** no caching, no logging, no disk writes.

### What is deliberately *not* in this API

- No class. No constructor that fixes `image_size` ahead of time. Reusing an extractor across different image sizes is a valid request, and stateless functions handle it for free.
- No `image_id` / `index` argument. That is metadata the caller owns; it is passed to `pair()` (see §3 of THEORY.md) when pairing, not to `extract()`.
- No cache. Caching a patch tensor is rarely worth it — the unfold call is cheap, the storage cost is the full image expansion (up to `ph · pw / (sh · sw)` × the pixel count). When caching *is* wanted, callers wrap `extract` with `patchcraft.Cache` explicitly.
- No batched input (`(B, C, H, W)`). v0.1 is single-image. Looping in Python is acceptable for the initial consumer; a batched variant or a `vmap` recipe will be added when benchmarks justify it.
- No PIL input. Callers convert via `torch.from_numpy(np.asarray(pil_img))` or `torchvision.transforms.functional.to_tensor`. Accepting PIL internally would hide the conversion cost and the dtype/range ambiguity.
- No padding modes. Truncation is the only policy. Callers needing full coverage pad the image before calling `extract`.

## Consequences

**Positive.**
- Pure function is trivial to test and to reason about. Shape invariants are the whole contract.
- Composes with `Cache`, `pair`, `reconstruct`, and any future primitive without coupling.
- Works identically on CPU and CUDA tensors; the user picks the device.
- Stateless means no "is my extractor configured for this image size?" footgun.

**Negative.**
- Callers that process many same-size images pay a small per-call overhead for argument validation and the reshape. Expected to be noise next to `unfold` itself; will revisit if a benchmark disagrees.
- No amortization of the reshape logic across calls — the per-call work is constant but non-zero. Again, measure first.
- Users migrating from `OptimizedPatchExtractor` need to rewrite: `extractor.process(img)` becomes `patchcraft.extract(img, patch_size=..., stride=...)`. Caching moves from implicit to explicit. This is intentional — the migration makes the prior hidden state visible.

**Neutral.**
- The metadata contract (`image_id`, `patch_index`, coordinates) lives in `pair()` and in the `PatchMeta` dataclass, not in `extract()`. Extraction produces pixels; pairing produces correspondences.

## Alternatives considered

### A. Class-based `PatchExtractor` (QSVM_patchkit style)

```python
extractor = PatchExtractor(patch_size=4, stride=2, dilation=1)
patches = extractor.extract(image)
```

**Rejected.** Construction cost is zero, so the class buys nothing over a function except the temptation to add state later ("it's already a class, just cache inside it"). The archive implementation illustrates exactly this drift: `OptimizedPatchExtractor` grew an LRU cache, a disk cache, and a fixed `image_size`.

### B. Tuple-returning `extract` that yields `(patches, meta)`

```python
patches, meta = extract(image, ...)
# meta.num_patches_h, meta.num_patches_w, meta.coords
```

**Rejected.** Makes the common path (just want the patches) uglier, and the metadata is recomputable from the inputs in one line. `pair()` needs metadata; `extract()` alone does not.

### C. Accept PIL and numpy as inputs, with automatic conversion

**Rejected.** Three input types × three dtype conventions = nine ambiguity cases. Converting upfront is one line and forces the caller to acknowledge dtype/range.

### D. Support dilation in both extract and reconstruct

**Rejected** (at least for v0.1). `F.fold` does not natively reverse dilated unfold; implementing a custom scatter-divide doubles the reconstruction code and edge cases. Users who want dilated features do not round-trip; users who want round-trip use `dilation=1`.

## Status after M2

When M2 closes (extract implemented + tests green), this ADR becomes the enforced contract. Later changes to the signature require a new ADR superseding this one and a major-version bump (`patchcraft 1.x`).
