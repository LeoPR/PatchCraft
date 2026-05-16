# PatchKit — Roadmap

Milestone-based plan. Each milestone is "done" only when its tests pass and the theory doc section it depends on is written.

## M0 — Scaffold (this commit)

- [x] `pyproject.toml` (hatchling, Python >=3.12).
- [x] `src/patchkit/__init__.py` with `__version__`.
- [x] `tests/test_import.py` verifying the package imports.
- [x] `docs/THEORY.md` and `docs/ROADMAP.md` skeletons.
- [x] `archive/` populated with reference implementations.
- [x] Venv on `Z:\venvs\patchkit` with torch+CUDA (**Stage 3** in the parent workflow).
- [x] `pytest` green with the single import test.

## M1 — Theory distilled

- [x] Read `archive/PatchHub/src/patchhub/{cache,quantization,resize,resize_cache,subset}.py`.
- [x] Read `archive/QSVM_patchkit/patchkit/{patches,processed,quantize,image_utils,superres}.py`.
- [x] Fill in the "Design decision" paragraph of each section in `docs/THEORY.md`.
- [x] Write `docs/ADR/0001-patch-extraction-api.md` (first Architecture Decision Record).

## M2 — Patch extraction

- [x] `patchkit.extract(image, patch_size, stride, dilation) -> Tensor[L, C, ph, pw]`.
- [x] Implementation via `torch.nn.functional.unfold`.
- [x] Tests:
  - Shape invariants for several `(H, W, ph, pw, sh, sw, d)` combinations.
  - [ ] Round-trip with reconstruction — deferred to M3 (needs `reconstruct`).
  - [x] `torch.cuda` path when GPU available (marker `gpu`).
  - [x] Rejection tests for §10.1 negative conditions.

## M3 — Reconstruction

- [ ] `patchkit.reconstruct(patches, image_shape, stride, dilation) -> Tensor[C, H, W]`.
- [ ] Implementation via `torch.nn.functional.fold` plus overlap normalization.
- [ ] Tests:
  - Bit-exact round-trip for `stride == patch_size`.
  - Weighted reconstruction matches expected behaviour for known overlaps.
  - Rejects `dilation != 1` with a clear error (documented limitation).

## M4 — LR ↔ HR pairing

- [ ] `patchkit.pair(lr_image, hr_image, lr_patch_size, scale_factor, stride) -> PairIterator`.
- [ ] Invariant tests: same `k` yields patches covering the same image region.
- [ ] Metadata contract documented (image id, patch index, coords).

## M5 — Resize + cache

- [ ] `patchkit.resize(image, target_size, backend="pil"|"torch") -> Image | Tensor`.
- [ ] Content-addressed disk cache (optional `zstandard` dep for compression).
- [ ] Tests:
  - Resize parity within a backend across runs.
  - Cache hit/miss behaviour with controlled configuration changes.

## M6 — Label-stratified subsets

- [ ] `patchkit.label_subset(dataset, n_per_label, seed) -> torch.utils.data.Subset`.
- [ ] Tests for determinism and correct counts.

## M7 — First release

- [ ] Version bump to `0.1.0`.
- [ ] `CHANGELOG.md` created.
- [ ] Build wheel: `uv build` produces `dist/patchkit-0.1.0-py3-none-any.whl`.
- [ ] GitHub repo `LeoPR/patchkit` created and first tag pushed.
- [ ] (Optional) PyPI publish.

## Post-release

- Integration back into QPatchSR: `pip install patchkit` and implement kernels/regressors on top.
- Potential auxiliary #2 for quantum components (naming TBD).
