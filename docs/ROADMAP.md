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
- [x] `patchkit.Patchify(patch_size, stride, dilation)` — callable wrapper for `torchvision.transforms.Compose`, ADR 0002.
- [x] Implementation via `torch.nn.functional.unfold`.
- [x] Tests:
  - Shape invariants for several `(H, W, ph, pw, sh, sw, d)` combinations.
  - [x] Round-trip with reconstruction — covered by M3 tests.
  - [x] `torch.cuda` path when GPU available (marker `gpu`).
  - [x] Rejection tests for §9.1 negative conditions.
  - [x] `Patchify` delegation, eager validation, repr, statelessness.

## M3 — Reconstruction

- [x] `patchkit.reconstruct(patches, image_shape, stride, dilation) -> Tensor[C, H, W]`.
- [x] Implementation via `torch.nn.functional.fold` plus overlap normalization.
- [x] Tests:
  - Bit-exact round-trip for `stride == patch_size` (basic, rectangular, multichannel, single-patch, patch_size=1).
  - Weighted reconstruction for overlap (half overlap, max overlap stride=1, asymmetric, float32+float64).
  - Count-map correctness (uniform image reconstructs uniformly; corners covered).
  - Rejects `dilation != 1` (§9.2).
  - Rejects `sh > ph` or `sw > pw` (partial coverage forbidden, §9.2).
  - Rejects shape/arity/channel/L-mismatch/image-too-small with explicit messages.
- [x] First `lab/` script: `imagem → extract → reconstruct → assert close` on MNIST.

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

## M6 — *(removed)*

Label-stratified subsetting was moved out of the core library per the binding scope in [`THEORY.md`](THEORY.md) §0 (PatchKit operates on one image at a time; no dataset abstractions). The function lives in [`tests/_datasets.py::label_subset`](../tests/_datasets.py) as part of the auxiliary framework (test fixtures + `lab/`). The roadmap renumbers no further — there is no M6 in v0.1.

## M7 — First release

- [ ] Version bump to `0.1.0`.
- [ ] `CHANGELOG.md` created.
- [ ] Build wheel: `uv build` produces `dist/patchkit-0.1.0-py3-none-any.whl`.
- [ ] GitHub repo `LeoPR/patchkit` created and first tag pushed.
- [ ] (Optional) PyPI publish.

## Post-release

- Integration back into QPatchSR: `pip install patchkit` and implement kernels/regressors on top.
- Potential auxiliary #2 for quantum components (naming TBD).
