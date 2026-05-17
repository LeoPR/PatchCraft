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
- [x] `patchkit.num_patches(image_shape, patch_size, stride, dilation=1)` and `patchkit.tilings(image_shape, allow_overlap=..., min_patch_size=..., max_patch_size=...)` — pre-flight geometry helpers (THEORY §1.5 / §9.6). 28 tests, incl. 28×28 divisor enumeration and exhaustive round-trip guarantee.

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

- [x] `patchkit.pair(lr_image, hr_image, lr_patch_size, scale_factor, stride, *, image_id=None) -> PatchPair`.
- [x] `PatchPair` frozen dataclass: `lr_patches`, `hr_patches`, `metas`. `__len__` and `zip(...)` iteration sugar.
- [x] `PatchMeta` frozen dataclass: `patch_index`, `row`, `col` (LR coords), `lr_patch_size`, `hr_patch_size`, `image_id`.
- [x] Invariant test: same `k` yields LR and HR patches covering the same image region (validated via subview equality on both sides).
- [x] Metadata contract documented in THEORY §3 / §9.3.
- [x] Rejection tests for §9.3 negatives (shape mismatch, scale non-int, channel/dtype/device mismatch).

## M5 — Resize + cache

- [x] `patchkit.resize(image, target_size, backend="pil"|"torch", resample=None) -> Image | Tensor`. Output type matches input; cross-backend conversions via float32 [0,1] / uint8 hop. CUDA tensors only with backend="torch".
- [x] `patchkit.Cache(root, namespace, version=1)` — content-addressed disk cache with OneDrive-race retry (5 attempts with exponential backoff on put, 2 on get), optional `zstandard` compression (transparent fallback when not installed), sidecar JSON with checksum, atomic rename via `*.tmp`.
- [x] Resize tests (38, 2 GPU skip).
- [x] Cache tests (35): roundtrip with bytes/bytearray/memoryview, version invalidation, namespace isolation, checksum/zstd-decode corruption, retry-on-PermissionError simulation, non-ASCII paths, key determinism + namespace/version sensitivity.

## M6 — *(removed)*

Label-stratified subsetting was moved out of the core library per the binding scope in [`THEORY.md`](THEORY.md) §0 (PatchKit operates on one image at a time; no dataset abstractions). The function lives in [`tests/_datasets.py::label_subset`](../tests/_datasets.py) as part of the auxiliary framework (test fixtures + `lab/`). The roadmap renumbers no further — there is no M6 in v0.1.

## M7 — First release (v0.1.0 — 2026-05-16)

- [x] Version bump to `0.1.0`.
- [x] `CHANGELOG.md` created (Keep-a-Changelog format).
- [x] Build wheel: `uv build` produces `dist/patchkit-0.1.0-py3-none-any.whl` and `.tar.gz`.
- [x] GitHub repo [`LeoPR/PatchKit`](https://github.com/LeoPR/PatchKit) created, tag `v0.1.0` pushed at `a612c16`.
- [x] Post-release docs: [`USAGE.md`](USAGE.md), [`SCOPE.md`](SCOPE.md), [`AUXILIARY.md`](AUXILIARY.md).

## M8 — v0.2.0 expansion (2026-05-17)

Motivated by the QPatchSR consumer's needs surfaced after v0.1.0 shipped. Public API: 11 → 18 symbols. No breaking changes.

- [x] Cross-resolution geometry: `scale_factor`, `paired_tilings`, `PairedTilingSpec` (THEORY §1.5 expansion, §9.7).
- [x] Pixel metrics module: `patch_metrics`, `per_patch_mse`, `per_patch_psnr` (THEORY §1.6, §9.8; SCOPE §4.3 explains why SSIM/LPIPS stayed out).
- [x] `stitch(..., weight="uniform"|"hann"|"gaussian")` — seam-aware reassembly for modified patches (THEORY §2.5, §9.9; SCOPE §4.4 explains why it is a separate function from `reconstruct`).
- [x] README ASCII "Visual cheat sheet" covering extract / reconstruct / pair / stitch.
- [x] `__version__` bump to `0.2.0`, CHANGELOG `[0.2.0]` section closed.
- [x] CI scaffold: [`.github/workflows/test.yml`](../.github/workflows/test.yml) (PR validation) and [`.github/workflows/release.yml`](../.github/workflows/release.yml) (PyPI publish via Trusted Publishing on `vX.Y.Z` tag push).
- [ ] Tag `v0.2.0` pushed.
- [ ] PyPI account + Trusted Publisher setup (manual, one-time).
- [ ] First PyPI release published (triggered by tag push).

## Post-release

- Integration back into QPatchSR: `pip install patchkit` and implement kernels/regressors on top of the validated v0.2 API.
- Potential auxiliary #2 for quantum components (naming TBD).
- Companion package `patchkit-quant` for quantization primitives (THEORY §6) if a second consumer surfaces the need.
