# Changelog

All notable changes to PatchKit will be documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] — 2026-05-17

Second public release. Adds three feature groups motivated by the QPatchSR
super-resolution consumer plus internal ergonomics. No breaking changes vs
v0.1.0 — all v0.1.0 imports keep working.

### Added — cross-resolution geometry (THEORY §1.5, §9.7)

Motivated by the QPatchSR consumer's question: "given two image shapes
(LR and HR of the same source), what `(patch_size, stride)` on each
side yields the same number of patches with corresponding regions?"
Three new helpers in `patchkit.geometry`:

- **`scale_factor(lr_shape, hr_shape) -> int | None`** — returns the
  integer `k` such that `hr.shape[-2:] == (k * lr.shape[-2], k *
  lr.shape[-1])`, or `None`. Accepts `(H, W)` or `(C, H, W)`. Pre-
  flight check for `pair`.
- **`paired_tilings(lr_shape, hr_shape, *, allow_overlap=False, ...)`**
  — enumerates every `(lr_spec, hr_spec)` pair where both fully cover
  their respective image and produce identical patch counts. Patch
  `k` on each side covers the same image region.
  Example: `paired_tilings((14, 14), (28, 28))` returns three pairs:
  `(p_lr=2, p_hr=4, total=49)`, `(p_lr=7, p_hr=14, total=4)`,
  `(p_lr=14, p_hr=28, total=1)`.
- **`PairedTilingSpec(lr, hr, scale_factor)`** — `NamedTuple` carrying
  both sides and the discovered scale factor.

### Added — patch-level pixel metrics (THEORY §1.6, §9.8)

Canonical reductions so consumers don't reinvent slightly-divergent
versions in every project. New module `patchkit.metrics`:

- **`patch_metrics(a, b, *, max_value=1.0) -> dict[str, float]`** —
  scalar `mae`, `mse`, `max_abs`, `psnr_db` over the full tensor
  (any matching shape works). Internal accumulation in `float64`
  for stability; PSNR returns `+inf` for identical inputs.
- **`per_patch_mse(a, b) -> Tensor[L]`** — one MSE per patch in a
  `(L, C, h, w)` stack.
- **`per_patch_psnr(a, b, *, max_value=1.0) -> Tensor[L]`** — one
  PSNR per patch. Identical patches yield `+inf` via `torch.where`
  (no clamp tricks).

Explicitly **not** included: SSIM, MS-SSIM, LPIPS, FID, any windowed
or learned metric. Use `pytorch-msssim`, `lpips`, `clean-fid` on the
caller side ([SCOPE.md](docs/SCOPE.md) §4.3 explains the boundary).

### Added — patch stitching for modified patches (THEORY §2.5, §9.9)

`reconstruct` is the bit-exact inverse of `extract`. When patches have
been modified (model output, denoised, super-resolved), averaging them
back uniformly shows visible seams at patch boundaries. `stitch` is the
seam-aware counterpart: it folds patches weighted by a 2-D window
kernel so each pixel "trusts" patches closer to its center more.

- **`stitch(patches, image_shape, stride, *, weight="uniform"|"hann"|"gaussian", dilation=1)`**
  — same `F.fold` geometry and rejections as `reconstruct`; adds a
  weighted-blend numerator over a weighted-sum denominator. With
  `weight="uniform"` it is mathematically equivalent to `reconstruct`
  (covered by a bit-exact equality test on no-overlap and `allclose`
  on overlap). With `"hann"` it strongly suppresses seams at the
  cost of zeroing image corners that fall on Hann's edge-weight-zero
  region (documented artifact). With `"gaussian"`
  (`sigma = max(1, min(ph, pw) / 4)`) it blends smoothly with no
  corner artifact.

Floating-point patches only — window kernels are float-valued and we
refuse to silently quantize or implicitly promote. Caller converts to
`float` first.

### Changed

- Public API surface: 11 → 18 symbols.
- [`docs/SCOPE.md`](docs/SCOPE.md) gains rows for paired tilings,
  pixel metrics, and stitch; §4.3 discusses why pixel metrics stayed
  core while windowed/learned metrics did not, §4.4 explains why
  `stitch` is a separate function from `reconstruct` rather than a
  kwarg.
- [`docs/THEORY.md`](docs/THEORY.md) gains §1.5 expansion (cross-
  resolution paragraphs), §1.6 (patch comparison metrics), §2.5
  (stitch — math, kernels, why it is separate), §9.7
  (paired tilings contract), §9.8 (metrics contract), §9.9 (stitch
  contract).

## [0.1.0] — 2026-05-16

First public release. Public API stable; signatures will only change in 1.x.

### Added — core (one image at a time)

- **`extract(image, patch_size, stride, dilation=1)`** — patches from a
  `(C, H, W)` tensor via `torch.nn.functional.unfold`. Truncation-only
  boundary; returns `Tensor[0, C, ph, pw]` when geometry fits no patch.
  Per [ADR 0001](docs/ADR/0001-patch-extraction-api.md).
- **`Patchify(patch_size, stride, dilation=1)`** — callable wrapper for
  `torchvision.transforms.Compose([...])`. Eager geometry validation in
  `__init__`; `__slots__`-bound (no state beyond config). Per
  [ADR 0002](docs/ADR/0002-patchify-transform.md).
- **`reconstruct(patches, image_shape, stride, dilation=1)`** — inverse
  of `extract` via `F.fold` plus a fold-of-ones count map. Bit-exact
  round-trip for `stride == patch_size`; weighted-exact for overlap.
  Rejects `dilation != 1` and `stride > patch_size` (partial coverage
  is forbidden by design — synthesizing pixel values is not PatchKit's
  job).
- **`pair(lr_image, hr_image, lr_patch_size, scale_factor, stride, *, image_id=None)`**
  — LR/HR patch correspondences. Returns a frozen `PatchPair`
  dataclass with `lr_patches`, `hr_patches`, `metas`. Integer
  `scale_factor` only. LR and HR must share `C`, dtype, and device.
- **`PatchPair`**, **`PatchMeta`** — frozen `@dataclass(slots=True)`.
  `PatchMeta` carries `patch_index`, `row`, `col` (LR coords),
  `lr_patch_size`, `hr_patch_size`, `image_id`. CPU-only metadata.
- **`resize(image, target_size, backend="pil", resample=None)`** —
  single-image resize. Output type matches input
  (PIL → PIL, Tensor → Tensor). Cross-backend conversions go through
  a float32 [0, 1] / uint8 hop (numpy intermediate; no torchvision in
  the core). CUDA tensors accepted only with `backend="torch"`.
- **`Cache(root, namespace, version=1)`** — content-addressed disk
  cache. `key_for(*parts) → str`, `put(key, bytes)`, `get(key) → bytes | None`.
  Atomic write via `*.tmp` + `os.replace` with retry on transient
  `PermissionError` (5 attempts on put with exponential backoff
  `0.25/0.5/1/2/4` s — handles OneDrive, antivirus, Windows Search
  races). Optional zstandard compression at level 3 (`[cache]` extra);
  uncompressed fallback when not installed. Sidecar JSON carries
  SHA-256 checksum; corruption surfaces as `OSError`.
- **`num_patches(image_shape, patch_size, stride, dilation=1)`** — the
  patch count formula, exposed as a function. No allocation, no
  tensor. Accepts `(H, W)` or `(C, H, W)`.
- **`tilings(image_shape, *, allow_overlap=False, min_patch_size=2, max_patch_size=None)`**
  — enumerate every square, full-coverage `(patch_size, stride)`
  geometry. Always emits `dilation=(1, 1)`. With default flags returns
  exact tilings only (`patch_size == stride`, divisibility); with
  `allow_overlap=True` adds clean-edge overlap geometries. Truncated
  geometries are deliberately excluded — the function answers "what is
  sound by construction?", not "what will `extract` accept?".
- **`TilingSpec`** — `NamedTuple(patch_size, stride, dilation,
  num_patches, total_patches, overlap)`.

### Added — packaging

- `py.typed` marker (PEP 561): downstream `mypy` now honors PatchKit's
  type hints.
- `[cache]` extra: pulls `zstandard>=0.22` for compressed cache
  entries. Core works without it.

### Out of scope (v0.1.x)

- Multi-image batched API — use a `for` loop, `torch.vmap`, or a
  `DataLoader`. See `Patchify` for `transforms.Compose` integration.
- Dataset orchestration (download, batching, sampling) — the auxiliary
  framework in [`tests/_datasets.py`](tests/_datasets.py) handles this
  for the test suite and `lab/` scripts; it is not shipped in the wheel.
- Channels-last layout, quantization, `nn.Module` integration —
  documented in [`docs/THEORY.md`](docs/THEORY.md) §6 and §8 (open
  questions).

### Documentation

- [`docs/THEORY.md`](docs/THEORY.md) — §0 binding scope, §§1–6 design
  decisions per primitive, §7 resolved questions, §8 open questions,
  §9 the per-API condition contract (Accepts / Rejects / Out of scope)
  that the test suite mirrors.
- [`docs/ADR/0001-patch-extraction-api.md`](docs/ADR/0001-patch-extraction-api.md)
  and [`docs/ADR/0002-patchify-transform.md`](docs/ADR/0002-patchify-transform.md).
- [`README.md`](README.md) — installation, the car-vs-track metaphor,
  validation lab.

[0.2.0]: https://github.com/LeoPR/PatchKit/releases/tag/v0.2.0
[0.1.0]: https://github.com/LeoPR/PatchKit/releases/tag/v0.1.0
