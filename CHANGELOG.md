# Changelog

All notable changes to PatchCraft will be documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Known defects in 0.2.0 (found by a full audit on 2026-08-03, fixes pending)

All four were reproduced against `patchcraft 0.2.0` on `torch 2.13.0+cpu`. The
test suite is green (309 passed) because its blind spots are systematic: it
exercises only divisible geometries, only `float32`/`float64`, and only inputs
already inside `[0, 1]`.

- **`reconstruct` and `stitch` silently zero every pixel the patch grid does
  not cover.** Both validate only the patch *count*
  (`n_patches == num_h * num_w`) and never the *coverage*
  (`(num_h - 1) * sh + ph == h`). A truncated grid therefore returns a
  partly-black image instead of raising, contradicting the bit-exact
  round-trip guarantee in `docs/THEORY.md` §9.2. Measured: `10×10` with
  `patch_size=4, stride=4` returns 36 of 100 pixels zeroed; `13×13` with
  `patch_size=5, stride=5` returns 69 of 169.
- **`stitch(..., weight="hann")` zeroes most of the image, not the four
  corners.** See `docs/THEORY.md` §2.5 for the measurements. `patch_size=2`
  degenerates to an all-zero window and returns an all-black image with no
  error.
- **`resize` corrupts integer dtypes on the torch backend.** `_resize_torch`
  casts back with a bare `out[0].to(original_dtype)`, with no clamp and no
  round. Bicubic legitimately overshoots the input range, so the cast wraps:
  `-9.0 → 247`, `281.9 → 25`. On an 8×8 uint8 hard edge resized to 32×32,
  256 of 1024 pixels are wrong, with black pixels becoming 254 and white
  pixels becoming 1. Separately, the default `pil` backend silently clamps
  out-of-range floats: `resize(torch.full((3, 8, 8), 200.0), (4, 4))` returns
  all `1.0`.
- **`per_patch_mse` / `per_patch_psnr` do not promote to `float64`** the way
  `patch_metrics` does. `uint8` input raises a raw torch `RuntimeError` with
  no mention of PatchCraft, and `float16` input silently returns `inf`.

Also recorded: `float16` overflows to `inf` in `reconstruct`/`stitch`
(`docs/THEORY.md` §9.2), integer dtypes are unsupported end to end rather than
merely `uint8` (§9.1), and `WeightKind` is unreachable from the public
namespace despite appearing in `stitch`'s signature.

### Fixed

- The sdist no longer ships `lab/` or `.vscode/`. The published
  `patchcraft-0.2.0.tar.gz` contains `lab/usage_demo.py`,
  `lab/usage_demo.out` and `lab/2026-05-16-roundtrip-mnist.py`, three files
  that `lab/.gitignore` excludes from version control, which made that sdist
  impossible to reproduce from any commit. The wheel was never affected.
- `.github/workflows/release.yml` passes `skip-existing: true` to
  `gh-action-pypi-publish`, so re-running the pipeline on an already-published
  version is a no-op instead of a hard failure that also skips the
  GitHub Release job. The `validate` job now refuses to proceed when the tag
  does not match `patchcraft.__version__`.
- Both workflows run `uv sync --locked`, so `uv.lock` is now actually enforced
  in CI rather than being advisory.

## [0.2.0] 2026-05-17

Second public release. Adds three feature groups motivated by the QPatchSR
super-resolution consumer plus internal ergonomics. No breaking changes vs
v0.1.0, so all v0.1.0 imports keep working.

### Changed: package name (twice, both on 2026-05-17)

The project shipped v0.1.0 to GitHub under the name **PatchKit** and was
renamed twice in the run-up to the first PyPI upload, each time because the
name was already taken:

1. `patchkit` → `patchforge` (commit `f761834`), because `pypi.org/project/patchkit/`
   belongs to an unrelated model-patching utility.
2. `patchforge` → `patchcraft` (commit `627a9c8`, final), because
   `pypi.org/project/patchforge/` was also taken, by a llama-server CLI.

Both renames were done by string substitution across the tree, which rewrote
the historical entries below: the v0.1.0 section of this file says
`patchcraft`, but `git show v0.1.0:pyproject.toml` says `name = "patchkit"`
and `git ls-tree v0.1.0 src/` says `src/patchkit`. **v0.1.0 was never
published to PyPI under any name**: `patchcraft` 0.2.0 is the first and only
PyPI release, so no import path that ever existed on PyPI has changed, and no
migration shim is needed. Read the v0.1.0 section as a record of *what the API
was*, not of *what the package was called*.

### Added: cross-resolution geometry (THEORY §1.5, §9.7)

Motivated by the QPatchSR consumer's question: "given two image shapes
(LR and HR of the same source), what `(patch_size, stride)` on each
side yields the same number of patches with corresponding regions?"
Three new helpers in `patchcraft.geometry`:

- **`scale_factor(lr_shape, hr_shape) -> int | None`** returns the
  integer `k` such that `hr.shape[-2:] == (k * lr.shape[-2], k *
  lr.shape[-1])`, or `None`. Accepts `(H, W)` or `(C, H, W)`. Pre-
  flight check for `pair`.
- **`paired_tilings(lr_shape, hr_shape, *, allow_overlap=False, ...)`**
  enumerates every `(lr_spec, hr_spec)` pair where both fully cover
  their respective image and produce identical patch counts. Patch
  `k` on each side covers the same image region.
  Example: `paired_tilings((14, 14), (28, 28))` returns three pairs:
  `(p_lr=2, p_hr=4, total=49)`, `(p_lr=7, p_hr=14, total=4)`,
  `(p_lr=14, p_hr=28, total=1)`.
- **`PairedTilingSpec(lr, hr, scale_factor)`** is a `NamedTuple` carrying
  both sides and the discovered scale factor.

### Added: patch-level pixel metrics (THEORY §1.6, §9.8)

Canonical reductions so consumers don't reinvent slightly-divergent
versions in every project. New module `patchcraft.metrics`:

- **`patch_metrics(a, b, *, max_value=1.0) -> dict[str, float]`**:
  scalar `mae`, `mse`, `max_abs`, `psnr_db` over the full tensor
  (any matching shape works). Internal accumulation in `float64`
  for stability; PSNR returns `+inf` for identical inputs.
- **`per_patch_mse(a, b) -> Tensor[L]`** gives one MSE per patch in a
  `(L, C, h, w)` stack.
- **`per_patch_psnr(a, b, *, max_value=1.0) -> Tensor[L]`** gives one
  PSNR per patch. Identical patches yield `+inf` via `torch.where`
  (no clamp tricks).

Explicitly **not** included: SSIM, MS-SSIM, LPIPS, FID, any windowed
or learned metric. Use `pytorch-msssim`, `lpips`, `clean-fid` on the
caller side ([SCOPE.md](docs/SCOPE.md) §4.3 explains the boundary).

### Added: patch stitching for modified patches (THEORY §2.5, §9.9)

`reconstruct` is the bit-exact inverse of `extract`. When patches have
been modified (model output, denoised, super-resolved), averaging them
back uniformly shows visible seams at patch boundaries. `stitch` is the
seam-aware counterpart: it folds patches weighted by a 2-D window
kernel so each pixel "trusts" patches closer to its center more.

- **`stitch(patches, image_shape, stride, *, weight="uniform"|"hann"|"gaussian", dilation=1)`**
  keeps the same `F.fold` geometry and rejections as `reconstruct`; adds a
  weighted-blend numerator over a weighted-sum denominator. With
  `weight="uniform"` it is mathematically equivalent to `reconstruct`
  (covered by a bit-exact equality test on no-overlap and `allclose`
  on overlap). With `"hann"` it strongly suppresses seams at the
  cost of zeroing image corners that fall on Hann's edge-weight-zero
  region (documented artifact). With `"gaussian"`
  (`sigma = max(1, min(ph, pw) / 4)`) it blends smoothly with no
  corner artifact.

Floating-point patches only, because window kernels are float-valued and we
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
  (stitch: math, kernels, why it is separate), §9.7
  (paired tilings contract), §9.8 (metrics contract), §9.9 (stitch
  contract).

## [0.1.0] 2026-05-16

First public release. Public API stable; signatures will only change in 1.x.

### Added: core (one image at a time)

- **`extract(image, patch_size, stride, dilation=1)`** gives patches from a
  `(C, H, W)` tensor via `torch.nn.functional.unfold`. Truncation-only
  boundary; returns `Tensor[0, C, ph, pw]` when geometry fits no patch.
  Per [ADR 0001](docs/ADR/0001-patch-extraction-api.md).
- **`Patchify(patch_size, stride, dilation=1)`** is the callable wrapper for
  `torchvision.transforms.Compose([...])`. Eager geometry validation in
  `__init__`; `__slots__`-bound (no state beyond config). Per
  [ADR 0002](docs/ADR/0002-patchify-transform.md).
- **`reconstruct(patches, image_shape, stride, dilation=1)`** is the inverse
  of `extract` via `F.fold` plus a fold-of-ones count map. Bit-exact
  round-trip for `stride == patch_size`; weighted-exact for overlap.
  Rejects `dilation != 1` and `stride > patch_size` (partial coverage
  is forbidden by design, because synthesizing pixel values is not PatchCraft's
  job).
- **`pair(lr_image, hr_image, lr_patch_size, scale_factor, stride, *, image_id=None)`**
  gives LR/HR patch correspondences. Returns a frozen `PatchPair`
  dataclass with `lr_patches`, `hr_patches`, `metas`. Integer
  `scale_factor` only. LR and HR must share `C`, dtype, and device.
- **`PatchPair`**, **`PatchMeta`** are frozen `@dataclass(slots=True)`.
  `PatchMeta` carries `patch_index`, `row`, `col` (LR coords),
  `lr_patch_size`, `hr_patch_size`, `image_id`. CPU-only metadata.
- **`resize(image, target_size, backend="pil", resample=None)`**:
  single-image resize. Output type matches input
  (PIL → PIL, Tensor → Tensor). Cross-backend conversions go through
  a float32 [0, 1] / uint8 hop (numpy intermediate; no torchvision in
  the core). CUDA tensors accepted only with `backend="torch"`.
- **`Cache(root, namespace, version=1)`** is a content-addressed disk
  cache. `key_for(*parts) → str`, `put(key, bytes)`, `get(key) → bytes | None`.
  Atomic write via `*.tmp` + `os.replace` with retry on transient
  `PermissionError` (5 attempts on put with exponential backoff
  `0.25/0.5/1/2/4` s, which handles OneDrive, antivirus, Windows Search
  races). Optional zstandard compression at level 3 (`[cache]` extra);
  uncompressed fallback when not installed. Sidecar JSON carries
  SHA-256 checksum; corruption surfaces as `OSError`.
- **`num_patches(image_shape, patch_size, stride, dilation=1)`** is the
  patch count formula, exposed as a function. No allocation, no
  tensor. Accepts `(H, W)` or `(C, H, W)`.
- **`tilings(image_shape, *, allow_overlap=False, min_patch_size=2, max_patch_size=None)`**
  enumerates every square, full-coverage `(patch_size, stride)`
  geometry. Always emits `dilation=(1, 1)`. With default flags returns
  exact tilings only (`patch_size == stride`, divisibility); with
  `allow_overlap=True` adds clean-edge overlap geometries. Truncated
  geometries are deliberately excluded, because the function answers "what is
  sound by construction?", not "what will `extract` accept?".
- **`TilingSpec`** is a `NamedTuple(patch_size, stride, dilation,
  num_patches, total_patches, overlap)`.

### Added: packaging

- `py.typed` marker (PEP 561): downstream `mypy` now honors PatchCraft's
  type hints.
- `[cache]` extra: pulls `zstandard>=0.22` for compressed cache
  entries. Core works without it.

### Out of scope (v0.1.x)

- Multi-image batched API: use a `for` loop, `torch.vmap`, or a
  `DataLoader`. See `Patchify` for `transforms.Compose` integration.
- Dataset orchestration (download, batching, sampling): the auxiliary
  framework in [`tests/_datasets.py`](tests/_datasets.py) handles this
  for the test suite and `lab/` scripts; it is not shipped in the wheel.
- Channels-last layout, quantization, `nn.Module` integration:
  documented in [`docs/THEORY.md`](docs/THEORY.md) §6 and §8 (open
  questions).

### Documentation

- [`docs/THEORY.md`](docs/THEORY.md) has §0 binding scope, §§1–6 design
  decisions per primitive, §7 resolved questions, §8 open questions,
  §9 the per-API condition contract (Accepts / Rejects / Out of scope)
  that the test suite mirrors.
- [`docs/ADR/0001-patch-extraction-api.md`](docs/ADR/0001-patch-extraction-api.md)
  and [`docs/ADR/0002-patchify-transform.md`](docs/ADR/0002-patchify-transform.md).
- [`README.md`](README.md) covers installation, the car-vs-track metaphor,
  validation lab.

[0.2.0]: https://github.com/LeoPR/PatchCraft/releases/tag/v0.2.0
[0.1.0]: https://github.com/LeoPR/PatchCraft/releases/tag/v0.1.0
