# Changelog

All notable changes to PatchKit will be documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0]: https://github.com/LeoPR/patchkit/releases/tag/v0.1.0
