# Changelog

All notable changes to PatchCraft will be documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `.pre-commit-config.yaml`, running the gates CI runs. `ruff-format` is
  deliberately absent: the project has never used it and adopting it would
  reformat 36 of 65 files in a commit that changed no behaviour. Two things
  surfaced while wiring it up. `docs/FOCO-1.0.md` had no trailing newline, now
  fixed with the file's own CRLF rather than a bare LF, which is a mistake the
  `mixed-line-ending` hook caught on the first attempt. And on a machine with a
  global `core.hooksPath`, `pre-commit install` writes a hook git then ignores,
  so it is inert without saying so; `CONTRIBUTING.md` gives the one command
  that tells you which case you are in.
- `CITATION.cff`, validated against Citation File Format 1.2.0, so GitHub shows
  the "Cite this repository" button and the metadata has one authoritative
  home. It ships in the sdist. The BibTeX entry in GUIDE section 9 stays,
  because `cffconvert`'s generated form drops the version and emits a
  placeholder key, but the `.cff` is now the source and the release checklist
  moves both. A test pins its version against the newest released section of
  this file, since comparing it to `patchcraft.__version__` cannot work in a
  checkout, where setuptools-scm resolves a development version.

### Fixed

- **THEORY §9 justified promoting `bfloat16` with a fact that is false of it,
  and §9 is the arbiter of the contract.** Both half formats were promoted to
  a `float32` accumulator citing `float16`'s finite maximum of 65504. Only
  `float16` overflows there: on the documented case its numerator reaches
  `inf` in 144 of 256 pixels, while `bfloat16` carries `float32`'s exponent
  and peaked at `9.0112e+04` against a finite maximum of `3.390e+38`. What the
  promotion buys `bfloat16` is precision, since it has 8 mantissa bits to
  `float16`'s 11, and on ordinary [0, 1] data it cut the maximum error from
  `9.262e-03` to `1.935e-03`. §9.2 now carries one entry per format with the
  measurement behind each, and the two docstrings and GUIDE section 4 say the
  same. Two tests pin it.
- Closed blocker B5 in `docs/FOCO-1.0.md`. The contradiction it named, §9.2
  accepting the half-precision promotion in one list and excluding it as out
  of scope eight lines below, had already been removed in 0.3.0 and never
  recorded. Checking that is what surfaced the `bfloat16` error above.


## [0.5.2] - 2026-09-03

### Fixed

- **`stitch(weight="hann")` corrupted the corner band of every patch of 99 or
  more.** The denominator was floored by an absolute `clamp(min=1e-6)` whose
  comment called it a defensive no-op. It was not: the 2-D hann corner weight
  is `(pi/(n+1))**4`, which crosses below `1e-6` at patch 99, so the corner
  band was divided by the floor rather than by the real weight. Measured at
  640x640, patch 256, stride 128: maximum error 0.94 on data in [0, 1] and 960
  wrong pixels, identical in float64 because the constant and not the precision
  was the cause. The floor is now `finfo(dtype).tiny`, the model already used
  in `metrics.py`, which never reaches a legitimate weight: the smallest 2-D
  hann corner even at patch 1024 is 8.8e-11. THEORY 9.9 claimed no covered
  pixel is zeroed by the window, which was true of the window and false of the
  result.
- **`reconstruct` returned a view aliasing the caller's patches on a
  single-patch grid.** Writing to the patches afterwards changed an image the
  function had already returned, measured as 0.744 becoming 99.0. `extract`
  guards the mirror image of this case and `reconstruct` did not.
- **`resize(..., backend="pil")` raised a raw `RuntimeError` on `int8`.** The
  inbound `clamp(0, 255)` was applied in the input dtype, where the bound 255
  is unrepresentable, so torch raised `value cannot be converted to type
  int8_t without overflow` before clamping anything. Every integer dtype now
  works on both backends, and THEORY 9.4 gained the row for integer tensors on
  the PIL backend, which it had never specified.


### Removed

- The agentic workflow's leftovers. `.superpowers/` at the repository root
  held 62 files of review diffs and task-progress notes, hidden from
  `git status` by a nested `.gitignore` containing `*`, so it sat in the tree
  unseen and reached the 0.5.0 sdist. The three implementation plans under
  `docs/superpowers/plans/` were 4527 lines of checkbox task lists addressed
  "For agentic workers", and they shipped to PyPI inside every sdist.

### Added

- `docs/ADR/0004-precision-and-effort-parameters.md`, in `Proposed` status: names the
  precision policy the library already applies in five places, and designs an `effort=`
  preset under the constraint that it may change speed and memory but never a returned
  value. Nothing is implemented; the ADR records the design, the effort per step and the
  order. It amends ADR 0003 on one measured point, that the exactness predicate is a
  property of the geometry and the accumulator rather than of the geometry alone.
- `outreach/`, material for presenting the project publicly, in the shape the
  sibling projects use: a dated news source at the root and one subfolder per
  channel, with the rule that no channel text changes before the source does.
  It publishes no new measurement, and it is pruned from the sdist.

### Changed

- The three design specs that were under `docs/superpowers/specs/` moved to
  `docs/design/`. They are the record of what was measured before each phase
  was built, including the spike table that rejected `conv_transpose2d` and
  `index_add_` before the Rust kernel was written, and Amendment A, which
  `tests/test_exactness.py` cites for the per-pixel error bound.
- The sdist is 264 KiB where it was 316 KiB.
- Nothing ephemeral is kept in the project folder any more. `accel/target`
  had reached 269 MiB of Rust build output, and `dist/`, `.pytest_cache` and
  `src/patchcraft.egg-info` were accumulating alongside it. The caches are
  now redirected by user-level environment variables, joining the ones this
  machine already had for uv, ruff, mypy, pip and `__pycache__`, and
  `CONTRIBUTING.md` records the convention and why it is not committed here.
- `.vscode/settings.json` stays tracked and is pruned from the sdist instead.
- **The versioning rule was self-contradictory and is corrected.** One row said
  an output value change is a `0.Y.0`, the next said a fix is a `0.y.Z`, and a
  bug fix satisfies both, so the table made every fix a minor and left patch
  releases unable to fix anything. It also contradicted the project's own
  history: `0.2.1` changed `stitch(weight="hann")` output for every geometry
  and shipped as a patch. The rule now turns on whether the contract moved or
  the implementation stopped violating it, which sorts `0.2.1`, `0.3.0`,
  `0.5.0` and `0.5.2` correctly. `CONTRIBUTING.md` carries the rule with those
  four worked, and GUIDE section 8 item 6 no longer cites a patch release as
  an example of what only a minor may do.
  Its `${workspaceFolder}/.venv` path is not broken: `.venv` is a directory
  junction to the real environment outside the tree, so the setting resolves
  and is portable to any checkout following the same convention.

## [0.5.1] - 2026-09-02

> Numbered `0.6.0` while it was being built. Corrected before release: the
> middle digit is the compatibility boundary, and this release changes no
> signature, no name and no output value. See the versioning policy and its
> errata in [`CONTRIBUTING.md`](CONTRIBUTING.md).

### Changed

- **The native accelerator ships inside this wheel.** It was going to be a
  second PyPI project, `patchcraft-accel`, reached through
  `pip install patchcraft[accel]`. That project was never published and now
  never will be. A release produces one sdist and six wheels for the single
  `patchcraft` project: five tagged `cp312-abi3-<platform>` with the Rust
  kernel compiled in as `patchcraft._accel_native`, and one `py3-none-any`
  for every other platform. Installers prefer the most specific compatible
  tag, so `pip install patchcraft` picks up the accelerator where one exists
  and the pure wheel everywhere else, with no extra to remember. The pattern
  and its universal fallback follow `coverage`, which publishes the same two
  shapes for one version.
- The `accel` extra is removed. Nothing depended on it, because it never
  reached the index.
- The accelerator is now on by default where a wheel carries it, rather than
  opt-in. The two paths are bit-identical by test, and `PATCHCRAFT_ACCEL=0`
  still forces the pure one at runtime.
- Build backend moves from hatchling to setuptools with setuptools-rust, which
  is what makes one project able to emit both shapes of wheel. `setup.py`
  holds the single decision, driven by `PATCHCRAFT_PURE_PYTHON` and
  `PATCHCRAFT_REQUIRE_EXTENSION`; everything else stays declarative in
  `pyproject.toml`. Installing from the sdist compiles the extension when a
  Rust toolchain is present and degrades to a pure install when it is not.
- License metadata follows PEP 639: the SPDX expression `MIT` with
  `license-files`, and the superseded `License :: OSI Approved` classifier is
  gone. setuptools rejects carrying both.

### Added

- **The version comes from the git tag.** `setuptools-scm` derives it at build
  time, so no file in the source states a version, the release procedure loses
  its bump step, and a tag and a release can no longer disagree.
  `patchcraft.__version__` still works: the build writes a generated
  `_version.py` that `__init__.py` imports, which costs nothing at import time,
  with `importlib.metadata` as a fallback for a tree that was never built.
  Measured before choosing: `importlib.metadata` alone was the wrong default,
  because it took 56 ms on first call and reported a stale version in an
  editable install.
- `docs/PERFORMANCE.md`, which is where the accelerator's measurements live.
  It carries the test machine, the versions, the date, the reproduction
  command and what the numbers do not say, and its table is pasted from
  `tools/benchmark.py --markdown` rather than retyped.
- The Rust kernel is now always built in release mode, including in an
  editable install. It was following the build command, so a development
  checkout compiled a debug kernel: measured here, that turned a 14x speedup
  into 2.1x, which made local benchmarks meaningless and made the accelerated
  CI job exercise code no user runs.
- `patchcraft.__version__` is re-exported explicitly, so `mypy --strict` in a
  consuming project still accepts it. Moving it out of a literal and into an
  import had quietly made it an implicit re-export, which strict mode rejects.
- `tools/benchmark.py` measures the accelerated fold against the pure one and
  refuses to report a timing unless the two results are bit-identical. The
  numbers in the documentation come from it, and it is the reproduction
  command they cite.
- Every build checkout uses `fetch-depth: 0`. The default shallow clone hides
  the tags, and setuptools-scm's fallback for that is a silent
  `0.1.dev1+g<sha>`, a version that walks backwards rather than an error.
- `tools/check_dist.py` grew `--expect-version` and `--require-universal`, and
  it now rejects any artifact carrying a local version segment. PyPI refuses
  those on upload, so this turns a late and confusing server-side rejection
  into an early local one. It also fixes a bug in the job that checks the
  native wheels, which demanded a universal wheel that job never has.
- Linux aarch64 wheels, which the previous four-target matrix did not cover.
- `tools/check_dist.py`, run before every upload. It fails the release if a
  platform wheel lost its extension, which `optional=True` would otherwise
  let through as a silently slow wheel, or if the universal wheel gained one.
- The versioning policy now names the resolver rule it follows, which is that
  `0.y` is the compatibility boundary in Cargo, npm and PEP 440 alike, and it
  carries an errata for the two releases numbered before the rule was written.
- Version numbers are gone from the documentation surface. The three cover
  pages carry none at all, since the PyPI badge already shows the current
  release, and the manual keeps only the one the citation entry needs. The
  release archaeology those pages used to recite lives here, and the rest is
  in the git history.
- `MANIFEST.in`, which puts `accel/` in the sdist so the source path can build
  the extension, and keeps `lab/`, `.superpowers/` and cargo's `target/` out.

- **Corrected: the hand-rolled and library tile-and-blend results are not
  bit-identical.** The three cover pages and GUIDE section 1 claimed they were,
  and the GUIDE block asserted it. Re-measured: they agree to 1.17e-05 on a
  value in [0, 1], differing on 24500 of 49152 elements. The cause is the 0.3.0
  separable denominator, where `stitch` sums two 1-D window folds while the
  hand-rolled twin folds a replicated 2-D kernel. `stitch(weight="uniform")`
  is still bit-identical to `reconstruct`. Nothing about the round-trip
  contract changed; the documentation was simply asserting more than the code
  delivered.
- Documentation corrections found by auditing every page against the code:
  the changelog's reference links did not define the two most recent versions;
  `docs/AUXILIARY.md` listed wheel contents that predate three releases;
  `docs/SCOPE.md` still promised "pure-Python single-threaded", which the
  rayon-parallel kernel contradicts; `docs/ROADMAP.md` still named hatchling;
  GUIDE said the library "is not faster", that the no-zstandard branch is
  untested, and that CI is six cells; the cover pages pointed at `USAGE.md` as
  a walkthrough of 20 symbols when it covers 18 and says so. `CONTRIBUTING.md`
  told a contributor to run `pytest` after a `uv sync` that does not install it.

### Notes

- No signature, no name and no output value changed. The public surface is the
  same 20 names, still pinned by `tests/test_public_api.py`.
- Measured on torch 2.14, CPU, mean over the dev machine: `reconstruct` at
  3x512x512 p=32 s=16 goes 11.4 ms to 2.5 ms (4.5x), at 3x2048x2048 p=64 s=32
  477.9 ms to 31.8 ms (15.0x); `stitch[hann]` 21.6 ms to 7.0 ms (3.1x) and
  488.7 ms to 38.8 ms (12.6x). The gain is algorithmic rather than a thread
  count: the pure `F.fold` measured between 365 and 465 ms on that largest
  case at 4, 8, 16 and 36 torch threads.

## [0.5.0] - 2026-09-02

### Added

- Python 3.14 is supported and declared. The classifiers list it, the CI
  matrix runs the full suite plus `ruff check` and `mypy --strict` against it
  on Ubuntu and Windows, and all six cells are green. The floor stays at 3.12
  because `cache.py` uses PEP 695 generic syntax, which 3.11 cannot parse.

### Changed

- `tilings`/`paired_tilings` no longer emit degenerate single-patch overlap
  specs: with one patch the stride is unobservable and the spec duplicated
  the exact tile. `tilings((28, 28), allow_overlap=True)` now returns 73
  specs instead of 100; `paired_tilings((14, 14), (28, 28), allow_overlap=True)`
  returns 27 instead of 40.
- The development lock moved to torch 2.14.0 and torchvision 0.29.0, with the
  linters and the rest of the toolchain along with it. The full
  126,736-geometry exactness sweep passes on torch 2.14, so the count-map
  contract is unchanged by the upgrade. Runtime requirements are untouched and
  stay at `torch>=2.6`.

### Fixed

- **Public retraction: the documented exactness boundary was wrong.** The old
  wording (`k_max <= 4`, "`stride == patch_size / 2`", "within ~1 ULP" outside
  the rule) was measured false: the error grows with the pixel's coverage
  count (up to 19 ULP at count 81 in float32). The correct contract
  (ADR 0003): the `extract`/`reconstruct` round trip is bit-exact iff every
  value of the overlap count map is a power of two, which is always true
  at `stride == patch_size`, and outside the rule the per-pixel error is
  bounded by `(k+1)·eps·|v|`, with `k` the pixel's coverage count.
  Docstrings and docs (SCOPE, THEORY, GUIDE, USAGE, READMEs) now state this
  form; THEORY §9.1 records the extract-truncates / reconstruct-rejects
  asymmetry.
- Packaging: the sdist no longer carries `.superpowers/`, an untracked
  directory of agent tooling that hatchling swept in (62 files). The same
  class of leak as the `lab/` files that reached PyPI in 0.2.0.
- Documentation: GUIDE section 6 still printed the pre-0.5.0 enumeration
  output and described the single-patch duplicates as an open wart; section 7
  asserted `__version__ == "0.4.0"`; section 8 and the BibTeX entry quoted
  0.4.0 and its test counts. All four now match the release.
- Prose: the em-dashes that entered the docstrings, docs and tests during
  0.3.0 to 0.5.0 are rewritten as ordinary sentences, per the project's
  writing rule.
- Test suite: round-trip assertions now run on seeded full-mantissa noise
  (integer ramps and widened-float32 data could mask ULP-level errors); a
  falsification suite (`tests/test_exactness.py`) enumerates the 126,736
  legal geometries, samples 256 (seeded; full sweep via
  `PATCHCRAFT_SWEEP_FULL=1`), and pins both halves of the predicate; a naive
  loop-based reference (`tests/test_reference.py`) cross-checks the fast
  paths; the 20-name public surface is frozen by `tests/test_public_api.py`;
  the no-`zstandard` cache branch is now covered.

### Notes

- No signature changed. The public surface is exactly the 20 names of 0.4.0,
  now pinned by test.

## [0.4.0] - 2026-09-01

> Source milestone, never uploaded to PyPI. Its contents reached
> users inside 0.5.0. See the versioning policy in
> [`CONTRIBUTING.md`](CONTRIBUTING.md) for why the number was minted
> early and why that no longer happens.


### Added

- Optional native accelerator `patchcraft-accel` (`pip install patchcraft[accel]`):
  a Rust/pyo3 kernel for the overlap fold of `reconstruct`/`stitch`. Prebuilt
  abi3 wheels (Python 3.12+) for Windows x64, Linux x86_64 (manylinux), macOS
  arm64 and macOS x86_64; self-contained, no system dependencies, no torch
  linkage. When it is absent, ABI-mismatched, ineligible (CUDA/integer
  tensors), or disabled via `PATCHCRAFT_ACCEL=0`, every path falls back
  silently to pure torch.
- `patchcraft.accel_available()` reports whether the accelerator is active.

### Performance

- Overlap fold with the accelerator, mean of `lab/bench_phase2.py` on the
  dev machine: reconstruct 3x512x512 p=32 s=16 17.727 ms -> 2.609 ms (6.80x);
  stitch[hann] same geometry 23.714 ms -> 8.894 ms (2.67x); 3x2048x2048 p=64
  s=32 reconstruct 469.814 ms -> 32.458 ms (14.47x). The accelerated `stitch`
  also skips the full pre-multiply pass over the patches tensor.

### Notes

- Numeric equivalence accel vs pure: bit-exact (`torch.equal`) across the
  test grid (overlap x rectangular x stride∤patch x C in {1,3,4} x f32/f64).
- Linux validation: wheel
  `patchcraft_accel-0.1.0-cp312-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`
  installed and the test suite passed accelerated on Ubuntu 26.04 (WSL),
  built via the manylinux maturin container.

## [0.3.0] - 2026-09-01

> Source milestone, never uploaded to PyPI. Its contents reached
> users inside 0.5.0. See the versioning policy in
> [`CONTRIBUTING.md`](CONTRIBUTING.md) for why the number was minted
> early and why that no longer happens.


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

## [0.2.2] - 2026-08-30

Documentation release. No behaviour changes, so every call that worked on 0.2.1
works identically here, and the only file under `src/` that changed is a
docstring.

### Fixed

- **`reconstruct`'s docstring claimed the overlap round trip returns the
  original image.** It said each pixel comes back as "the average of all patches
  covering it, the same as the original when patches came from `extract`
  unmodified". The second half is false. Measured on a 16x16 image with
  `patch_size=4, stride=1`, unmodified patches straight from `extract` come back
  with `torch.equal` False and a maximum absolute error of 2.4e-07 in float32.

  The docstring now states the real rule: the round trip is bit-exact when every
  value in the count map is a power of two, because dividing a float by a power
  of two is the one division that never rounds. It also says that widening the
  dtype does not rescue it, since the deciding axis is the geometry rather than
  the precision.
- **`pair()`'s docstring double-applied the stride, and could silently
  misplace every patch for a caller who trusted it.** It said patch `k` has
  its top-left at LR pixel `(row * sh_lr, col * sw_lr)`, but `PatchMeta.row`
  and `PatchMeta.col` already have the stride applied: they are built as
  `(k // num_w_lr) * sh_lr` and `(k % num_w_lr) * sw_lr`. Following the
  docstring multiplied by the stride a second time and landed on a different
  patch. Only the text was wrong; the code, the tests and the runtime
  behaviour were correct throughout, so no output of any released version
  changes.

  The failure mode was quiet, which is why it is worth a changelog entry: no
  exception is raised, the patch exists, and only its recorded position is
  wrong, so a consumer using patch position as a feature gets a worse model
  rather than a crash. It is also invisible at `stride == 1`, where the grid
  index and the pixel coordinate coincide, which is why it survived review.

  Root cause was a name collision rather than a typo. `row`/`col` mean grid
  indices in `THEORY.md` §1 (where `(row · sh, col · sw)` is the correct
  pixel formula) and pixel coordinates in `PatchMeta`. The `pair()` docstring
  took §1's formula and applied it to `PatchMeta`'s fields. `THEORY.md` §3
  now flags the collision explicitly instead of carrying both meanings eight
  lines apart, and `tests/test_pair.py` gains
  `test_meta_coords_are_pixels_not_grid_indices`, which uses `stride != 1`
  and asserts both that the stored coordinate indexes the right region and
  that the discarded formula indexes a different one.

  Reported from an API review done for the QSR prototype, which uses patch
  position as a context feature.

### Changed

- **The README is now a call page and the manual moved to `docs/GUIDE.md`.** The
  README had reached 864 lines, which made it a manual printed on the front
  door. It is 156 lines now, and it answers what the library is, where you are
  getting into, and which of `reconstruct` or `stitch` you want. Nothing was
  discarded: every measurement, table and long example moved into the guide,
  which gained its own table of contents.
- **PyPI gets its own page.** `pyproject.toml` publishes `README.pypi.md`, which
  stands alone for a reader arriving from a search, and carries only absolute
  links because PyPI drops relative ones.
- **A Portuguese cover page joins the English one.** `README.pt-BR.md` is the
  translation, `README.md` stays canonical, both carry an l10n colophon sharing
  one `doc_id`, and git measures whether the translation is current.

## [0.2.1] 2026-08-04

Bugfix release. Closes the full correctness backlog found by the 0.2.0 audit
(2026-08-03). Every fix below was first reproduced as a failing regression
test measuring the same quantities quoted from the audit, then fixed, then
verified by round-trip lab scripts under `lab/2026-08-04-*.py` (reports in
`Z:\outputs\patchcraft\2026-08-04-*\`). Test suite: 309 → 345 passed. No new
features; the only namespace change is `WeightKind` becoming reachable, which
was itself an audit defect.

### Fixed: the four audit defects

- **`reconstruct` and `stitch` silently zeroed every pixel the patch grid did
  not cover.** Both validated only the patch *count*
  (`n_patches == num_h * num_w`) and never the *coverage*
  (`(num_h - 1) * sh + ph == h`). A truncated grid returned a partly-black
  image instead of raising, contradicting the bit-exact round-trip guarantee
  in `docs/THEORY.md` §9.2. Measured on 0.2.0: `10×10` with `patch_size=4,
  stride=4` returned 36 of 100 pixels zeroed; `13×13` with `patch_size=5,
  stride=5` returned 69 of 169. Both functions now reject any grid whose last
  patch does not end exactly on the image edge, with a `ValueError` naming the
  covered vs. requested extents and pointing at `patchcraft.tilings`.
- **`stitch(..., weight="hann")` zeroed most of the image, not the four
  corners.** The old `_hann_1d(n)` was the symmetric window
  `0.5·(1 − cos(2πi/(n−1)))`, exactly zero at both endpoints. Measured on
  0.2.0: `12×12, patch 4, stride 4` came back with 108 of 144 pixels zeroed;
  `patch_size=2` degenerated to an all-zero window and returned an all-black
  image with no error. The window is now the interior of a longer symmetric
  Hann window, `hann_window(n + 2, periodic=False)[1:-1]`, strictly positive
  on every sample (THEORY §2.5). Output values of `stitch(hann)` change for
  every geometry; round-trip of unmodified patches remains exact.
- **`resize` corrupted integer dtypes on the torch backend.** `_resize_torch`
  cast back with a bare `out[0].to(original_dtype)`, with no clamp and no
  round. Bicubic legitimately overshoots the input range, so the cast wrapped:
  `-9.0 → 247`, `281.9 → 25`. Measured on 0.2.0: an 8×8 uint8 hard edge
  resized to 32×32 had 256 of 1024 pixels wrong, black pixels becoming 254
  and white pixels becoming 1. The cast back now rounds and clamps to the
  `torch.iinfo` range of the integer dtype. Separately, the `pil` backend's
  clamp of out-of-range floats to `[0, 1]` (the uint8 hop) is now documented
  behavior in the docstring and THEORY §9.4 instead of a silent surprise.
- **`per_patch_mse` / `per_patch_psnr` did not promote to `float64`** the way
  `patch_metrics` does. On 0.2.0, `uint8` input raised a raw torch
  `RuntimeError` with no mention of PatchCraft, and `float16` input silently
  returned `inf`. Both now compute in `float64` for any input dtype and always
  return `float64` (contract change recorded in THEORY §9.8).

### Fixed: secondary defects recorded by the audit

- **`reconstruct` and `stitch` overflowed to `inf` on `float16`/`bfloat16`.**
  `F.fold` accumulates the sum of all overlapping patches before the count-map
  division, and the numerator exceeds the fp16 finite max (65504) well before
  the division happens. Measured on 0.2.0: fp16 constant image of `10000.0`,
  `patch 3, stride 1`, returned `inf` in 144 of 256 pixels. Half-precision
  inputs now accumulate internally in `float32` (fold, count map, division)
  and are cast back to the original dtype on return.
- **`reconstruct` rejected integer dtypes with a raw torch
  `NotImplementedError`** (`col2im_out_cpu` not implemented for
  `Byte`/`Int`/`Long`). It now raises a clear `ValueError` up front, matching
  the `stitch` guard.
- **`WeightKind` was unreachable from the public namespace** despite appearing
  in `stitch`'s signature. It is now exported by `patchcraft.stitch` and
  re-exported by `patchcraft` (public API: 18 → 19 symbols).

### Fixed: CI and packaging (carried over from the audit pass)

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
  (per-axis `sigma = max(1.0, n / 4.0)`) it blends smoothly with no
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

[Unreleased]: https://github.com/LeoPR/PatchCraft/compare/v0.5.2...HEAD
[0.5.2]: https://github.com/LeoPR/PatchCraft/releases/tag/v0.5.2
[0.5.1]: https://github.com/LeoPR/PatchCraft/releases/tag/v0.5.1
[0.5.0]: https://github.com/LeoPR/PatchCraft/releases/tag/v0.5.0

<!-- 0.4.0 and 0.3.0 are deliberately unlinked: they are source milestones that
     were never tagged and never released, as their own entries say. -->
[0.2.2]: https://github.com/LeoPR/PatchCraft/releases/tag/v0.2.2
[0.2.1]: https://github.com/LeoPR/PatchCraft/releases/tag/v0.2.1
[0.2.0]: https://github.com/LeoPR/PatchCraft/releases/tag/v0.2.0
[0.1.0]: https://github.com/LeoPR/PatchCraft/releases/tag/v0.1.0
