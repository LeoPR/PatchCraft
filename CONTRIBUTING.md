# Contributing to PatchCraft

If you've cloned this repo and want to run the test suite, understand how the project is laid out, or follow the validation conventions, this is the page. For installation and usage, see the [README](README.md).

---

## Run tests

The test runner and the linters live in the `dev` extra, so a plain `uv sync`
does not install them:

```
uv sync --locked --extra cache --extra dev
```

The same gates run before a commit if you install the hooks:

```
uvx pre-commit install
uvx pre-commit run --all-files     # once, to see the tree is clean
```

**Check that the hook is actually live**, because on some machines it is not.
`pre-commit install` writes `.git/hooks/pre-commit`, and git ignores that file
entirely when a global `core.hooksPath` is set:

```
git config --get core.hooksPath        # empty means the hook will run
```

If it prints a path, the hooks in it run instead of yours, and `pre-commit`
is silently inert here. Either make that global hook delegate, or set
`core.hooksPath` for this clone alone, remembering that doing so also disables
whatever else the global hooks were doing.

`ruff-format` is deliberately not among the hooks. This project has never used
it, and adopting it would reformat 36 of 65 files in a commit that changed no
behaviour.

```
pytest
pytest -m "not gpu"        # skip GPU-requiring tests
```

CI runs the full suite on every push and PR. See [`.github/workflows/test.yml`](.github/workflows/test.yml). Matrix is `{ubuntu-latest, windows-latest} × {python 3.12, 3.13, 3.14}` for the pure path, plus a second job on both operating systems for the accelerated one. To run the same checks locally before pushing:

```
ruff check src tests
mypy --strict src
pytest -m "not gpu"
```

---

## The two paths

The overlapping fold in `reconstruct` and `stitch` has a Rust implementation
in [`accel/`](accel/), compiled into the wheel on the platforms CI builds for.
Everything else runs the same operation in torch. The two are bit-identical by
test, so a contribution has to keep both green rather than pick one.

Which one your checkout runs depends on whether the extension got built:

```
python -c "import patchcraft; print(patchcraft.accel_available())"
```

`uv sync --extra cache --extra dev` builds it when a Rust toolchain is on
PATH, always in release mode. Two environment variables override that, and CI
sets both explicitly rather than relying on what happens to be installed:

| Variable | Effect |
|---|---|
| `PATCHCRAFT_PURE_PYTHON=1` | Skip the extension. How the universal wheel is built. |
| `PATCHCRAFT_REQUIRE_EXTENSION=1` | Make a Rust build failure fatal instead of degrading to the pure path. |
| `PATCHCRAFT_ACCEL=0` | Runtime only: ignore an extension that is already built. |

The Rust kernel has its own tests, which need no Python at all:

```
cargo test --manifest-path accel/Cargo.toml
```

One local wrinkle worth knowing. If your checkout sits under a path with
non-ASCII characters, `setuptools-rust` mis-decodes the artifact path that
cargo reports and the copy step fails. Point the build at an ASCII directory
and it works:

```
CARGO_TARGET_DIR=/tmp/pc-target uv sync --extra cache --extra dev
```

---

## Nothing ephemeral lives in the project folder

Caches, build output and virtual environments are kept off the tree, because
they are large, regenerable and machine-specific. They are redirected by
user-level environment variables rather than by anything committed here: a
machine path in `pyproject.toml` would be wrong for every other machine and
for CI, which is why one was removed in 0.3.0.

| Variable | Sends |
|---|---|
| `CARGO_TARGET_DIR` | the Rust build output, which reached 269 MiB inside `accel/target` before it was redirected |
| `PYTEST_ADDOPTS=-o cache_dir=...` | `.pytest_cache` |
| `RUFF_CACHE_DIR`, `MYPY_CACHE_DIR` | the linter caches |
| `UV_CACHE_DIR`, `PIP_CACHE_DIR` | downloaded wheels |
| `PYTHONPYCACHEPREFIX` | `__pycache__` |

The virtual environment lives outside the tree too; see
[`docs/AUXILIARY.md`](docs/AUXILIARY.md).

Two directories still appear during a build and are gitignored: `build/` and
`src/patchcraft.egg-info/`, both setuptools working state. They exist only
while you build, and redirecting them would need a machine path in a tracked
file, which is the thing this section is avoiding.

Three files are written into `src/patchcraft/` by an editable install and
are gitignored: the compiled `_accel_native`, the generated `_version.py`, and
`__pycache__`. Those have to be there for the package to import.

---

## Layout

```
PatchCraft/
├── pyproject.toml                  package metadata, build backend, cibuildwheel targets
├── setup.py                        the one build decision: with or without the Rust extension
├── tools/check_dist.py             gate: extension present, versions agree, tag respected
├── tools/benchmark.py              accelerated vs pure, and proves they agree first
├── MANIFEST.in                     what the sdist carries
├── README.md                       the call page, canonical, English
├── README.pt-BR.md                 the same call page in Portuguese
├── README.pypi.md                  the PyPI page (long_description), links absolute
├── CONTRIBUTING.md                 this file
├── MAP.md                          one-page wayfinding: the tree, and a want-to/go-to table
├── AGENTS.md                       the canonical rules; eleven invariants, each earned
├── CLAUDE.md                       thin pointer to AGENTS.md plus tool-specific notes
├── STATUS.md                       the snapshot; what is in flight and what waits on whom
├── CHANGELOG.md                    Keep-a-Changelog format
├── LICENSE                         MIT
├── .python-version                 3.13
├── .gitignore                      ignores archive/, venvs, caches, outputs
├── .pre-commit-config.yaml         the CI gates, before the commit instead of after
├── CITATION.cff                    citation metadata; GitHub reads it for the cite button
├── SECURITY.md                     how to report, and what the real attack surface is
├── CODE_OF_CONDUCT.md              Contributor Covenant 2.1, verbatim
├── .github/
│   ├── ISSUE_TEMPLATE/             five forms; the required fields are what makes a report reproducible
│   ├── PULL_REQUEST_TEMPLATE.md    asks whether a returned value moved, which decides the version bump
│   └── workflows/
│   ├── test.yml                    matrix CI on PRs/main, plus the accelerated job
│   └── release.yml                 publishes to PyPI on vX.Y.Z tag push (Trusted Publishing)
├── src/patchcraft/                 library core, one-image-at-a-time primitives
│   ├── __init__.py                 re-exports the full public API and the generated version
│   ├── extract.py                  patches via strided view or F.unfold; Patchify (ADR 0002)
│   ├── reconstruct.py              inverse via F.fold + count map, closed-form fast path
│   ├── stitch.py                   weighted reassembly for modified patches
│   ├── geometry.py                 pre-flight: num_patches, tilings, scale_factor, paired_tilings
│   ├── metrics.py                  patch_metrics, per_patch_mse, per_patch_psnr
│   ├── pair.py                     LR↔HR pairing; PatchPair, PatchMeta
│   ├── resize.py                   resize with PIL or torch backends
│   ├── cache.py                    content-addressed disk cache
│   ├── _accel.py                   bridge to the optional native accelerator, falls back silently
│   └── _foldgeom.py                shared fold-geometry validation for reconstruct and stitch
├── accel/                          the Rust crate compiled into this wheel (pyo3, setuptools-rust)
│   ├── Cargo.toml                  crate metadata; the version there is internal
│   ├── src/kernel.rs               the gather-fold kernel, pure Rust, cargo-tested
│   ├── src/lib.rs                  pyo3 glue, exports patchcraft._accel_native
│   └── README.md                   how to build it, and how to check it is active
├── tests/                          pytest suite (contract tests for src/)
│   ├── test_extract.py             extract + Patchify
│   ├── test_reconstruct.py
│   ├── test_stitch.py
│   ├── test_geometry.py            num_patches + tilings + scale_factor + paired_tilings
│   ├── test_metrics.py
│   ├── test_pair.py
│   ├── test_resize.py
│   ├── test_cache.py
│   ├── test_exactness.py           falsifies the count-map predicate over the legal space
│   ├── test_reference.py           naive loop-based reference for the fast paths
│   ├── test_public_api.py          freezes the 20 names, their signatures and carrier fields
│   ├── test_accel.py               native path, skipped unless the extension was built
│   ├── test_datasets_helper.py     label_subset
│   ├── test_import.py
│   ├── _rng.py                     audited round-trip helpers (data generation, bit equality)
│   ├── test_rng.py                 tests for those helpers
│   └── _datasets.py                dev-only fixtures (MNIST, etc), NOT public API
├── lab/                            ephemeral experiments; see lab/README.md
│   ├── README.md                   bench rules (tracked; the scripts themselves are not)
│   └── .gitignore                  ignores everything else (tracked)
├── docs/
│   ├── GUIDE.md                    the manual, every README claim measured with its output
│   ├── PERFORMANCE.md              what the accelerator is worth, and how to re-measure
│   ├── USAGE.md                    REPL walkthrough of every public symbol; run as a doctest
│   ├── SCOPE.md                    responsibilities matrix + parallelization analysis
│   ├── AUXILIARY.md                tests/_datasets, lab/, Z:\ conventions (NOT part of the wheel)
│   ├── THEORY.md                   distilled design + §9 condition contract; §0 binding scope
│   ├── ROADMAP.md                  milestone plan
│   ├── FOCO-1.0.md                 what 1.0 freezes and the blockers in the way
│   ├── ADR/                        one file per decision; the choice only, evidence in STUDIES/
│   ├── STUDIES/                    background reading behind those decisions
│   └── design/                     one spec per work phase: the alternatives measured and the decision
├── outreach/                       material for presenting the project publicly, not shipped
│   ├── README.md                   how it is organized and what to check before publishing
│   ├── 2026-09-03-lancamento.md    the dated news source every channel text derives from
│   └── linkedin/                   post (short) and artigo (long), each in PT and EN
└── archive/                        reference-only; gitignored (pruned 2026-05-17, only HISTORY.md kept)
```

---

## Validation lab

The library is "one image in, one tensor out" by design, but you only know it works once you run it end-to-end on real images. That happens in two places, neither of which is part of the shipped package:

- [`tests/`](tests/) holds the formal pytest suite that defines the contract from [`docs/THEORY.md`](docs/THEORY.md) §9.
- [`lab/`](lab/) holds ephemeral scripts and notebooks for fast hypothesis-checking. See [`lab/README.md`](lab/README.md) for the bench rules; outputs go to `Z:\outputs\patchcraft\` (off-tree).

Datasets used by tests/lab are downloaded lazily into `Z:\caches\datasets\<name>\` on first use; they do not ship with the package and are never bundled into the wheel.

The dev framework that makes this all work (fixtures, dataset helpers, `Z:\` conventions) is documented in [`docs/AUXILIARY.md`](docs/AUXILIARY.md).

---

## Versioning policy

The scheme is [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html),
expressed in the [PEP 440](https://peps.python.org/pep-0440/) form that PyPI
accepts. Two rules matter more than the arithmetic, because both were broken
once and both cost work to undo.

**The number names a published artifact, not a unit of work.** Finishing a
phase, merging a branch and landing a refactor are not releases, and none of
them touches `__version__`. Everything between releases accumulates under
`[Unreleased]` in [`CHANGELOG.md`](CHANGELOG.md), and the number is assigned
in the release commit itself. Versions 0.3.0 and 0.4.0 came from the opposite
habit: each internal work phase minted a minor, so three numbers existed
before any of them reached PyPI.

**A published version is frozen.** PyPI refuses a filename it has already
seen, so a version that shipped can only be superseded, never edited. That is
also why a failed pipeline never costs a number: `skip-existing` makes the
upload a no-op and the same tag can be re-run.

**While the project is pre-1.0, the middle digit is the compatibility
boundary.** SemVer itself leaves `0.y.z` open, saying only that "anything MAY
change at any time", so the meaning comes from the resolvers instead, and they
agree: Cargo expands `^0.2.3` to `>=0.2.3, <0.3.0`, npm does the same, and
PEP 440's `~=0.5.0` means `>=0.5.0, <0.6.0`. In every one of them a `0.y` bump
is the announcement that something may break, and a `0.y.z` bump is the
promise that nothing does.

That fixes what each digit is for here:

One consequence worth stating on its own, because a consumer cannot infer it
from a `Literal`: **`WeightKind` is an open set.** Adding a member is additive
and lands in a `0.y` before 1.0 and a minor after it; removing or renaming one
is a break. Code that matches on it exhaustively should keep a fallback arm.

The question that decides it is not "did a returned value change". A bug fix
changes returned values by definition, and if that forced a `0.Y.0` then a
patch release could never fix anything. The question is **whether the contract
moved, or whether the implementation stopped violating it**:

| Change | Bump | Why |
|---|---|---|
| The documented contract moves: a signature, a name, or behaviour the documentation described as correct | `0.Y.0` | Code that was right against the old documentation can now be wrong |
| The implementation stops violating the contract, however visibly | `0.y.Z` | The documentation always promised this; only the code was wrong |
| New functionality that breaks nothing | `0.y.Z` | Anyone pinned to the y-series should get it |
| Documentation, tests, CI, packaging or tooling only | `0.y.Z` | Still a release if it is published, still no break |
| Nothing published | nothing | Accumulate under `[Unreleased]` |

The project's own history sorts cleanly under that rule, and does not sort
under any rule keyed on whether values changed:

- `0.2.1` rewrote the hann window, changing `stitch(weight="hann")` output for
  **every** geometry. THEORY §9.9 had promised no covered pixel is zeroed by
  the window; the old window zeroed 108 of 144 pixels in one measured case. A
  violation, so a **z**, and it shipped as one.
- `0.5.2` floored the `stitch` denominator by the dtype's tiny instead of an
  absolute `1e-6`, changing hann output at patch 99 and above. The same
  promise, the same kind of violation, so also a **z**.
- `0.3.0` moved `stitch` output by ULPs through a different summation order.
  Nothing was wrong before, so behaviour the documentation endorsed changed.
  A **y**.
- `0.5.0` made `tilings` return 73 specs where it returned 100. The
  enumeration's output set is the contract. A **y**.

After 1.0 the ordinary reading applies: breaking is major, additive is minor,
fixed is patch. What 1.0 freezes is written in
[`docs/FOCO-1.0.md`](docs/FOCO-1.0.md).

### Errata

The rule above was written after the fact, and two releases predate it.

`0.4.0` should have been `0.3.1`. It added `accel_available()` and nothing
else: no signature changed, and the accelerated path was bit-exact against the
pure one. Additive and non-breaking is a z-bump under the boundary rule, and
minting a y announced a break that never happened.

`0.5.1` was briefly numbered `0.6.0` before release. It changes only how the
package is built and shipped, so it is a z-bump.

Two others were right and are worth recording as the shape to copy. `0.3.0`
moved `stitch` output by ULPs through a different summation order, which is an
output change and belongs at a y boundary. `0.5.0` made `tilings` return 73
specs where it returned 100, which is squarely a break.

The accelerator in [`accel/`](accel/) has no version of its own to manage. It
is compiled into this wheel, so it ships when `patchcraft` ships. The version
in `accel/Cargo.toml` is internal and nothing reads it; what the Python side
actually checks is `_ABI_VERSION` in `accel/src/lib.rs`, and a mismatch there
means a silent fall back to the pure path rather than a crash.

---

## Where the version comes from

The git tag, and nowhere else. `setuptools-scm` derives it at build time, so
no file in the source states a version and there is no literal to forget.
`v0.5.1` produces `0.5.1`; the leading `v` is stripped for you.

At build time setuptools-scm writes `src/patchcraft/_version.py`, which
`__init__.py` imports and re-exports as `patchcraft.__version__`. That file is
generated and gitignored, so a checkout that was never built falls back to
`importlib.metadata` and then to `0+unknown`. The fallback exists so an
un-built tree still imports; it is not a supported way to learn the version.

Anything other than a clean tree sitting exactly on a tag produces a version
with a local segment, like `0.5.2.dev0+gb4edfd81.d20260902` from a dirty tree
or `0.1.dev1+gb4edfd81` from a checkout whose tags were never fetched. PyPI
refuses local segments outright, so such a build can never be published by
accident, and `tools/check_dist.py` rejects it before the upload rather than
letting the upload fail.

That last point is why every `actions/checkout` that builds uses
`fetch-depth: 0`. The default shallow clone hides the tags, and the failure is
silent: a version that walks backwards rather than an error.

---

## Releasing (maintainer only)

1. Close the `[Unreleased]` section in [`CHANGELOG.md`](CHANGELOG.md) as `[X.Y.Z] YYYY-MM-DD`.
2. Update `version` and `date-released` in [`CITATION.cff`](CITATION.cff), and the `version` field of the BibTeX entry in [`docs/GUIDE.md`](docs/GUIDE.md) section 9. Those are the only version strings left in the repository, because a citation needs a concrete one. `uvx --from cffconvert cffconvert --validate` checks the file.
3. Update [`docs/ROADMAP.md`](docs/ROADMAP.md) milestone checkboxes.
4. Commit: `release: vX.Y.Z`.
5. Tag + push: `git tag -a vX.Y.Z -m "..."` then `git push origin vX.Y.Z`.
6. `release.yml` fires automatically: validates, builds, publishes to PyPI via Trusted Publishing, then creates the GitHub Release with every artifact attached.

There is no version to bump in step 4, and no way for a tag and a release to
disagree. The `validate` job still cross-checks that setuptools-scm derived
exactly what the tag says, which catches a dirty or shallow checkout before
anything is built.

One release produces one sdist and six wheels: five `cp312-abi3-<platform>`
wheels with the Rust extension inside, and one `py3-none-any` wheel for
everywhere else. `tools/check_dist.py` runs before the upload and fails the
release if a platform wheel lost its extension, if the universal wheel gained
one, if the artifacts disagree about the version, or if any of them is not the
version the tag names. Publishing is a single project, so there is nothing to
register on PyPI beyond the publisher that already exists.
