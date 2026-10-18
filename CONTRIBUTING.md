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
it, and adopting it would reformat 37 files in a commit that changed no
behaviour.

```
pytest
pytest -m "not gpu"        # skip GPU-requiring tests
```

CI runs the full suite on every push to `main` and on every pull request against it. See [`.github/workflows/test.yml`](.github/workflows/test.yml). Matrix is `{ubuntu-latest, windows-latest} × {python 3.12, 3.13, 3.14}` for the pure path, plus a job on both operating systems for the accelerated one and a job at the declared floor, which runs both paths. While GitHub moves `ubuntu-latest` from Ubuntu 24.04 to 26.04, until 2026-11-19, one pure cell, the accelerated job and the floor job also run on `ubuntu-26.04` by name. [`release.yml`](.github/workflows/release.yml) builds on `ubuntu-24.04`, and moving it is a separate change, which the comment at its top describes. To run the same checks locally before pushing:

```
ruff check src tests tools
mypy --strict src
pytest -m "not gpu"
```

Every action in the workflows is pinned to an exact version, the PyPI upload to a commit, and [`.github/dependabot.yml`](.github/dependabot.yml) proposes updates on the first of each month, never for a release younger than 14 days. Its pull requests come in two groups: `tested-by-ci`, the actions that the pull request's own CI runs, and `release-only`, the actions that only `release.yml` runs. `dtolnay/rust-toolchain` stays on its branch refs, which choose the toolchain, and Dependabot does not track it.

Every one of them that changes `release.yml` also gets a dry run of it (see [Releasing](#releasing-maintainer-only)), which builds every artifact and applies the gates; only the PyPI upload and the GitHub Release step meet a new version first at the next tag. A `setup-uv` bump also moves uv, and the dry run builds the sdist and the universal wheel with it. A new major arrives as a pull request of its own: adopt it for a real gain, or close it with `@dependabot ignore this major version`, which records the decision. After merging, fix any comment that names the old version, since Dependabot rewrites only the `uses:` line.

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
├── tools/compare_exactness_rules.py  the power-of-two rule against k_max <= 4, measured
├── tools/compare_border_strategies.py  truncate, pad or move the last window, measured
├── tools/make_guide_figures.py     every GUIDE image and the numbers under it; --check runs in the suite
├── tools/make_outreach_figures.py  every outreach figure and generated page, computed
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
│   ├── ISSUE_TEMPLATE/             four forms and a chooser; the required fields are what makes a report reproducible
│   ├── PULL_REQUEST_TEMPLATE.md    asks whether a returned value moved, which decides the version bump
│   ├── dependabot.yml              monthly update proposals for the pinned actions, none under 14 days old
│   └── workflows/
│       ├── test.yml                matrix CI on PRs/main, plus the accelerated and floor jobs
│       └── release.yml             publishes to PyPI on vX.Y.Z tag push (Trusted Publishing); without a tag, a dry run
├── src/patchcraft/                 library core, one-image-at-a-time primitives
│   ├── __init__.py                 re-exports the full public API and the generated version
│   ├── extract.py                  patches via a strided window view and one copy; Patchify (ADR 0002)
│   ├── reconstruct.py              inverse via the fold and a closed-form count map; exact integer mean
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
│   ├── test_integer.py             integer images through extract and reconstruct, the exact mean
│   ├── test_vmap.py                torch.vmap over the primitives, pinned to a plain loop
│   ├── test_forward_ad.py          forward-mode AD through the fold, both paths
│   ├── test_accel_eligibility.py   which tensors the native kernel may take, and which it must decline
│   ├── test_exactness.py           falsifies the count-map predicate over the legal space
│   ├── test_operator.py            extract as a matrix U: the fold is U^T, reconstruct and stitch are least squares
│   ├── test_reference.py           naive loop-based reference for the fast paths
│   ├── test_public_api.py          freezes the 20 names, their signatures and carrier fields
│   ├── test_accel.py               native path, skipped unless the extension was built
│   ├── test_datasets_helper.py     label_subset
│   ├── test_import.py
│   ├── test_docs_usage.py          runs docs/USAGE.md as a doctest
│   ├── test_docs_figures.py        runs tools/make_guide_figures.py --check
│   ├── test_citation.py            CITATION.cff agrees with the package
│   ├── conftest.py                 shared fixtures
│   ├── _rng.py                     audited round-trip helpers (data generation, bit equality)
│   ├── test_rng.py                 tests for those helpers
│   └── _datasets.py                dev-only fixtures (MNIST, etc), NOT public API
├── lab/                            ephemeral experiments; see lab/README.md
│   ├── README.md                   bench rules (tracked; the scripts themselves are not)
│   └── .gitignore                  ignores everything else (tracked)
├── docs/
│   ├── GUIDE.md                    the manual, every README claim measured with its output
│   ├── figures/                    the GUIDE's images, written by tools/make_guide_figures.py
│   ├── PERFORMANCE.md              what the accelerator is worth, and how to re-measure
│   ├── USAGE.md                    REPL walkthrough of every public symbol; run as a doctest
│   ├── SCOPE.md                    responsibilities matrix + parallelization analysis
│   ├── AUXILIARY.md                tests/_datasets, lab/, Z:\ conventions (NOT part of the wheel)
│   ├── THEORY.md                   distilled design + §9 condition contract; §0 binding scope
│   ├── ROADMAP.md                  milestone plan
│   ├── FOCO-1.0.md                 what 1.0 freezes, and the six blockers, all closed
│   ├── ADR/                        one file per decision; the choice only, evidence in STUDIES/
│   ├── STUDIES/                    background reading behind those decisions
│   ├── design/                     one spec per work phase: the alternatives measured and the decision
│   └── PAPER/                      an article in preparation, with its experiments; not shipped
├── outreach/                       material for presenting the project publicly, not shipped
│   ├── README.md                   how it is organized and what to check before publishing
│   ├── 2026-10-10-lancamento.md    the current dated news source the channel texts derive from
│   ├── 2026-09-0*-lancamento.md    the earlier ones, kept as the records of 0.5.4 and 0.5.1
│   └── linkedin/                   post, artigo, the illustrated page and the figures, PT and EN
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
seen, so a version that shipped can only be superseded, never edited. A failed
pipeline still costs no number if it is finished in time: until a file reaches
PyPI the tag can move to a fix, and after that, re-running the failed jobs
uploads the missing files while `skip-existing` passes over the ones already
there, but PyPI takes new files only within 14 days of the version's first
upload. [When a release run fails](#when-a-release-run-fails) has the steps.

**While the project is pre-1.0, the middle digit is the compatibility
boundary.** SemVer itself leaves `0.y.z` open, saying only that "anything MAY
change at any time", so the meaning comes from the resolvers instead, and they
agree: Cargo expands `^0.2.3` to `>=0.2.3, <0.3.0`, npm does the same, and
PEP 440's `~=0.5.0` means `>=0.5.0, <0.6.0`. In every one of them a `0.y` bump
is the announcement that something may break, and a `0.y.z` bump is the
promise that nothing does.

That fixes what each digit is for here.

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

A new keyword-only parameter with a default breaks no call, but it moves a
signature, so it takes the first row and not the third: 0.6.0 is the example,
where `resize` gained `antialias`. The third row is for new behaviour behind
an unchanged signature, such as 0.5.7 accepting integer images.

The project's own history sorts cleanly under that rule, and does not sort
under any rule keyed on whether values changed:

- `0.2.1` rewrote the hann window, changing `stitch(weight="hann")` output for
  **every** geometry. THEORY §9.9 had promised no covered pixel is zeroed by
  the window; the old window zeroed 108 of 144 pixels in one measured case. A
  violation, so a **z**, and it shipped as one.
- `0.5.2` floored the `stitch` denominator by the dtype's tiny instead of an
  absolute `1e-6`, changing hann output at patch 99 and above. The same
  promise, the same kind of violation, so also a **z**.
- `0.3.0` moved `stitch` output through a different summation order.
  Nothing was wrong before, so behaviour the documentation endorsed changed.
  A **y**. Its notes said the move was by ULPs; measured on 0.6.0, the new
  window denominator cancelled on the last `patch - stride` rows and columns,
  by thousands of eps and at large patches to zero. That was a defect of its
  own, and 0.6.1 fixes it as a **z**, under the promise 0.5.2 kept.
- `0.5.0` made `tilings` return 73 specs where it returned 100. The
  enumeration's output set is the contract. A **y**.

One consequence worth stating on its own, because a consumer cannot infer it
from a `Literal`: **`WeightKind` is an open set.** Adding a member is additive
and lands in a `0.y` before 1.0 and a minor after it; removing or renaming one
is a break. Code that matches on it exhaustively should keep a fallback arm.

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
2. Update `version` and `date-released` in [`CITATION.cff`](CITATION.cff), the `version` field of the BibTeX entry in [`docs/GUIDE.md`](docs/GUIDE.md) section 9, and the release named at the top of [`STATUS.md`](STATUS.md). Those are the version strings a release bumps, because a citation needs a concrete one and a snapshot names its release; [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) names the version it measured and changes only when the table is re-measured. `uvx --from cffconvert cffconvert --validate` checks the file.
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

**A dry run is `release.yml` without a tag.** Start one from Actions, release,
Run workflow, on any branch: it runs the tests, builds the sdist and all six
wheels, and applies every gate, then stops, because `publish-pypi` and
`github-release` run only on a tag. It also runs by itself on a push to `main`
or a pull request that changes a packaging file: `release.yml`,
`pyproject.toml`, `setup.py`, `MANIFEST.in`, `tools/check_dist.py` or the
Cargo files. The artifacts carry setuptools-scm's development version, which
the dry run's gates accept and a release's refuse. Its `linux_runner` choice
puts the Linux x86_64 jobs on `ubuntu-26.04`, to try that image before the
release moves to it.

### When a release run fails

Look first at the version's files on PyPI, because what to do depends on
whether any of them got there.

**None reached PyPI.** The number is not spent and no PyPI clock is running,
because PyPI creates a release only with its first file. When nothing has to
change, use **Re-run failed jobs**, which GitHub allows for 30 days; a re-run
uses the original commit and `release.yml`, so it cannot pick up a fix. When
something does have to change, cancel the old run if it is still going, commit
the fix, check it with a dry run, delete the tag (`git push origin --delete vX.Y.Z`, then
`git tag -d vX.Y.Z`), and tag and push the fixed commit as in step 5. v0.5.1
went through this three times.

**Some files reached PyPI.** Do not move the tag. PyPI keeps those files, and
`skip-existing` passes over a rebuilt file of the same name with only a
warning, so the version would mix two builds while CI stays green. Use **Re-run
failed jobs**, not Re-run all jobs, so the missing files go up from the same
artifacts, and do it within 14 days (336 hours) of the version's first upload:
the earliest `upload_time` on its PyPI page, which deleting a file does not
reset. Since July 2026 PyPI answers a new file after that with HTTP 400,
"Uploading new files to releases older than 14 days is not allowed.", which
`skip-existing` does not skip. The version can then only be superseded by a
new one, which is also the way out when the cause needs a code change.

**Only `github-release` failed.** PyPI has all seven files, so its window no
longer matters. Re-run that job within GitHub's 30 days; after that, or when
the job itself needs a fix, create the GitHub Release by hand from the seven
files on PyPI.
