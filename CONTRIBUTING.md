# Contributing to PatchCraft

If you've cloned this repo and want to run the test suite, understand how the project is laid out, or follow the validation conventions, this is the page. For installation and usage, see the [README](README.md).

---

## Run tests

```
pytest
pytest -m "not gpu"        # skip GPU-requiring tests
```

CI runs the full suite on every push and PR. See [`.github/workflows/test.yml`](.github/workflows/test.yml). Matrix is `{ubuntu-latest, windows-latest} × {python 3.12, 3.13, 3.14}`. To run the same checks locally before pushing:

```
ruff check src tests
mypy --strict src
pytest -m "not gpu"
```

---

## Layout

```
PatchCraft/
├── pyproject.toml                  package metadata, build backend (hatchling)
├── README.md                       the call page, canonical, English
├── README.pt-BR.md                 the same call page in Portuguese
├── README.pypi.md                  the PyPI page (long_description), links absolute
├── CONTRIBUTING.md                 this file
├── CHANGELOG.md                    Keep-a-Changelog format
├── LICENSE                         MIT
├── .python-version                 3.13
├── .gitignore                      ignores archive/, venvs, caches, outputs
├── .github/workflows/
│   ├── test.yml                    matrix CI on PRs/main, plus the accelerated job
│   └── release.yml                 publishes to PyPI on vX.Y.Z tag push (Trusted Publishing)
├── src/patchcraft/                 library core, one-image-at-a-time primitives
│   ├── __init__.py                 re-exports the full public API, holds __version__
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
├── accel/                          patchcraft-accel, a separate distribution (Rust, pyo3/maturin)
│   ├── Cargo.toml                  crate metadata, holds the accel version
│   ├── src/kernel.rs               the gather-fold kernel, pure Rust, cargo-tested
│   ├── src/lib.rs                  pyo3 glue
│   └── README.md                   how to build it from source
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
│   ├── test_accel.py               native path, skipped unless the accelerator is installed
│   ├── test_datasets_helper.py     label_subset
│   ├── test_import.py
│   ├── _rng.py                     audited round-trip helpers (data generation, bit equality)
│   ├── test_rng.py                 tests for those helpers
│   └── _datasets.py                dev-only fixtures (MNIST, etc), NOT public API
├── lab/                            ephemeral experiments; see lab/README.md
│   ├── README.md                   bench rules (tracked)
│   ├── usage_demo.py               regenerates the live REPL outputs in docs/USAGE.md
│   └── .gitignore                  ignores everything else (tracked)
├── docs/
│   ├── GUIDE.md                    the manual, every README claim measured with its output
│   ├── USAGE.md                    REPL walkthrough of every public symbol (behind, see B4)
│   ├── SCOPE.md                    responsibilities matrix + parallelization analysis
│   ├── AUXILIARY.md                tests/_datasets, lab/, Z:\ conventions (NOT part of the wheel)
│   ├── THEORY.md                   distilled design + §9 condition contract; §0 binding scope
│   ├── ROADMAP.md                  milestone plan
│   ├── FOCO-1.0.md                 what 1.0 freezes and the blockers in the way
│   ├── ADR/                        one file per decision that shaped the API
│   ├── STUDIES/                    background reading behind those decisions
│   └── superpowers/                the design specs and implementation plans per work phase
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

Which component moves, while the project is pre-1.0 and SemVer leaves `0.y.z`
open:

| Change | Bump |
|---|---|
| Any output value, signature, name or documented behaviour changes | minor, `0.Y.0` |
| A fix that changes no documented behaviour and no signature | patch, `0.y.Z` |
| Documentation, tests, CI or tooling only | nothing |

After 1.0 the ordinary reading applies: breaking is major, additive is minor,
fixed is patch. What 1.0 freezes is written in
[`docs/FOCO-1.0.md`](docs/FOCO-1.0.md).

The accelerator in [`accel/`](accel/) is a separate distribution with its own
version in `accel/Cargo.toml`. It moves on its own schedule, and the two
numbers are not meant to track each other.

---

## Releasing (maintainer only)

1. Bump `__version__` in [`src/patchcraft/__init__.py`](src/patchcraft/__init__.py).
2. Close the `[Unreleased]` section in [`CHANGELOG.md`](CHANGELOG.md) as `[X.Y.Z] YYYY-MM-DD`.
3. Sweep the version strings the docs quote, which live in
   [`docs/GUIDE.md`](docs/GUIDE.md) (the provenance note, the section 7
   assertion, section 8 and the BibTeX entry) and in the Status block of the
   three cover pages.
4. Update [`docs/ROADMAP.md`](docs/ROADMAP.md) milestone checkboxes.
5. Commit: `release: vX.Y.Z`.
6. Tag + push: `git tag -a vX.Y.Z -m "..."` then `git push origin vX.Y.Z`.
7. `release.yml` fires automatically: validates → builds → publishes to PyPI via Trusted Publishing → creates GitHub Release with `.whl` + `.tar.gz`.

The tag must match `__version__`, and the `validate` job fails the run if it
does not.
