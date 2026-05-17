# Contributing to PatchCraft

If you've cloned this repo and want to run the test suite, understand how the project is laid out, or follow the validation conventions, this is the page. For installation and usage, see the [README](README.md).

---

## Run tests

```
pytest
pytest -m "not gpu"        # skip GPU-requiring tests
```

CI runs the full suite on every push and PR — see [`.github/workflows/test.yml`](.github/workflows/test.yml). Matrix is `{ubuntu-latest, windows-latest} × {python 3.12, 3.13}`. To run the same checks locally before pushing:

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
├── README.md                       public-facing (also goes to PyPI as long_description)
├── CONTRIBUTING.md                 this file
├── CHANGELOG.md                    Keep-a-Changelog format
├── LICENSE                         MIT
├── .python-version                 3.13
├── .gitignore                      ignores archive/, venvs, caches, outputs
├── .github/workflows/
│   ├── test.yml                    matrix CI on PRs/main
│   └── release.yml                 publishes to PyPI on vX.Y.Z tag push (Trusted Publishing)
├── src/patchcraft/                 library core — one-image-at-a-time primitives
│   ├── __init__.py                 re-exports the full public API
│   ├── extract.py                  patches via F.unfold; Patchify wrapper (ADR 0002)
│   ├── reconstruct.py              inverse via F.fold + count map
│   ├── stitch.py                   weighted reassembly for modified patches
│   ├── geometry.py                 pre-flight: num_patches, tilings, scale_factor, paired_tilings
│   ├── metrics.py                  patch_metrics, per_patch_mse, per_patch_psnr
│   ├── pair.py                     LR↔HR pairing; PatchPair, PatchMeta
│   ├── resize.py                   resize with PIL or torch backends
│   └── cache.py                    content-addressed disk cache
├── tests/                          pytest suite (contract tests for src/)
│   ├── test_extract.py             extract + Patchify
│   ├── test_reconstruct.py
│   ├── test_stitch.py
│   ├── test_geometry.py            num_patches + tilings + scale_factor + paired_tilings
│   ├── test_metrics.py
│   ├── test_pair.py
│   ├── test_resize.py
│   ├── test_cache.py
│   ├── test_datasets_helper.py     label_subset
│   ├── test_import.py
│   └── _datasets.py                dev-only fixtures (MNIST, etc) — NOT public API
├── lab/                            ephemeral experiments; see lab/README.md
│   ├── README.md                   bench rules (tracked)
│   ├── usage_demo.py               regenerates the live REPL outputs in docs/USAGE.md
│   └── .gitignore                  ignores everything else (tracked)
├── docs/
│   ├── USAGE.md                    live REPL walkthrough of every public API
│   ├── SCOPE.md                    responsibilities matrix + parallelization analysis
│   ├── AUXILIARY.md                tests/_datasets, lab/, Z:\ conventions (NOT part of the wheel)
│   ├── THEORY.md                   distilled design + §9 condition contract; §0 binding scope
│   ├── ROADMAP.md                  milestone plan
│   └── ADR/
│       ├── 0001-patch-extraction-api.md   pure function `extract`
│       └── 0002-patchify-transform.md     callable wrapper for Compose pipelines
└── archive/                        reference-only; gitignored (pruned 2026-05-17 — only HISTORY.md kept)
```

---

## Validation lab

The library is "one image in, one tensor out" by design — but you only know it works once you run it end-to-end on real images. That happens in two places, neither of which is part of the shipped package:

- [`tests/`](tests/) — formal pytest suite that defines the contract from [`docs/THEORY.md`](docs/THEORY.md) §9.
- [`lab/`](lab/) — ephemeral scripts and notebooks for fast hypothesis-checking. See [`lab/README.md`](lab/README.md) for the bench rules; outputs go to `Z:\outputs\patchcraft\` (off-tree).

Datasets used by tests/lab are downloaded lazily into `Z:\caches\datasets\<name>\` on first use; they do not ship with the package and are never bundled into the wheel.

The dev framework that makes this all work (fixtures, dataset helpers, `Z:\` conventions) is documented in [`docs/AUXILIARY.md`](docs/AUXILIARY.md).

---

## Releasing (maintainer only)

1. Bump `__version__` in [`src/patchcraft/__init__.py`](src/patchcraft/__init__.py).
2. Close the `[Unreleased]` section in [`CHANGELOG.md`](CHANGELOG.md) as `[X.Y.Z] — YYYY-MM-DD`.
3. Update [`docs/ROADMAP.md`](docs/ROADMAP.md) milestone checkboxes.
4. Commit: `release: vX.Y.Z`.
5. Tag + push: `git tag -a vX.Y.Z -m "..."` then `git push origin vX.Y.Z`.
6. `release.yml` fires automatically: validates → builds → publishes to PyPI via Trusted Publishing → creates GitHub Release with `.whl` + `.tar.gz`.
