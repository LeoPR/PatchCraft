# PatchKit

A small library for **encoding an image into patches and decoding it back**. Built to slot into other people's `torch` pipelines as one transform among many — like a `GaussianBlur` step in a `Compose([...])`.

> **Status (2026-05-16):** v0.1.0 ready. Public API stable: `extract`, `Patchify`, `reconstruct`, `pair`, `resize`, `Cache`, plus `num_patches`/`tilings`/`TilingSpec` (geometry helpers) and `PatchPair`/`PatchMeta`.

## The lib vs. this repo

Think of the lib as a **car** and this repo as the **car plus its test track**.

- **The car** — [`src/patchkit/`](src/patchkit/) — is what gets installed by `pip install patchkit`. It is a single library with one job: take one image (`Tensor[C, H, W]`), encode it into patches, decode patches back into the image, optionally pair LR/HR, resize, cache. **One image at a time, every time.** No datasets, no training, no orchestration, no batching across images. Multi-image is the caller's `for` loop, or `torch.vmap`, or a `DataLoader`.
- **The track** — [`tests/`](tests/), [`lab/`](lab/), [`tests/_datasets.py`](tests/_datasets.py), and the dev extras (`torchvision`, etc.) — is the pit crew, telemetry, driver and stopwatch that **prove the car works** on real images (MNIST today; more later). It downloads datasets, drives the lib through varied geometries, measures correctness. It never ships in the wheel.

The car is also **acoplável** — designed to drop into someone else's pipeline:

```python
from patchkit import Patchify
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.GaussianBlur(kernel_size=3),
    Patchify(patch_size=4, stride=2),   # ← PatchKit as one step
])
```

`Patchify` is a callable; chain it inside a `Compose`, let `DataLoader` parallelize over workers. PatchKit gives you the primitive; the surrounding pipeline stays your code.

## Scope (what the car does)

- **Extract** patches from a single image with configurable size, stride and dilation (`extract`, `Patchify`).
- **Reconstruct** an image from its patches — exact and weighted-overlap (`reconstruct`).
- **Plan** the geometry ahead of time: `num_patches((H, W), ...)` for the count, `tilings((H, W), allow_overlap=...)` for every full-coverage `(patch_size, stride)` combo (no image, no allocation — just arithmetic).
- **Pair** LR and HR patches with metadata sufficient to reconstruct either (`pair`, M4).
- **Resize** with pluggable backends — PIL or torch (`resize`, M5).
- **Cache** results on disk with content-addressed keys (`Cache`, M5).

## Scope (what the car does NOT do)

- **Not a dataset manager.** PatchKit does not load, download, batch, shuffle, or stream datasets. That's the track's job — `tests/_datasets.py` has `mnist_subset(...)` for dev fixtures, and `torchvision` is in the `[dev]` extra (never a runtime dep of the car).
- **Not a multi-image API.** Every primitive takes one image. Use `vmap` or a Python loop if you need to apply it to many.
- No SVMs, no kernels, no quantum circuits — those belong to other projects.
- No neural network training — PatchKit is infrastructure, not a model.

## Install

### From PyPI

```
pip install patchkit            # core only
pip install patchkit[cache]     # adds zstandard for compressed Cache entries
```

### From source (development)

```
git clone https://github.com/LeoPR/patchkit.git
cd patchkit
pip install -e ".[dev,cache]"
```

For GPU support, install a matching torch wheel before PatchKit
(e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu124`).

## Run tests

```
pytest
pytest -m "not gpu"        # skip GPU-requiring tests
```

## Layout

```
PatchKit/
├── pyproject.toml                  package metadata, build backend (hatchling)
├── README.md                       this file
├── LICENSE                         MIT
├── .python-version                 3.13
├── .gitignore                      ignores archive/, venvs, caches, outputs
├── src/patchkit/                   library core — one-image-at-a-time primitives
│   ├── __init__.py                 re-exports the full public API
│   ├── extract.py                  patches via F.unfold; Patchify wrapper (ADR 0002)
│   ├── reconstruct.py              inverse via F.fold + count map
│   ├── geometry.py                 pre-flight: num_patches, tilings, TilingSpec
│   ├── pair.py                     LR↔HR pairing; PatchPair, PatchMeta
│   ├── resize.py                   resize with PIL or torch backends
│   └── cache.py                    content-addressed disk cache
├── tests/                          pytest suite (contract tests for src/)
│   ├── test_extract.py             extract + Patchify
│   ├── test_reconstruct.py
│   ├── test_geometry.py            num_patches + tilings
│   ├── test_pair.py
│   ├── test_resize.py
│   ├── test_cache.py
│   ├── test_datasets_helper.py     label_subset
│   ├── test_import.py
│   └── _datasets.py                dev-only fixtures (MNIST, etc) — NOT public API
├── lab/                            ephemeral experiments; see lab/README.md
│   ├── README.md                   bench rules (tracked)
│   └── .gitignore                  ignores everything else (tracked)
├── docs/
│   ├── THEORY.md                   distilled design + §9 condition contract; §0 binding scope
│   ├── ROADMAP.md                  milestone plan
│   └── ADR/
│       ├── 0001-patch-extraction-api.md   pure function `extract`
│       └── 0002-patchify-transform.md     callable wrapper for Compose pipelines
└── archive/                        reference-only; gitignored
    ├── PatchHub/                   earlier standalone patch library (own .git)
    └── QSVM_patchkit/              relevant subset of the QSVM legacy project
```

## Validation lab

The library is "one image in, one tensor out" by design — but you only know it works once you run it end-to-end on real images. That happens in two places, neither of which is part of the shipped package:

- [`tests/`](tests/) — formal pytest suite that defines the contract from [`docs/THEORY.md`](docs/THEORY.md) §9.
- [`lab/`](lab/) — ephemeral scripts and notebooks for fast hypothesis-checking. See [`lab/README.md`](lab/README.md) for the bench rules; outputs go to `Z:\outputs\patchkit\` (off-tree).

Datasets used by tests/lab are downloaded lazily into `Z:\caches\datasets\<name>\` on first use; they do not ship with the package and are never bundled into the wheel.

## Archive policy

The [`archive/`](archive/) folder contains prior implementations kept strictly as reading material to extract theory and design ideas. **Do not import code from archive at runtime.** When a pattern from the archive proves useful, reimplement cleanly in `src/patchkit/` with tests.

## Roadmap

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the ordered list of milestones.

## Author

Leonardo Marques de Souza
