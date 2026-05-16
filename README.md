# PatchKit

Image patch extraction, pairing and reconstruction utilities. Focused on providing a clean, tested toolkit for building **super-resolution datasets** (low-resolution ↔ high-resolution patch pairs) and for any workflow that benefits from deterministic patch-based manipulation of images.

> **Status (2026-05-15):** M0 scaffold, M1 theory + ADR 0001, M2 `extract` shipped. Working on the validation lab before M3 (`reconstruct`).

## Scope

PatchKit operates on **one image at a time**. The core takes a `(C, H, W)` tensor and produces patches, reconstructed images, paired LR/HR tensors, resized variants, or a cache hit. Dataset orchestration, batching across many images, and training loops live outside this library.

- **Extract** patches from a single image with configurable size, stride and dilation.
- **Pair** LR and HR patches with metadata sufficient to reconstruct either image.
- **Reconstruct** an image from its patches (exact and weighted overlap modes).
- **Resize** with pluggable backends (PIL, torch).
- **Cache** results on disk with content-addressed keys.

## Scope (what PatchKit is NOT)

- **Not a dataset manager.** PatchKit does not load, download, batch, shuffle, or stream datasets. Test fixtures that exercise the core against real images (MNIST, etc.) live in [`tests/_datasets.py`](tests/_datasets.py) and are dev-only — `torchvision` is in the `[dev]` extra, not in the runtime deps.
- No SVMs, no kernels, no quantum circuits — those belong to other projects.
- No neural network training — PatchKit is infrastructure, not a model.

## Install (development, from source)

```powershell
# Create venv on Z: (outside OneDrive)
py -V:3.13 -m venv Z:\venvs\patchkit
Z:\venvs\patchkit\Scripts\Activate.ps1

# Install uv (fast package manager)
pip install uv

# Install torch with CUDA wheels (RTX 3060 + driver 596.21 → cu124)
uv pip install torch --extra-index-url https://download.pytorch.org/whl/cu124

# Install PatchKit editable + dev extras
uv pip install -e ".[dev,cache]"
```

## Install (once published)

```powershell
pip install patchkit
```

## Run tests

```powershell
pytest
pytest -m "not slow"       # skip slow tests
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
│   ├── __init__.py
│   └── extract.py                  M2: patches via F.unfold
├── tests/                          pytest suite (contract tests for src/)
│   ├── test_extract.py
│   └── _datasets.py                dev-only fixtures (MNIST, etc) — NOT public API
├── lab/                            ephemeral experiments; see lab/README.md
│   ├── README.md                   bench rules (tracked)
│   └── .gitignore                  ignores everything else (tracked)
├── docs/
│   ├── THEORY.md                   distilled design + §10 condition contract
│   ├── ROADMAP.md                  milestone plan
│   └── ADR/                        architecture decision records
└── archive/                        reference-only; gitignored
    ├── PatchHub/                   earlier standalone patch library (own .git)
    └── QSVM_patchkit/              relevant subset of the QSVM legacy project
```

## Validation lab

The library is "one image in, one tensor out" by design — but you only know it works once you run it end-to-end on real images. That happens in two places, neither of which is part of the shipped package:

- [`tests/`](tests/) — formal pytest suite that defines the contract from [`docs/THEORY.md`](docs/THEORY.md) §10.
- [`lab/`](lab/) — ephemeral scripts and notebooks for fast hypothesis-checking. See [`lab/README.md`](lab/README.md) for the bench rules; outputs go to `Z:\outputs\patchkit\` (off-tree).

Datasets used by tests/lab are downloaded lazily into `Z:\caches\datasets\<name>\` on first use; they do not ship with the package and are never bundled into the wheel.

## Archive policy

The [`archive/`](archive/) folder contains prior implementations kept strictly as reading material to extract theory and design ideas. **Do not import code from archive at runtime.** When a pattern from the archive proves useful, reimplement cleanly in `src/patchkit/` with tests.

## Roadmap

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the ordered list of milestones.

## Author

Leonardo Marques de Souza
