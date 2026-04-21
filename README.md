# PatchKit

Image patch extraction, pairing and reconstruction utilities. Focused on providing a clean, tested toolkit for building **super-resolution datasets** (low-resolution ↔ high-resolution patch pairs) and for any workflow that benefits from deterministic patch-based manipulation of images.

> **Status: pre-alpha (2026-04-21).** Scaffolding only. Theory distillation in [`docs/THEORY.md`](docs/THEORY.md) precedes implementation.

## Scope

- **Extract** patches from images with configurable size, stride and dilation.
- **Pair** LR and HR patches with metadata sufficient to reconstruct either image.
- **Reconstruct** images from their patches (exact and weighted overlap modes).
- **Resize** with pluggable backends (PIL, torch).
- **Cache** processed datasets and patches on disk with content-addressed keys.
- **Subset** datasets by label with stratification.

## Scope (what PatchKit is NOT)

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
├── src/patchkit/                   library source (currently empty)
│   └── __init__.py
├── tests/                          pytest suite
├── docs/
│   ├── THEORY.md                   distilled theory (to be written from archive/)
│   └── ROADMAP.md                  milestone-based implementation plan
└── archive/                        reference-only; gitignored
    ├── PatchHub/                   earlier standalone patch library (own .git)
    └── QSVM_patchkit/              relevant subset of the QSVM legacy project
```

## Archive policy

The [`archive/`](archive/) folder contains prior implementations kept strictly as reading material to extract theory and design ideas. **Do not import code from archive at runtime.** When a pattern from the archive proves useful, reimplement cleanly in `src/patchkit/` with tests.

## Roadmap

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the ordered list of milestones.

## Author

Leonardo Marques de Souza
