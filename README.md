# PatchCraft

A small library for **encoding an image into patches and decoding it back**. Built to slot into other people's `torch` pipelines as one transform among many — like a `GaussianBlur` step in a `Compose([...])`.

> **Status (2026-05-17):** v0.1.0 released; v0.2.0-track is on `main` (not yet tagged). Public API: `extract`, `Patchify`, `reconstruct`, `stitch`, `pair`, `resize`, `Cache`, plus geometry helpers (`num_patches`, `tilings`, `TilingSpec`, `scale_factor`, `paired_tilings`, `PairedTilingSpec`), pixel metrics (`patch_metrics`, `per_patch_mse`, `per_patch_psnr`), and `PatchPair`/`PatchMeta`.

## The lib vs. this repo

Think of the lib as a **car** and this repo as the **car plus its test track**.

- **The car** — the [`patchcraft`](https://github.com/LeoPR/PatchCraft/tree/main/src/patchcraft) package — is what gets installed by `pip install patchcraft`. It is a single library with one job: take one image (`Tensor[C, H, W]`), encode it into patches, decode patches back into the image, optionally pair LR/HR, resize, cache. **One image at a time, every time.** No datasets, no training, no orchestration, no batching across images. Multi-image is the caller's `for` loop, or `torch.vmap`, or a `DataLoader`.
- **The track** — `tests/`, `lab/`, `tests/_datasets.py`, and the dev extras (`torchvision`, etc.) in the repo — is the pit crew, telemetry, driver and stopwatch that **prove the car works** on real images. It downloads datasets, drives the lib through varied geometries, measures correctness. It never ships in the wheel. See [CONTRIBUTING.md](https://github.com/LeoPR/PatchCraft/blob/main/CONTRIBUTING.md) if you're contributing.

The car is also **acoplável** — designed to drop into someone else's pipeline:

```python
from patchcraft import Patchify
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.GaussianBlur(kernel_size=3),
    Patchify(patch_size=4, stride=2),   # ← PatchCraft as one step
])
```

`Patchify` is a callable; chain it inside a `Compose`, let `DataLoader` parallelize over workers. PatchCraft gives you the primitive; the surrounding pipeline stays your code.

## Visual cheat sheet

The five core operations, one diagram each. Letters mark which patch each cell came from / goes to.

### `extract` — image → patch stack

`patch_size=4`, `stride=4` (no overlap) on an 8×8 image:

```
   image (1, 8, 8)                  patches (4, 1, 4, 4)
   +-----------------+              +-----+  +-----+
   | . . . . | . . . . |             |  A  |  |  B  |
   | . A . . | . B . . |  extract    +-----+  +-----+
   | . . . . | . . . . |  -------->   patch0   patch1
   | . . . . | . . . . |
   |---------+---------|             +-----+  +-----+
   | . . . . | . . . . |             |  C  |  |  D  |
   | . C . . | . D . . |             +-----+  +-----+
   | . . . . | . . . . |              patch2   patch3
   | . . . . | . . . . |             (row-major order)
   +-----------------+
```

### `reconstruct` — patch stack → image (bit-exact when `stride == patch_size`)

Each output pixel = sum of patch contributions / `count` map (= how many patches covered it). When `stride == patch_size`, `count` is all-ones and the divide is a no-op.

```
   stride == patch  -->  count map all 1   -->  trivial copy
   stride <  patch  -->  count map > 1     -->  weighted average

   patch=4, stride=2, image cols 0..7:
     col:    0  1  2  3  4  5  6  7
     patch0: x  x  x  x
     patch1:       x  x  x  x
     patch2:             x  x  x  x
     count:  1  1  2  2  2  2  1  1   <- divide sum by this
```

### `pair` — LR <-> HR, same image region, different resolution

`scale_factor=2`: every k-th LR patch corresponds to the k-th HR patch; HR coords are LR coords times the integer scale.

```
   LR (1, 4, 4)               HR (1, 8, 8)
   +---------+                +-------------+
   | . . . . |                | . . . . . . . . |
   | .[A]. . |   k = 1  -->   | . .[A A]. . . . |
   | . . . . |                | . .[A A]. . . . |
   | . . . . |                | . . . . . . . . |
   +---------+                | . . . . . . . . |
                              | . . . . . . . . |
                              | . . . . . . . . |
                              | . . . . . . . . |
                              +-------------+

   LR patch at (row=1, col=1)  <-->  HR patch at (row=2, col=2)
```

### `stitch` — same fold geometry as `reconstruct`, but each patch weighted by a window kernel

Use when patches were modified by a model and uniform averaging shows boundary seams. Window kernels for `patch_size=4`:

```
   weight="uniform"     weight="hann"        weight="gaussian"
   (== reconstruct)     centers > edges      centers >> edges (never 0)

   + + + +              . . . .              . o o .
   + + + +              . X X .              o X X o
   + + + +              . X X .              o X X o
   + + + +              . . . .              . o o .

   no seam attenuation  strong attenuation,  smooth attenuation,
                        image corners -> 0   corners preserved
```

### Everything stays one-image-at-a-time

```
   for image in images:
       patches = extract(image, ...)       # PatchCraft primitive
       result  = model(patches)            # caller's work
       out     = stitch(result, ...)       # PatchCraft primitive
```

Multi-image parallelism is the caller's pipeline (`torch.vmap`, `DataLoader` workers, etc.) — see [SCOPE.md](https://github.com/LeoPR/PatchCraft/blob/main/docs/SCOPE.md) §2.

## Scope (what the car does)

- **Extract** patches from a single image with configurable size, stride and dilation (`extract`, `Patchify`).
- **Reconstruct** an image from its patches — exact and weighted-overlap (`reconstruct`).
- **Stitch** *modified* patches (model output, denoised, super-resolved) back into one image with a window kernel that attenuates boundary seams (`stitch`, with `weight="uniform"|"hann"|"gaussian"`).
- **Plan** the geometry ahead of time: `num_patches((H, W), ...)` for the count, `tilings((H, W), allow_overlap=...)` for every full-coverage `(patch_size, stride)` combo (no image, no allocation — just arithmetic). For LR↔HR setups: `scale_factor(...)` and `paired_tilings(...)`.
- **Pair** LR and HR patches with metadata sufficient to reconstruct either (`pair`, `PatchPair`, `PatchMeta`).
- **Measure** pixel-level error between two patch stacks: `patch_metrics`, `per_patch_mse`, `per_patch_psnr`.
- **Resize** with pluggable backends — PIL or torch (`resize`).
- **Cache** results on disk with content-addressed keys, OneDrive-race retry, optional zstd (`Cache`).

## Scope (what the car does NOT do)

- **Not a dataset manager.** PatchCraft does not load, download, batch, shuffle, or stream datasets. That's the track's job — `tests/_datasets.py` has `mnist_subset(...)` for dev fixtures, and `torchvision` is in the `[dev]` extra (never a runtime dep of the car).
- **Not a multi-image API.** Every primitive takes one image. Use `vmap` or a Python loop if you need to apply it to many.
- No SVMs, no kernels, no quantum circuits — those belong to other projects.
- No neural network training — PatchCraft is infrastructure, not a model.

## Install

### From PyPI

```
pip install patchcraft            # core only
pip install patchcraft[cache]     # adds zstandard for compressed Cache entries
```

### From source (development)

```
git clone https://github.com/LeoPR/PatchCraft.git
cd patchcraft
pip install -e ".[dev,cache]"
```

For GPU support, install a matching torch wheel before PatchCraft
(e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu124`).

## Where to read next

| If you want… | Open |
|---|---|
| A hands-on tour with real REPL outputs for every public API | [USAGE.md](https://github.com/LeoPR/PatchCraft/blob/main/docs/USAGE.md) |
| The line between "PatchCraft's job" and "your pipeline's job", plus the parallelization story | [SCOPE.md](https://github.com/LeoPR/PatchCraft/blob/main/docs/SCOPE.md) |
| Design decisions, math, the per-API contract | [THEORY.md](https://github.com/LeoPR/PatchCraft/blob/main/docs/THEORY.md) |
| Architecture Decision Records | [ADR/](https://github.com/LeoPR/PatchCraft/tree/main/docs/ADR) |
| Per-release changes | [CHANGELOG.md](https://github.com/LeoPR/PatchCraft/blob/main/CHANGELOG.md) |
| Cloning and contributing (run tests, layout, validation conventions) | [CONTRIBUTING.md](https://github.com/LeoPR/PatchCraft/blob/main/CONTRIBUTING.md) |

## Author

Leonardo Marques de Souza
