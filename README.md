<!-- l10n: doc_id=patchcraft-readme · lang=en · canonical -->
**English** · [Português](README.pt-BR.md)

# PatchCraft

[![CI status for main](https://github.com/LeoPR/PatchCraft/actions/workflows/test.yml/badge.svg)](https://github.com/LeoPR/PatchCraft/actions/workflows/test.yml)
[![PyPI version](https://img.shields.io/pypi/v/patchcraft.svg)](https://pypi.org/project/patchcraft/)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/patchcraft.svg)](https://pypi.org/project/patchcraft/)
[![License MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/LeoPR/PatchCraft/blob/main/LICENSE)

**Encode one image into patches, decode it back.** PatchCraft owns the `unfold` and `fold` arithmetic, the geometry validation and the seam blending, so that your pipeline can own everything else.

**One image at a time, on purpose.** Every call takes one `(C, H, W)` float tensor and returns one tensor, because the patch count depends on the image and a batched API would have to pad or return a list. Your `for` loop, `torch.vmap` or `DataLoader` supplies the batching.

```
   one image                     the patch stack                one image again
   (1, 4, 4)                     (4, 1, 2, 2), row-major        (1, 4, 4)

   +-----+-----+                 +-----+   +-----+              +-----+-----+
   | A A | B B |                 | A A |   | B B |              | A A | B B |
   | A A | B B |    extract      | A A |   | B B |  reconstruct | A A | B B |
   +-----+-----+   ---------->   +--p0-+   +--p1-+  ----------> +-----+-----+
   | C C | D D |   patch_size=2  +-----+   +-----+   stride=2   | C C | D D |
   | C C | D D |   stride=2      | C C |   | D D |              | C C | D D |
   +-----+-----+                 | C C |   | D D |              +-----+-----+
                                 +--p2-+   +--p3-+
```

The image goes out as a stack of patches, you do your work on the stack, and it comes back as one
image. That last arrow has two doors: `reconstruct` when the patches are untouched, and `stitch`
when a model rewrote them and the seams need to fade.

This page is the call page. The manual is [docs/GUIDE.md](docs/GUIDE.md), and it carries the measurements, the tables and the long examples that used to live here.

## Install

```
pip install patchcraft
pip install "patchcraft[cache]"     # adds zstandard, which compresses Cache payloads
```

Or with the optional native accelerator (prebuilt wheels for Windows x64,
Linux x86_64, macOS arm64 and macOS x86_64; the package stays pure-Python
and fully functional without it):

```bash
pip install patchcraft[accel]
```

`patchcraft.accel_available()` reports at runtime whether the accelerator is
active; `PATCHCRAFT_ACCEL=0` in the environment forces the pure path.

The distribution name and the import name are both `patchcraft`. The runtime dependencies are `torch>=2.6`, `numpy>=1.26` and `pillow>=10`, and the supported Python versions are in [the guide](docs/GUIDE.md#9-install-details-and-citation) together with the note you need before you install a GPU wheel.

## Sixty seconds

```python
import torch
from patchcraft import extract, reconstruct, stitch

torch.manual_seed(0)
image = torch.rand(3, 256, 256)                     # one float (C, H, W) tensor

patches = extract(image, patch_size=32, stride=32)  # (L, C, ph, pw) == (64, 3, 32, 32)

back = reconstruct(patches, image.shape, stride=32) # the inverse, for untouched patches
assert torch.equal(back, image)                     # the same tensor, bit for bit

edited = patches * 1.05 - 0.01                      # stands in for a per-patch model
blended = stitch(edited, image.shape, stride=32, weight="hann")
assert blended.shape == image.shape                 # seams smoothed, geometry preserved
```

That is the whole loop. You extract, you do your work per patch, and then you come back with either `reconstruct` or `stitch`. The dtype and the device of the input survive both directions.

## reconstruct or stitch

The two functions answer two different questions, so choosing between them is the first decision you make.

`reconstruct` is the inverse. It assumes the patches still hold the pixels `extract` gave you, it divides by the count map, and on a covering geometry it hands the image back unchanged.

`stitch` is for patches a model rewrote. Neighbouring patches now disagree about the pixels they share, and uniform averaging leaves that disagreement visible as a grid of seams, so `stitch` weights each patch by a window that fades toward its border.

| You want | Call | Because |
|---|---|---|
| The patches back as an image, untouched | `reconstruct` | It is the exact inverse of `extract` |
| A model's output back as an image | `stitch` | Overlapping patches disagree, and the window hides the grid |

## Why not unfold and fold directly

Because two defects wait there, both of them silent, and you meet them in the first hour.

The first one is the reshape. `F.unfold` returns `(1, C*ph*pw, L)`, and the intuitive reshape into `(L, C, ph, pw)` gives you the right shape with the wrong pixels.

```python
import torch
import torch.nn.functional as F
from patchcraft import extract

image = torch.arange(64, dtype=torch.float32).reshape(1, 8, 8)
patches = extract(image, patch_size=4, stride=4)             # (4, 1, 4, 4)
cols = F.unfold(image.unsqueeze(0), kernel_size=4, stride=4) # (1, C*ph*pw, L)

scrambled = cols[0].view(-1, 1, 4, 4)                        # the intuitive reshape
assert scrambled.shape == patches.shape                      # the right shape
assert not torch.equal(scrambled, patches)                   # and the wrong pixels
```

The second one is a stride that does not cover the image. On a 128 by 128 image with `patch=32, stride=20` the grid stops at pixel 112, which leaves 3840 of the 16384 pixels at zero, and a hand-rolled `fold` returns that partly black image without complaining.

Writing the tile-and-blend loop by hand costs 17 non-blank lines against 3 here, and the two results are bit-identical. [The guide](docs/GUIDE.md#1-why-not-unfold-and-fold-directly) runs both versions side by side.

## The geometry has to cover the image

`reconstruct` and `stitch` refuse a grid that does not cover the image exactly, and the error names the extent it did cover. The answer is to pick a legal geometry rather than to pad, because padding synthesizes pixels you never had.

```python
import torch
from patchcraft import extract, reconstruct, tilings

image = torch.rand(1, 128, 128)
patches = extract(image, patch_size=32, stride=20)   # a grid that stops at pixel 112
try:
    reconstruct(patches, image.shape, stride=20)
except ValueError as error:
    print(error)                                     # ... covers (112, 112) of (128, 128)

print([s.patch_size for s in tilings(image.shape)])  # 7 exact tilings, from the shape alone
```

`tilings` is arithmetic on the shape, so it reads nothing and allocates nothing. Pass `allow_overlap=True` when you want the overlapping geometries too.

## Where you are getting into

The surface is one tensor in and one tensor out. There is no batch axis, no dataset, no dataloader and no training, and that boundary is binding rather than provisional, recorded in [docs/THEORY.md](docs/THEORY.md) §0.

PatchCraft helps when you tile one image, do something per patch, and put it back. That covers inference on an image too large for a single forward pass, LR and HR patch datasets, sliding window analysis, and patch-level error maps.

PatchCraft does not help when you want a batched op over N images, a dataset or a dataloader, padding that makes an awkward geometry fit, or a model. The first three are your pipeline's job and the fourth is a network. [docs/SCOPE.md](docs/SCOPE.md) draws the line in full.

## When the round trip is bit for bit

The round trip is exact when every value in the count map is a power of two, which happens because dividing a float by a power of two is the one division that never rounds. Outside that rule float32 misses by roughly 1e-7 and float64 by roughly 1e-16, so float64 is not a safe harbour. The axis that decides the answer is the geometry and not the dtype, and [the guide](docs/GUIDE.md#4-when-the-round-trip-is-bit-for-bit) has the sweep that measures it.

## Status

**0.4.0, pre-1.0.** Output values can still change in a minor release, and [CHANGELOG.md](CHANGELOG.md) records each change with the measurement behind it.

540 tests pass and CI is green on {Ubuntu, Windows} x {Python 3.12, 3.13}, with `ruff check` and `mypy --strict` in the same run. The package is typed and ships `py.typed`.

No external project has consumed it yet, and no CUDA path in this library has ever executed. [The guide](docs/GUIDE.md#8-what-this-project-does-not-claim) lists what else this project declines to claim.

## Where to read next

| If you want | Open |
|---|---|
| The manual: every argument above, measured, with its output | [docs/GUIDE.md](docs/GUIDE.md) |
| A walkthrough of each of the 19 public symbols | [docs/USAGE.md](docs/USAGE.md) |
| The line between this library and your pipeline | [docs/SCOPE.md](docs/SCOPE.md) |
| The math, the decisions and the per-function contract | [docs/THEORY.md](docs/THEORY.md) |
| Why the API looks like this | [docs/ADR/](docs/ADR) |
| What changed in each release | [CHANGELOG.md](CHANGELOG.md) |
| To clone, test and contribute | [CONTRIBUTING.md](CONTRIBUTING.md) |

## License and citation

MIT, in [LICENSE](LICENSE). There is no DOI yet, so if you need to cite this work the BibTeX entry is in [the guide](docs/GUIDE.md#9-install-details-and-citation).
