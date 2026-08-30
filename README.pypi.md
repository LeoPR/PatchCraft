# PatchCraft

Encode one image into patches, decode it back, and decide what happens at the seams.

[![Latest version on PyPI](https://img.shields.io/pypi/v/patchcraft?color=3775A9&label=PyPI)](https://pypi.org/project/patchcraft/)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/patchcraft?color=3776AB)](https://pypi.org/project/patchcraft/)
[![License MIT](https://img.shields.io/badge/License-MIT-750014)](https://github.com/LeoPR/PatchCraft/blob/main/LICENSE)
[![Scope: one image at a time](https://img.shields.io/badge/scope-one%20image%20at%20a%20time-555555)](https://github.com/LeoPR/PatchCraft/blob/main/docs/SCOPE.md)

PatchCraft takes a single `(C, H, W)` float tensor, cuts it into a patch stack, and puts it back
together: exactly with `reconstruct`, or through a window that hides seams with `stitch` when a
model has rewritten the patches. On the return leg, `reconstruct` and `stitch` reject a geometry
that would drop or invent pixels instead of returning a plausible tensor. `extract` follows
whatever grid you hand it, so plan the geometry with `num_patches` or `tilings` first.

**The scope, stated once so you can stop here if it is wrong for you: one image at a time.**
No batching across images, no `Dataset`, no `DataLoader`, no training loop. Multi-image is your
`for` loop, your `torch.vmap`, or your `DataLoader` calling this once per item.

## Install

```bash
pip install patchcraft
```

> Distribution name `patchcraft`, import name `patchcraft`.

Python 3.12 or newer, tested on 3.12 and 3.13 across Ubuntu and Windows. Runtime dependencies:
`torch>=2.6`, `numpy>=1.26`, `pillow>=10`. PatchCraft ships no compiled code and no CUDA wheels
of its own, so it runs on whatever torch build you already have. One optional extra,
`pip install "patchcraft[cache]"`, adds `zstandard` so `Cache` compresses on disk. Without it,
`Cache` works and stores payloads uncompressed.

Float tensors only. An 8-bit image needs `image.float() / 255` before `extract`, which otherwise
raises torch's own `NotImplementedError: "im2col_out_cpu" not implemented for 'Byte'`.

## The contract, in six lines

```python
import torch
from patchcraft import extract, reconstruct

image = torch.rand(3, 256, 256)                      # any (C, H, W) float tensor
patches = extract(image, patch_size=32, stride=32)   # -> (L, C, ph, pw)
back = reconstruct(patches, image.shape, stride=32)  # -> (C, H, W)

assert tuple(patches.shape) == (64, 3, 32, 32)
assert torch.equal(back, image)                      # exact, not "close enough"
```

*Real output: 64 patches of 3x32x32, and `torch.equal` returns True on random `float32` data.
That is an equality assertion, not a tolerance.*

## Overlap: where exactness stops, and how to tell

Round-tripping with overlap divides each pixel by the number of patches covering it, so the
question is arithmetic rather than precision. A division is exact in binary floating point when
the divisor is a power of two, and every pixel carries its own divisor. Measured on 256x256
`float32` `torch.rand` images, on CPU, plus two smaller shapes chosen to break the tempting
shortcut:

| image | patch | stride | pixel coverage counts | all powers of two | `torch.equal` | max abs error |
|---|---:|---:|---|:---|:---|---:|
| 256x256 | 32 | 32 | 1 | yes | True | 0.0 |
| 256x256 | 32 | 16 | 1, 2, 4 | yes | True | 0.0 |
| 256x256 | 32 | 8 | 1, 2, 3, 4, 6, 8, 9, 12, 16 | no | **False** | 2.384e-07 |
| 256x256 | 64 | 32 | 1, 2, 4 | yes | True | 0.0 |
| 16x8 | (4, 8) | (1, 8) | 1, 2, 3, 4 | no | **False** | 5.960e-08 |
| 16x14 | (4, 5) | (4, 3) | 1, 2 | yes | True | 0.0 |

**The rule is the fourth column, not the count itself.** The round trip is `torch.equal` exact
when every value in the coverage map is a power of two, and lands within roughly `1e-7` in
`float32` otherwise. Row five is the counterexample worth keeping in mind: its largest coverage
count is 4 and it is still inexact, because a pixel covered by 3 patches is divided by 3. Row six
earns exactness with a stride that is neither the patch size nor half of it.

The coverage map factorises per axis, so you can evaluate the condition before you allocate
anything:

```python
def coverage_counts(length, patch, stride):
    """How many patches cover each pixel along one axis."""
    counts = [0] * length
    for start in range(0, length - patch + 1, stride):
        for i in range(start, start + patch):
            counts[i] += 1
    return set(counts)

def round_trip_is_exact(image_shape, patch, stride):
    _, h, w = image_shape
    counts = coverage_counts(h, patch, stride) | coverage_counts(w, patch, stride)
    return all(k & (k - 1) == 0 for k in counts)     # every count a power of two

print(round_trip_is_exact((3, 256, 256), 32, 16))    # True
print(round_trip_is_exact((3, 256, 256), 32, 8))     # False
```

*Real output: True then False, matching rows two and three of the table. Swept over the 116,964
legal geometries on images up to 16x16 in `float32`, the predicate promised exactness 57,121
times and was wrong in none of them.*

The everyday shorthand: `stride == patch_size` and `stride == patch_size / 2` always satisfy the
condition. They are sufficient rather than necessary, as row six shows. Moving to `float64`
shrinks the error outside the condition without reaching zero, which is why the deciding axis is
the coverage count and not the dtype. Half precision (`float16`, `bfloat16`) accumulates
internally in `float32` and casts back on return.

## Seams: `stitch` for patches you changed

`reconstruct` inverts `extract`. Once a model has rewritten each patch, the patches disagree
inside their overlaps, and uniform averaging turns that disagreement into a visible grid.

```python
import torch
from patchcraft import extract, stitch

clean = torch.linspace(0, 1, 512).repeat(512, 1).unsqueeze(0)  # a smooth ramp
patches = extract(clean, patch_size=128, stride=64)            # 7x7 grid, 50% overlap

g = torch.Generator().manual_seed(7)                           # stands in for a model that
bias = torch.rand(len(patches), 1, 1, 1, generator=g) * 0.2 - 0.1   # misses each patch's DC
out = patches + bias                                           # level by up to 0.10

uniform = stitch(out, clean.shape, stride=64, weight="uniform")   # "uniform" is the default
hann    = stitch(out, clean.shape, stride=64, weight="hann")

def seam(x):                           # largest |second difference| along one row
    a = (x - clean)[0, 256]
    return (a[2:] - 2 * a[1:-1] + a[:-2]).abs().max().item()

print(f"{seam(uniform):.6f}")                    # 0.018617
print(f"{seam(hann):.6f}")                       # 0.000097
print(f"{seam(uniform) / seam(hann):.1f}x")      # 191.0x
```

*Real output: on this geometry the uniform seam is 191 times the Hann seam, and the uniform steps
sit exactly on the six patch boundaries crossing that row.*

**Three honest details.** The 191x belongs to this geometry: rerun the same block with
`patch_size=8, stride=4` and the ratio is 2.1x, because Hann spreads the disagreement across an
overlap that widens with the patch while the uniform step stays one pixel wide. Hann costs
fidelity to the model's own output, scoring 27.14 dB mean against the modified patches where
uniform scores 29.60 dB, so it buys smoothness with accuracy. The scale-invariant figure is the
better headline: across 50% overlap at 32x32, 64x64, 128x128, 256x256 and 512x512, uniform put
96% of that row's total variation on the patch boundaries every time, which is exactly what the
eye reads as tiling.

The third window, `weight="gaussian"`, holds more weight at the patch edge than Hann does, so it
lands between uniform and Hann: on the geometry above it scores a seam of 0.007326 against Hann's
0.000097. Hann is the seam suppressor, gaussian is the gentler taper, and both are strictly
positive on every sample, so neither discards patch content.

## The public surface, and what it costs

Nineteen symbols, all importable from `patchcraft`.

| Symbol | Call | Cost |
|---|---|---|
| `extract` | `extract(image, patch_size, stride, dilation=1)` | allocates the whole `(L, C, ph, pw)` stack |
| `Patchify` | `Patchify(patch_size, stride)`, a callable transform | same as `extract`, as one step in a `Compose` |
| `reconstruct` | `reconstruct(patches, image_shape, stride)` | one `fold` plus a coverage map |
| `stitch` | `stitch(patches, image_shape, stride, *, weight="uniform")` | `reconstruct` plus one window kernel; `WeightKind` is `"uniform"`, `"hann"`, `"gaussian"` |
| `pair` | `pair(lr_image, hr_image, lr_patch_size, scale_factor, stride)` | a `PatchPair` holding aligned low-resolution and high-resolution stacks, one `PatchMeta` per patch |
| `resize` | `resize(image, target_size, backend="pil")` | one resample, tensor or PIL in and out |
| `Cache` | `Cache(root, namespace, version=1)`, then `put` and `get` | bytes on disk, compressed when `zstandard` is installed |
| geometry | `num_patches`, `tilings`, `paired_tilings`, `scale_factor` | arithmetic over shapes, allocating no image data; `tilings` and `paired_tilings` return lists of `TilingSpec` and `PairedTilingSpec`, and `scale_factor` returns an `int`, or `None` when the ratio is not an integer |
| metrics | `patch_metrics`, `per_patch_mse`, `per_patch_psnr` | one pass over two stacks |

Ask the geometry before you allocate. On one CPU machine,
`num_patches((3, 1024, 1920), 256, 128)` answered `(7, 14)` in a few microseconds and allocated
nothing, while the `extract` it predicts took 148 ms and allocated 74 MiB. The ratio is the
stable claim; the absolute timings belong to that machine.

## Why not `unfold` and `fold` directly

You can, and PatchCraft is a thin contract over exactly those two calls. Tiling an image, running
a per-patch model and blending back with a Hann window took 17 body lines by hand against 3 using
`extract` and `stitch`, and the two outputs were bit-identical (`torch.equal` True, maximum
absolute difference 0.0) on a 3x128x128 image at patch 32, stride 16. The three lines buy the
boundary checks. A stride that fails to tile the image exactly silently zeroes pixels through a
raw `fold`, 23% of the image in the case below, and here it says so instead:

```
ValueError: patch grid leaves pixels uncovered (partial coverage forbidden): image_shape=(1, 128, 128), patch_size=(32, 32), stride=(20, 20) covers (112, 112) of (128, 128). Choose a geometry with exact coverage (see patchcraft.tilings).
```

## What PatchCraft is not

It is not a dataset library: no `Dataset`, no sampler, no collate function. It is not a training
framework: no models, no losses, no schedulers, no checkpoints. It is not batched, by decision
rather than by omission, so `extract` takes `(C, H, W)` and rejects `(N, C, H, W)`. It is not an
image codec, since the round trip compresses nothing and `Cache` is a disk cache for bytes you
already hold. Reversible patch geometry is the contract, seam control is the consequence.

## Status

This page documents 0.2.1, pre-1.0. The API can change between minor versions until 1.0, and the
[changelog](https://github.com/LeoPR/PatchCraft/blob/main/CHANGELOG.md) records what moved. 346
tests pass in the current full local run, so run `pytest` for the number in your own environment.

Four limits are worth knowing before you depend on it. Every measurement on this page is CPU
only: device is preserved through the pipeline, and the CUDA paths are not exercised by the test
matrix, so the exactness figures stay unverified on GPU. Several docstrings and prose sections
shipped in 0.2.1 state the overlap round trip as exact without the coverage condition above,
which this page corrects and the next release fixes at the source. The uncompressed `Cache`
branch, the one a plain `pip install patchcraft` runs, is covered by no CI job, because every job
installs the `cache` extra. And no downstream project has consumed the published API in real use
yet, which is this project's own stated gate for calling the shape settled.

## Documentation

- [Repository and full README](https://github.com/LeoPR/PatchCraft)
- [Theory and contract](https://github.com/LeoPR/PatchCraft/blob/main/docs/THEORY.md), the geometry and the guarantees
- [Usage guide](https://github.com/LeoPR/PatchCraft/blob/main/docs/USAGE.md), worked examples per symbol, currently verified against 0.2.0
- [Scope](https://github.com/LeoPR/PatchCraft/blob/main/docs/SCOPE.md) and [roadmap](https://github.com/LeoPR/PatchCraft/blob/main/docs/ROADMAP.md)
- [Architecture decisions](https://github.com/LeoPR/PatchCraft/tree/main/docs/ADR)
- [Changelog](https://github.com/LeoPR/PatchCraft/blob/main/CHANGELOG.md)
- [Contributing](https://github.com/LeoPR/PatchCraft/blob/main/CONTRIBUTING.md)
- [Issues](https://github.com/LeoPR/PatchCraft/issues)

## License

MIT. See [LICENSE](https://github.com/LeoPR/PatchCraft/blob/main/LICENSE).