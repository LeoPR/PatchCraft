# PatchCraft

Encode one image into patches, decode it back, and decide what happens at the seams.

[![Latest version on PyPI](https://img.shields.io/pypi/v/patchcraft?color=3775A9&label=PyPI)](https://pypi.org/project/patchcraft/)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/patchcraft?color=3776AB)](https://pypi.org/project/patchcraft/)
[![License MIT](https://img.shields.io/badge/License-MIT-750014)](https://github.com/LeoPR/PatchCraft/blob/main/LICENSE)
[![Scope: one image at a time](https://img.shields.io/badge/scope-one%20image%20at%20a%20time-555555)](https://github.com/LeoPR/PatchCraft/blob/main/docs/SCOPE.md)

PatchCraft takes a single `(C, H, W)` float tensor, cuts it into a stack of patches, and puts the image back together. It owns the `unfold` and `fold` arithmetic, the geometry validation and the seam blending, so that your pipeline can own everything else.

The scope is one image at a time, and that is worth knowing before you install anything, because it is the constraint that decides whether PatchCraft fits your problem at all. There is no batching across images, no `Dataset`, no `DataLoader` and no training loop, so multi-image work stays in your own `for` loop, in your `torch.vmap`, or in your `DataLoader` calling this once per item.

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

This page is the short one. The manual is [docs/GUIDE.md](https://github.com/LeoPR/PatchCraft/blob/main/docs/GUIDE.md), which carries the measurements, the tables and the long examples.

## Install

```bash
pip install patchcraft
pip install "patchcraft[cache]"     # adds zstandard, which compresses Cache payloads
```

There is nothing else to install for speed. On Windows x64, Linux x86_64 and
aarch64, and both macOS architectures, the wheel carries a Rust accelerator for
the overlapping fold, which is where `reconstruct` and `stitch` spend their
time. Every other platform gets the universal wheel and runs the pure-torch
paths, which return the same values.

`patchcraft.accel_available()` reports at runtime which one you got, and
`PATCHCRAFT_ACCEL=0` in the environment forces the pure path.

The distribution name and the import name are both `patchcraft`. The cache extra is optional, because `Cache` works without `zstandard` as well and simply stores its payload uncompressed.

## Sixty seconds

```python
import torch
from patchcraft import extract, reconstruct, stitch

image = torch.rand(3, 256, 256)                      # one float (C, H, W) tensor
patches = extract(image, patch_size=32, stride=16)   # (L, C, ph, pw) == (225, 3, 32, 32)

back = reconstruct(patches, image.shape, stride=16)  # the patches came back untouched
assert torch.equal(back, image)                      # the same tensor, bit for bit

edited = patches * 1.01                              # stands in for a per-patch model
blended = stitch(edited, image.shape, stride=16, weight="hann")
assert blended.shape == image.shape                  # seams smoothed, geometry preserved
```

PatchCraft accepts float tensors only. An 8-bit image has to become `image.float() / 255` before it reaches `extract`, because `extract` passes the tensor straight to `F.unfold`, and torch has no integer kernel there: it raises `NotImplementedError: "im2col_out_cpu" not implemented for 'Byte'`.

## reconstruct or stitch

`reconstruct` is the inverse of `extract`. It assumes the patches still hold the pixels `extract` gave you, it divides each pixel by the number of patches that covered it, and on the geometries described further down it hands the image back bit for bit.

`stitch` is for patches a model rewrote, because neighbours now disagree about the pixels they share, and that disagreement lands on the grid lines unless something spreads it.

| Call | Use it when | What it does at the overlaps |
|---|---|---|
| `reconstruct` | the patches are the ones `extract` produced, or you only read them | divides each pixel by how many patches covered it, which inverts `extract` |
| `stitch` | a model rewrote the patches, so neighbours now disagree | weights each patch through `"uniform"`, `"hann"` or `"gaussian"` before averaging |

Uniform averaging is the default, and it is the option that reports what the model actually produced, since it changes no value beyond dividing by the count. The price is a straight line of disagreement along every patch boundary, and that is what the eye reads as tiling. A Hann window spreads the same disagreement across the whole overlap instead, so the seam stops being visible, and what it costs is a little fidelity to the values the model returned.

## Why not unfold and fold directly

Nothing stops you, and PatchCraft is a thin contract over exactly those two calls. What the contract buys is the pixel order and the boundary checks, because the intuitive reshape after `F.unfold` returns a tensor of the right shape whose pixels are scrambled.

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

The saving is real on the other side too. Tiling an image, running a per-patch model and blending the result back with a Hann window took 17 non-blank lines by hand against 3 with `extract` and `stitch`, and the two outputs were bit-identical.

## The geometry has to cover the image

`extract` follows whatever grid you hand it, but `reconstruct` and `stitch` refuse a grid that does not cover the image exactly, rather than returning a plausible tensor built on missing pixels. On a 128x128 image with `patch_size=32` and `stride=20` the grid reaches only 112x112, which leaves 3840 of the 16384 pixels at zero, and the error message names that covered extent instead of hiding it.

The answer is to pick a legal geometry rather than to pad the image into one, because padding synthesizes pixels you never had. `tilings(image_shape)` enumerates the legal geometries from the shape alone and allocates nothing while it does so, so you can call it before you have committed to anything: a 28x28 image has 5 exact tilings, and 73 of them once `allow_overlap=True` lets the patches overlap.

Two narrower questions have their own entry points. `num_patches` takes a geometry you already have in mind and returns the grid it implies, and `paired_tilings` is the one to reach for when a low-resolution image and a high-resolution image have to stay aligned patch for patch.

## What you are getting into

The surface is one tensor in and one tensor out, with no batch axis anywhere in the signature, so `extract` accepts `(C, H, W)` and rejects `(N, C, H, W)` by decision rather than by omission. It is a geometry library and nothing else, which means it ships no models, no losses and no `Dataset`, and the one confusion worth heading off is compression: the round trip keeps every pixel it started with, and `Cache` only writes bytes you already hold.

It helps when you tile one image for an inference pass too large to run in a single forward call, when you build aligned low-resolution and high-resolution patch pairs, and when you run a sliding window analysis and need the pieces to go back together exactly.

## When the round trip is bit for bit

The round trip is exact when every value in the count map is a power of two. The reason is that reconstruction divides each pixel by the number of patches that covered it, and dividing a float by a power of two is the one division that never rounds.

That makes the geometry the deciding axis rather than the dtype, so `float64` is not a safe harbour: outside the rule the per-pixel error is bounded by `(k+1)·eps·|v|`, with `k` the pixel's coverage count. A wider float buys a smaller miss and never exactness.

The everyday shorthand is that `stride == patch_size` and `stride == patch_size / 2` always satisfy the rule. Both are sufficient conditions rather than necessary ones, so a geometry outside them can still be exact, and the [guide](https://github.com/LeoPR/PatchCraft/blob/main/docs/GUIDE.md#4-when-the-round-trip-is-bit-for-bit) carries the sweep that measures it.

## Status

This page documents 0.6.0, which is pre-1.0, so both the output values and the API shape can still change in a minor release, and the [changelog](https://github.com/LeoPR/PatchCraft/blob/main/CHANGELOG.md) is where each of those changes is recorded with the measurement behind it. The suite collects 1571 tests and passes on Python 3.12, 3.13 and 3.14, on Ubuntu and on Windows alike.

Two limits are worth knowing before you depend on it. Every figure on this page was measured on CPU, and no CUDA path has ever executed in the test matrix, so the pipeline does preserve the device you hand it while the exactness numbers stay unverified on GPU. The other limit is that no external project has consumed the published API in real use yet, and that consumption is this project's own stated gate for calling the shape settled.

## Documentation

- [Guide](https://github.com/LeoPR/PatchCraft/blob/main/docs/GUIDE.md), the manual, with every figure on this page shown as runnable code
- [Usage](https://github.com/LeoPR/PatchCraft/blob/main/docs/USAGE.md), a walkthrough of each of the 20 public symbols
- [Theory](https://github.com/LeoPR/PatchCraft/blob/main/docs/THEORY.md), the math and the per-function contract
- [Scope](https://github.com/LeoPR/PatchCraft/blob/main/docs/SCOPE.md), the line between this library and your pipeline
- [Repository](https://github.com/LeoPR/PatchCraft), [issues](https://github.com/LeoPR/PatchCraft/issues) and [contributing](https://github.com/LeoPR/PatchCraft/blob/main/CONTRIBUTING.md)

## License and citation

MIT, in [LICENSE](https://github.com/LeoPR/PatchCraft/blob/main/LICENSE). There is no DOI yet, so if you need to cite this work the BibTeX entry is in the [guide](https://github.com/LeoPR/PatchCraft/blob/main/docs/GUIDE.md#9-install-details-and-citation).
