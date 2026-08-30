# PatchCraft

[![CI status for main](https://github.com/LeoPR/PatchCraft/actions/workflows/test.yml/badge.svg)](https://github.com/LeoPR/PatchCraft/actions/workflows/test.yml)
[![PyPI version](https://img.shields.io/pypi/v/patchcraft.svg)](https://pypi.org/project/patchcraft/)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/patchcraft.svg)](https://pypi.org/project/patchcraft/)
[![License MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/LeoPR/PatchCraft/blob/main/LICENSE)

**Encode one image into patches, decode it back.** PatchCraft owns the `unfold`/`fold` arithmetic, the geometry validation and the seam blending. Your pipeline owns everything else.

**One image at a time, on purpose.** Every call takes one `(C, H, W)` float tensor and returns one tensor. There is no batch axis, no dataset, no dataloader and no training. Your `for` loop, `torch.vmap` or `DataLoader` supplies the batching, and [Scope](#scope-one-image-at-a-time) argues why.

> **What if the tile-and-blend block you rewrite in every image project were three lines, with the geometry checked before it runs instead of after the output looks wrong?**

Every fenced output block on this page is verbatim printed output of the code shown directly above it, run against `patchcraft` 0.2.1 on CPU (Python 3.13.13, torch 2.13.0+cpu). Figures quoted in prose are read off those blocks or are arithmetic on them, with two exceptions that name their own source: the test-suite counts in [Status and maturity](#status-and-maturity), and the file and line references in [Exactness, measured](#exactness-measured).

**Contents.** [Why not just `unfold` and `fold`](#why-not-just-unfold-and-fold) · [What the artifact looks like](#what-the-artifact-looks-like) · [Use this when](#use-this-when) · [Install](#install) · [One minute](#one-minute) · [Scope](#scope-one-image-at-a-time) · [Exactness, measured](#exactness-measured) · [Seams](#seams-stitch-against-reconstruct) · [Planning the geometry](#planning-the-geometry-before-you-allocate) · [LR and HR pairs](#lr-and-hr-pairs) · [The public API](#the-public-api-19-symbols) · [Status and maturity](#status-and-maturity) · [Contributing](#contributing-and-running-the-tests) · [How to cite](#how-to-cite)

## Why not just `unfold` and `fold`

Because two defects wait there, both silent, and you meet them in the first hour.

**The reshape after `unfold`.** `F.unfold` returns `(1, C*ph*pw, L)`. The intuitive reshape into `(L, C, ph, pw)` produces the right shape, raises nothing, and scrambles the pixels.

```python
import torch
import torch.nn.functional as F
from patchcraft import extract

image = torch.arange(64, dtype=torch.float32).reshape(1, 8, 8)
patches = extract(image, patch_size=4, stride=4)                # (L, C, ph, pw)
cols = F.unfold(image.unsqueeze(0), kernel_size=4, stride=4)    # (1, C*ph*pw, L)

wrong = cols[0].view(-1, 1, 4, 4)                               # the intuitive reshape
right = cols[0].view(1, 4, 4, -1).permute(3, 0, 1, 2)           # the correct one

assert wrong.shape == right.shape == patches.shape              # all (4, 1, 4, 4)
assert not torch.equal(wrong, patches)
assert torch.equal(right, patches)

print("patch 0 from extract():"); print(patches[0, 0])
print("patch 0 from the intuitive reshape:"); print(wrong[0, 0])
```

```
patch 0 from extract():
tensor([[ 0.,  1.,  2.,  3.],
        [ 8.,  9., 10., 11.],
        [16., 17., 18., 19.],
        [24., 25., 26., 27.]])
patch 0 from the intuitive reshape:
tensor([[ 0.,  4., 32., 36.],
        [ 1.,  5., 33., 37.],
        [ 2.,  6., 34., 38.],
        [ 3.,  7., 35., 39.]])
```

*Both tensors have shape `(4, 1, 4, 4)`. Only the values tell you which one is your image.*

**A stride that does not cover the image.** With `patch=32, stride=20` on a 128 pixel axis the last patch ends at pixel 112. A hand-rolled `fold` divides by a count map that is zero on the remaining band and hands back a partly black image.

```python
import torch
import torch.nn.functional as F
from patchcraft import extract, reconstruct

torch.manual_seed(0)
image = torch.rand(1, 128, 128)
patch, stride = 32, 20                      # 5x5 grid, the last patch ends at pixel 112

cols = F.unfold(image.unsqueeze(0), kernel_size=patch, stride=stride)
count = F.fold(torch.ones_like(cols), output_size=(128, 128), kernel_size=patch, stride=stride)
print("hand-rolled fold, pixels covered by no patch:", int((count == 0).sum()), "of", 128 * 128)

try:
    reconstruct(extract(image, patch_size=patch, stride=stride), image.shape, stride=stride)
except ValueError as error:
    print("patchcraft.reconstruct:", error)
```

```
hand-rolled fold, pixels covered by no patch: 3840 of 16384
patchcraft.reconstruct: patch grid leaves pixels uncovered (partial coverage forbidden): image_shape=torch.Size([1, 128, 128]), patch_size=(32, 32), stride=(20, 20) covers (112, 112) of (128, 128). Choose a geometry with exact coverage (see patchcraft.tilings).
```

*23% of that image comes back zeroed with no exception raised. `reconstruct` refuses the geometry and names the covered extent against the requested one.*

Here is the whole job, tiling one image with overlap, running a per-patch model and blending the result back, written both ways.

```python
import inspect

import torch
import torch.nn.functional as F
from patchcraft import extract, stitch


def tile_and_blend_by_hand(image, model, patch, stride):
    c, h, w = image.shape
    nh = (h - patch) // stride + 1
    nw = (w - patch) // stride + 1
    if (nh - 1) * stride + patch != h or (nw - 1) * stride + patch != w:
        raise ValueError("grid does not cover the image exactly")
    cols = F.unfold(image.unsqueeze(0), kernel_size=patch, stride=stride)
    patches = cols[0].view(c, patch, patch, -1).permute(3, 0, 1, 2).contiguous()
    out = model(patches)
    win = torch.hann_window(patch + 2, periodic=False, dtype=out.dtype)[1:-1]
    kernel = win.unsqueeze(1) * win.unsqueeze(0)
    weighted = out * kernel
    num = weighted.permute(1, 2, 3, 0).reshape(c * patch * patch, -1).unsqueeze(0)
    num = F.fold(num, output_size=(h, w), kernel_size=patch, stride=stride)
    den = kernel.flatten().unsqueeze(1).repeat(1, nh * nw).unsqueeze(0)
    den = F.fold(den, output_size=(h, w), kernel_size=patch, stride=stride)
    return (num / den.clamp(min=1e-6))[0]


def tile_and_blend(image, model, patch, stride):
    patches = extract(image, patch_size=patch, stride=stride)
    return stitch(model(patches), image.shape, stride=stride, weight="hann")


def model(p):
    return p * 1.05 - 0.01                  # stands in for a real per-patch model


def statements(function):                   # non-blank lines, the def line included
    return sum(1 for line in inspect.getsource(function).splitlines() if line.strip())


torch.manual_seed(0)
image = torch.rand(3, 128, 128)
by_hand = tile_and_blend_by_hand(image, model, patch=32, stride=16)
with_patchcraft = tile_and_blend(image, model, patch=32, stride=16)

assert torch.equal(by_hand, with_patchcraft)
print(tuple(with_patchcraft.shape), with_patchcraft.dtype)
print("max_abs difference:", (by_hand - with_patchcraft).abs().max().item())
print("lines:", statements(tile_and_blend_by_hand), "by hand,",
      statements(tile_and_blend), "with patchcraft")
```

```
(3, 128, 128) torch.float32
max_abs difference: 0.0
lines: 17 by hand, 3 with patchcraft
```

*Real output: the two results are the same tensor to the last bit, not merely close.*

Two honest caveats before you keep reading. PatchCraft computes nothing torch cannot: it is the same `unfold`, the same `fold`, the same count map, and it is not faster. And the coverage defect above shipped inside PatchCraft itself, since 0.2.0 validated the patch count and never the coverage, so it returned partly black images until 0.2.1 fixed it ([CHANGELOG.md](https://github.com/LeoPR/PatchCraft/blob/main/CHANGELOG.md)). The argument for the library is that the guard is written once and regression tested, not that it was obvious.

## What the artifact looks like

`extract` turns one `(C, H, W)` image into one `(L, C, ph, pw)` stack, row-major, no padding, no batch axis.

```
   image (1, 8, 8)              patches (4, 1, 4, 4)
   +--------+--------+
   |        |        |          [0] top-left      [1] top-right
   |   A    |   B    |          [2] bottom-left   [3] bottom-right
   |        |        |
   +--------+--------+          patch k starts at
   |        |        |          (k // 2 * stride, k % 2 * stride)
   |   C    |   D    |
   |        |        |
   +--------+--------+
```

```python
import torch
from patchcraft import extract

image = torch.arange(64, dtype=torch.float32).reshape(1, 8, 8)
patches = extract(image, patch_size=4, stride=4)

assert tuple(patches.shape) == (4, 1, 4, 4)          # (L, C, ph, pw)
print("top-left value of each patch:", patches[:, 0, 0, 0].tolist())
print("patch 2:"); print(patches[2, 0])
```

```
top-left value of each patch: [0.0, 4.0, 32.0, 36.0]
patch 2:
tensor([[32., 33., 34., 35.],
        [40., 41., 42., 43.],
        [48., 49., 50., 51.],
        [56., 57., 58., 59.]])
```

The one object to carry into everything below is the **count map**: how many patches cover each pixel. `reconstruct` builds it internally and divides the folded sum by it, and `stitch` divides by the folded window instead. It is not a public symbol, so the block below rebuilds it with `F.fold` to make the arithmetic visible, taking the grid size from `num_patches`.

```python
import torch
import torch.nn.functional as F
from patchcraft import num_patches

shape, patch, stride = (1, 8, 8), 4, 2
nh, nw = num_patches(shape, patch_size=patch, stride=stride)   # (3, 3), a 3x3 grid

# One value of 1.0 per pixel per patch, folded over that grid.
count = F.fold(torch.ones(1, patch * patch, nh * nw), output_size=(8, 8),
               kernel_size=patch, stride=stride)
print(count[0, 0].int())
print("distinct counts:", sorted({int(v) for v in count.unique()}))
```

```
tensor([[1, 1, 2, 2, 2, 2, 1, 1],
        [1, 1, 2, 2, 2, 2, 1, 1],
        [2, 2, 4, 4, 4, 4, 2, 2],
        [2, 2, 4, 4, 4, 4, 2, 2],
        [2, 2, 4, 4, 4, 4, 2, 2],
        [2, 2, 4, 4, 4, 4, 2, 2],
        [1, 1, 2, 2, 2, 2, 1, 1],
        [1, 1, 2, 2, 2, 2, 1, 1]], dtype=torch.int32)
distinct counts: [1, 2, 4]
```

**How to read it.** Corners are covered once, edges twice, the interior four times. When `stride == patch_size` the map is all ones and the division is a no-op. Which values appear in this map, and specifically whether every one of them is a power of two, is what decides whether the round trip comes back bit for bit. That is the subject of [Exactness, measured](#exactness-measured).

## Use this when

Use PatchCraft when you tile one image, do something per patch, and put it back: super-resolution and denoising inference on images too large for a single forward pass, patch datasets for LR/HR training, sliding-window analysis, patch-level error maps.

Do not use PatchCraft when you want a batched op over N images (it takes one image, on purpose, see [Scope](#scope-one-image-at-a-time)), a dataset or a dataloader (it has none), padding to make an awkward geometry fit (it refuses instead of synthesizing pixels), or a model (it is infrastructure, not a network).

## Install

```
pip install patchcraft
```

Distribution name and import name are both `patchcraft`. Runtime dependencies are `torch>=2.6`, `numpy>=1.26` and `pillow>=10`.

```
pip install "patchcraft[cache]"     # adds zstandard, compresses Cache payloads
```

From source, for development:

```
git clone https://github.com/LeoPR/PatchCraft.git
cd PatchCraft
pip install -e ".[dev,cache]"
```

**Python.** Tested on 3.12 and 3.13, on Ubuntu and Windows, which is what CI runs and what the classifiers advertise. `requires-python` is `>=3.12` with no ceiling, so pip will also install this on 3.14, where nothing has been measured.

**GPU.** Install a matching torch wheel first, following [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/). Read [Status and maturity](#status-and-maturity) before you do: no CUDA path in this library has ever been executed.

## One minute

```python
import torch
from patchcraft import extract, reconstruct, stitch

torch.manual_seed(0)
image = torch.rand(3, 256, 256)                         # float (C, H, W), any device

patches = extract(image, patch_size=32, stride=32)      # (L, C, ph, pw)
assert tuple(patches.shape) == (64, 3, 32, 32)

back = reconstruct(patches, image.shape, stride=32)     # the inverse
assert torch.equal(back, image)                         # same tensor, bit for bit

edited = patches * 1.05 - 0.01                          # stands in for a model
blended = stitch(edited, image.shape, stride=32, weight="hann")
assert blended.shape == image.shape and blended.dtype == image.dtype
print(tuple(patches.shape), tuple(back.shape), tuple(blended.shape))
```

```
(64, 3, 32, 32) (3, 256, 256) (3, 256, 256)
```

That is the whole loop: `extract`, your work, then `reconstruct` for unmodified patches or `stitch` for modified ones. Dtype and device of the input are preserved in both directions.

**Float tensors only.** `extract` hands the tensor straight to `F.unfold`, which has no integer kernel, so an 8-bit image fails with torch's own message rather than with one of ours. Convert on the way in, which costs one division.

```python
import torch
from patchcraft import extract

eight_bit = (torch.rand(1, 64, 64) * 255).to(torch.uint8)   # what read_image or PIL gives you
try:
    extract(eight_bit, patch_size=16, stride=16)
except Exception as error:
    print(type(error).__name__ + ":", error)

patches = extract(eight_bit.float() / 255, patch_size=16, stride=16)
print(tuple(patches.shape), patches.dtype)
```

```
NotImplementedError: "im2col_out_cpu" not implemented for 'Byte'
(16, 1, 16, 16) torch.float32
```

*`reconstruct` and `stitch` raise a framed `ValueError` on integer input that names the conversion. `extract` does not, and that asymmetry is a rough edge rather than a design.*

End to end from a file, using the `pillow` and `numpy` that PatchCraft already depends on:

```python
import numpy as np
import torch
from PIL import Image
from patchcraft import extract, reconstruct

path = "example.png"                                  # any 8-bit image on disk
Image.effect_mandelbrot((96, 96), (-2, -1.5, 1, 1.5), 40).convert("L").save(path)

picture = Image.open(path)
image = torch.from_numpy(np.array(picture)).unsqueeze(0).float() / 255   # (C, H, W), float
print(picture.size, picture.mode, "->", tuple(image.shape), image.dtype)

patches = extract(image, patch_size=32, stride=16)
back = reconstruct(patches, image.shape, stride=16)
print(tuple(patches.shape), "and back to", tuple(back.shape), "exact:", torch.equal(back, image))

Image.fromarray((back[0] * 255).round().byte().numpy()).save("roundtrip.png")
```

```
(96, 96) L -> (1, 96, 96) torch.float32
(25, 1, 32, 32) and back to (1, 96, 96) exact: True
```

*The block writes `example.png` and `roundtrip.png` into the working directory, and the two files hold the same pixels. 96 with `patch=32, stride=16` is a covering geometry whose count map holds only 1, 2 and 4, which is why the round trip is exact here.*

## Scope: one image at a time

Every primitive takes one `(C, H, W)` tensor and returns one tensor. There is no batch axis, no dataset, no dataloader, no training. This is a decision, recorded as binding in [docs/THEORY.md](https://github.com/LeoPR/PatchCraft/blob/main/docs/THEORY.md) §0, not a gap waiting to be filled.

The reason is that patch counts depend on the image: different `H` and `W` give different `L`. A batched API would have to pad every image to a common geometry or return a list of tensors. Padding synthesizes pixels the caller never had, and a list-of-tensors return hands the same loop back to you with a worse type. So the loop stays yours, and it costs one line.

```python
import torch
from patchcraft import extract

torch.manual_seed(0)
batch = torch.rand(8, 3, 64, 64)                     # your batch, not PatchCraft's

loop = torch.stack([extract(im, patch_size=16, stride=16) for im in batch])
vmapped = torch.vmap(lambda im: extract(im, patch_size=16, stride=16))(batch)

assert torch.equal(loop, vmapped)
print(tuple(loop.shape))                             # (N, L, C, ph, pw)
```

```
(8, 16, 3, 16, 16)
```

*Real output: `torch.vmap` over the single-image function gives the batch axis back, and its result is bit-identical to the Python loop.*

The same boundary is what makes `Patchify` composable. It is a callable holding only the geometry, so it drops into a `torchvision` pipeline and `DataLoader` supplies the batching and the worker parallelism.

```python
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from patchcraft import Patchify

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.GaussianBlur(kernel_size=3),
    Patchify(patch_size=4, stride=2),          # PatchCraft as one step among many
])
print(transform)

torch.manual_seed(0)
images = [transforms.ToPILImage()(torch.rand(1, 28, 28)) for _ in range(8)]
one = transform(images[0])
assert tuple(one.shape) == (169, 1, 4, 4)      # (L, C, ph, pw), one image

batch = next(iter(DataLoader([transform(i) for i in images], batch_size=8)))
print(tuple(batch.shape))                      # DataLoader adds the batch axis, PatchCraft never does
```

```
Compose(
    ToTensor()
    GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))
    Patchify(patch_size=(4, 4), stride=(2, 2), dilation=(1, 1))
)
(8, 169, 1, 4, 4)
```

*`torchvision` is a development dependency of this repository and never a runtime dependency of the package. The block above needs it installed; nothing else on this page does.*

## Exactness, measured

`reconstruct` folds the patches and divides by the count map. Whether that returns the original tensor bit for bit is decided by the values in that map.

```python
import torch
import torch.nn.functional as F
from patchcraft import extract, num_patches, reconstruct


def count_map(shape, patch, stride):
    """How many patches cover each pixel, for this geometry."""
    _, h, w = shape
    nh, nw = num_patches(shape, patch_size=patch, stride=stride)
    ones = torch.ones(1, patch * patch, nh * nw)
    return F.fold(ones, output_size=(h, w), kernel_size=patch, stride=stride)[0, 0]


shape = (3, 256, 256)
print(f"{'dtype':>10}{'patch':>7}{'stride':>8}{'max count':>11}{'all 2^n':>9}{'exact':>8}{'max_abs':>11}")
for dtype in (torch.float64, torch.float32, torch.bfloat16, torch.float16):
    for patch, stride in ((32, 32), (32, 16), (32, 8)):
        torch.manual_seed(0)
        image = torch.rand(*shape, dtype=dtype)
        back = reconstruct(extract(image, patch_size=patch, stride=stride),
                           image.shape, stride=stride)
        counts = {int(v) for v in count_map(shape, patch, stride).unique()}
        power_of_two = all(k & (k - 1) == 0 for k in counts)
        error = (back.double() - image.double()).abs().max().item()
        print(f"{str(dtype).replace('torch.', ''):>10}{patch:>7}{stride:>8}{max(counts):>11}"
              f"{str(power_of_two):>9}{str(torch.equal(back, image)):>8}{error:>11.3e}")
```

```
     dtype  patch  stride  max count  all 2^n   exact    max_abs
   float64     32      32          1     True    True  0.000e+00
   float64     32      16          4     True    True  0.000e+00
   float64     32       8         16    False   False  4.441e-16
   float32     32      32          1     True    True  0.000e+00
   float32     32      16          4     True    True  0.000e+00
   float32     32       8         16    False   False  2.384e-07
  bfloat16     32      32          1     True    True  0.000e+00
  bfloat16     32      16          4     True    True  0.000e+00
  bfloat16     32       8         16    False    True  0.000e+00
   float16     32      32          1     True    True  0.000e+00
   float16     32      16          4     True    True  0.000e+00
   float16     32       8         16    False    True  0.000e+00
```

**The rule.** A pixel covered by `k` patches is summed `k` times and divided by `k`. In binary floating point that division is exact when `k` is a power of two and rounds otherwise, so the round trip in `float32` and `float64` is `torch.equal` exact **when every value in the count map is a power of two**. Two geometries guarantee it without any checking: `stride == patch_size` on both axes, where the map is all ones, and `stride == patch_size / 2`, where the map holds only 1, 2 and 4.

**The maximum of the map is not the rule**, which is worth showing because it is the shape the mistake takes. Here the maximum is 4 and the round trip still misses, because 3 is in the map.

```python
import torch
import torch.nn.functional as F
from patchcraft import extract, num_patches, reconstruct

shape, patch, stride = (3, 16, 8), (4, 8), (1, 8)     # legal, full coverage
torch.manual_seed(0)
image = torch.rand(*shape)

nh, nw = num_patches(shape, patch_size=patch, stride=stride)
ones = torch.ones(1, patch[0] * patch[1], nh * nw)
count = F.fold(ones, output_size=shape[1:], kernel_size=patch, stride=stride)[0, 0]
back = reconstruct(extract(image, patch_size=patch, stride=stride), image.shape, stride=stride)

print("distinct counts:", sorted({int(v) for v in count.unique()}))
print("maximum count  :", int(count.max()))
print("torch.equal    :", torch.equal(back, image))
print("max_abs        :", f"{(back - image).abs().max().item():.3e}")
```

```
distinct counts: [1, 2, 3, 4]
maximum count  : 4
torch.equal    : False
max_abs        : 5.960e-08
```

Three honest details.

- **Outside the predicate, exactness is a property of the data, not of the geometry.** The guarantee runs one way only: all powers of two means exact, and anything else means not guaranteed. Some inputs do come back exact on a mixed map by luck of rounding, so a geometry that survived one image says nothing about the next. What holds either way is the size of the miss, `2.384e-07` in float32 and `4.441e-16` in float64 on the geometries above, which is one ULP territory.
- **float64 is not a safe harbour.** The deciding axis is the count map, not the dtype. float64 misses the round trip at exactly the geometry float32 does, at `4.441e-16` instead of `2.384e-07`.
- **Half precision comes back exact here for a reason worth knowing, and not by a stronger guarantee.** 0.2.1 accumulates `float16` and `bfloat16` in a float32 buffer and rounds once on return, and that final rounding is far coarser than the float32 error, so these rows land back on the value they started from. The promotion exists because the folded sum overflows the fp16 finite range before the division happens.

**The library says this more broadly than the measurements support, in seven places.** `reconstruct`'s docstring (`src/patchcraft/reconstruct.py:24`), `stitch`'s module and function docstrings (`src/patchcraft/stitch.py:3` and `:89`), [THEORY §2.5](https://github.com/LeoPR/PatchCraft/blob/main/docs/THEORY.md) at line 153, [USAGE §5 and §6](https://github.com/LeoPR/PatchCraft/blob/main/docs/USAGE.md) at lines 154 and 163, and [SCOPE](https://github.com/LeoPR/PatchCraft/blob/main/docs/SCOPE.md) at line 229 all state the overlap round trip as exact with no condition on the count map, and USAGE line 154 names `float64` as the safe case, which the table above contradicts. [ADR 0003](https://github.com/LeoPR/PatchCraft/blob/main/docs/ADR/0003-reversibility-classes.md) is where the boundary is being written down, and it is still **Proposed**, so none of that wording has landed yet. THEORY §9.2 carries a second contradiction of its own: it documents the half-precision promotion at line 269 and still lists it as out of scope at line 279.

**And the test suite cannot currently catch a regression in this arithmetic.** Every round-trip case in `tests/test_reconstruct.py` builds its image from an integer-valued ramp, and integer-valued data round-trips exactly on geometries where random data does not. Every figure on this page is CPU, for the reason in [Status and maturity](#status-and-maturity).

## Seams: `stitch` against `reconstruct`

`reconstruct` is for patches that came out of `extract` untouched. `stitch` is for patches a model has changed, where uniform averaging leaves the disagreement between neighbouring patches sitting on the grid lines, which is exactly what the eye reads as tiling.

```python
import torch
from patchcraft import extract, per_patch_psnr, stitch

clean = torch.linspace(0, 1, 512).repeat(512, 1).unsqueeze(0)   # a perfectly smooth ramp
patches = extract(clean, patch_size=128, stride=64)             # 7x7 grid, 50% overlap

# A per-patch model that gets each patch's level wrong by up to +/- 0.10.
generator = torch.Generator().manual_seed(7)
edited = patches + (torch.rand(len(patches), 1, 1, 1, generator=generator) * 0.2 - 0.1)

uniform = stitch(edited, clean.shape, stride=64, weight="uniform")
hann = stitch(edited, clean.shape, stride=64, weight="hann")


def seam(image):                       # largest second difference along the middle row
    error = (image - clean)[0, 256]
    return (error[2:] - 2 * error[1:-1] + error[:-2]).abs().max().item()


print(f"uniform: {seam(uniform):.6f}")
print(f"hann:    {seam(hann):.6f}")
print(f"ratio:   {seam(uniform) / seam(hann):.0f}x")

for name, image in (("uniform", uniform), ("hann", hann)):
    psnr = per_patch_psnr(extract(image, patch_size=128, stride=64), edited)
    print(f"{name:>8} against the model's own patches: "
          f"mean {psnr.mean():.2f} dB, min {psnr.min():.2f} dB")
```

```
uniform: 0.018617
hann:    0.000097
ratio:   191x
 uniform against the model's own patches: mean 29.60 dB, min 20.01 dB
    hann against the model's own patches: mean 27.14 dB, min 19.41 dB
```

The image is a smooth ramp, so every step in the output is an artifact of the stitching and of nothing else. Here is the error itself, 16 pixels of one row crossing one patch boundary, at a geometry small enough to print.

```python
import torch
from patchcraft import extract, stitch

torch.set_printoptions(precision=3, sci_mode=False, linewidth=100)   # global, for this block

clean = torch.linspace(0, 1, 128).repeat(128, 1).unsqueeze(0)
patches = extract(clean, patch_size=32, stride=16)
generator = torch.Generator().manual_seed(7)
edited = patches + (torch.rand(len(patches), 1, 1, 1, generator=generator) * 0.2 - 0.1)

uniform = stitch(edited, clean.shape, stride=16, weight="uniform")
hann = stitch(edited, clean.shape, stride=16, weight="hann")

u = (uniform - clean)[0, 64, 56:72]            # row 64, across the boundary at column 64
h = (hann - clean)[0, 64, 56:72]
print("uniform:", u)
print("hann:   ", h)
print("largest jump between neighbours, uniform:", round(u.diff().abs().max().item(), 6))
print("largest jump between neighbours, hann   :", round(h.diff().abs().max().item(), 6))
```

```
uniform: tensor([0.037, 0.037, 0.037, 0.037, 0.037, 0.037, 0.037, 0.037, 0.021, 0.021, 0.021, 0.021, 0.021,
        0.021, 0.021, 0.021])
hann:    tensor([0.020, 0.017, 0.014, 0.012, 0.009, 0.007, 0.006, 0.005, 0.005, 0.005, 0.006, 0.006, 0.007,
        0.008, 0.010, 0.011])
largest jump between neighbours, uniform: 0.016545
largest jump between neighbours, hann   : 0.003106
```

**How to read it.** The uniform row is constant inside each 16 pixel stride and then drops from `0.037` to `0.021` between two adjacent pixels of a smooth ramp, a step of `0.0165` sitting exactly on the grid line. The hann row glides from `0.020` down to `0.005` and back to `0.011`, and its largest move between neighbours is `0.0031`, five times smaller. Same patches, same geometry, the same total disagreement, spread across the overlap instead of piled on one column.

**The ratio is geometry dependent. Never quote it without the geometry.**

```python
import torch
from patchcraft import extract, stitch


def seam(image, clean, row):           # largest second difference along one row
    error = (image - clean)[0, row]
    return (error[2:] - 2 * error[1:-1] + error[:-2]).abs().max().item()


print(f"{'image':>10}{'patch':>7}{'stride':>8}{'uniform':>10}{'gaussian':>10}{'hann':>9}{'ratio':>8}")
for size, patch, stride in ((32, 8, 4), (64, 16, 8), (128, 32, 16), (512, 128, 64)):
    clean = torch.linspace(0, 1, size).repeat(size, 1).unsqueeze(0)
    patches = extract(clean, patch_size=patch, stride=stride)
    generator = torch.Generator().manual_seed(7)
    edited = patches + (torch.rand(len(patches), 1, 1, 1, generator=generator) * 0.2 - 0.1)
    value = {kind: seam(stitch(edited, clean.shape, stride=stride, weight=kind), clean, size // 2)
             for kind in ("uniform", "gaussian", "hann")}
    ratio = f"{value['uniform'] / value['hann']:.1f}x"
    print(f"{f'{size}x{size}':>10}{patch:>7}{stride:>8}{value['uniform']:>10.4f}"
          f"{value['gaussian']:>10.4f}{value['hann']:>9.4f}{ratio:>8}")
```

```
     image  patch  stride   uniform  gaussian     hann   ratio
     32x32      8       4    0.0186    0.0086   0.0114    1.6x
     64x64     16       8    0.0186    0.0078   0.0047    3.9x
   128x128     32      16    0.0186    0.0076   0.0014   13.1x
   512x512    128      64    0.0186    0.0073   0.0001  191.0x
```

At 8 pixel patches hann wins by **1.6x**, which is not worth a paragraph, and gaussian beats hann outright in that row. Uniform's step stays a one-pixel discontinuity at every scale, while hann spreads the same disagreement across an overlap that widens with the patch, so the gap is a function of patch size. If your patches are small, uniform is fine.

**The three windows, and what each costs.**

- `"uniform"` is the default. Every covering patch contributes equally, which is `reconstruct`'s arithmetic and puts the whole disagreement on the grid lines.
- `"hann"` is the strong seam suppressor, and since 0.2.1 it is the interior of a longer symmetric Hann window, `hann_window(n + 2, periodic=False)[1:-1]`, so it is strictly positive on every sample and never zeroes a pixel. The plain symmetric window, which is exactly zero at both endpoints, is the 0.2.0 defect that rewrite fixed.
- `"gaussian"` keeps far more weight at the patch edge than hann does, so it suppresses seams less at the larger patch sizes in the sweep, and it wins at the smallest. [THEORY §2.5](https://github.com/LeoPR/PatchCraft/blob/main/docs/THEORY.md) records the tradeoff as weaker seam suppression than Hann in exchange for a flatter window.

**Hann costs fidelity, and the cost is in the numbers above.** Measured against the model's own patches, uniform keeps 29.60 dB mean and hann keeps 27.14 dB. Hann trades exactness for smoothness on purpose.

**A trap I walked into while measuring this.** A demo whose per-patch error alternates in sign from patch to patch shows uniform winning, because the count map averages the alternation away. The perturbation above is one independent level per patch, which is what a real model's disagreement looks like.

## Planning the geometry before you allocate

`reconstruct` and `stitch` require exact coverage, so the legal geometries are worth knowing before you extract anything. `tilings` enumerates them from the shape alone, and returns a list of `TilingSpec`.

```python
from patchcraft import tilings

for spec in tilings((1, 28, 28)):          # every full-coverage geometry, no image needed
    print(spec)
```

```
TilingSpec(patch_size=(2, 2), stride=(2, 2), dilation=(1, 1), num_patches=(14, 14), total_patches=196, overlap=False)
TilingSpec(patch_size=(4, 4), stride=(4, 4), dilation=(1, 1), num_patches=(7, 7), total_patches=49, overlap=False)
TilingSpec(patch_size=(7, 7), stride=(7, 7), dilation=(1, 1), num_patches=(4, 4), total_patches=16, overlap=False)
TilingSpec(patch_size=(14, 14), stride=(14, 14), dilation=(1, 1), num_patches=(2, 2), total_patches=4, overlap=False)
TilingSpec(patch_size=(28, 28), stride=(28, 28), dilation=(1, 1), num_patches=(1, 1), total_patches=1, overlap=False)
```

**When your image does not tile.** The obvious geometry for an HD frame, 256 pixel patches at 50% overlap, is not legal: seven rows of stride 128 reach pixel 1024 and the frame is 1080 tall, so `reconstruct` refuses it exactly as it refused the geometry in the [first section](#why-not-just-unfold-and-fold). The recovery is a filter over the enumeration rather than padding.

```python
from patchcraft import num_patches, tilings

shape = (3, 1080, 1920)                       # an HD frame

grid = num_patches(shape, patch_size=256, stride=128)
covered = tuple((n - 1) * 128 + 256 for n in grid)
print("patch 256 stride 128:", grid, "patches covering", covered, "of", shape[1:])

legal = tilings(shape, allow_overlap=True)
half = [s for s in legal
        if s.stride[0] * 2 == s.patch_size[0] and s.stride[1] * 2 == s.patch_size[1]]
print(len(legal), "legal geometries,", len(half), "of them at 50% overlap")
print("the largest:", max(half, key=lambda s: s.patch_size[0]))
```

```
patch 256 stride 128: (7, 14) patches covering (1024, 1920) of (1080, 1920)
3694 legal geometries, 16 of them at 50% overlap
the largest: TilingSpec(patch_size=(240, 240), stride=(120, 120), dilation=(1, 1), num_patches=(8, 15), total_patches=120, overlap=True)
```

*240 at stride 120 is the largest 50% overlap tiling that covers a 1080p frame exactly, and because its stride is half its patch size its count map holds only powers of two, so it also round-trips bit for bit. The alternatives to picking it are cropping the frame to a covering extent or resizing it, both of which are your call and not this library's.*

Enumeration is arithmetic on the shape. Nothing is read, nothing is allocated, no image is touched. The `extract` it describes is where the memory goes.

```python
import torch
from patchcraft import extract, num_patches, reconstruct


def mib(tensor):
    return f"{tensor.element_size() * tensor.nelement() / 2**20:.0f} MiB"


shape, patch, stride = (3, 1080, 1920), 240, 120
print(num_patches(shape, patch_size=patch, stride=stride), "patch grid, from arithmetic alone")

torch.manual_seed(0)
image = torch.rand(*shape)
patches = extract(image, patch_size=patch, stride=stride)
print(tuple(patches.shape), mib(patches), "of patches from", mib(image), "of image")
print("round trip exact:", torch.equal(reconstruct(patches, image.shape, stride=stride), image))
```

```
(8, 15) patch grid, from arithmetic alone
(120, 3, 240, 240) 79 MiB of patches from 24 MiB of image
round trip exact: True
```

*Real output: `num_patches` answers with a tuple what `extract` needs 79 MiB to discover. At 50% overlap the patch stack is more than three times the image it came from, which is the number to know before you send it to a GPU.*

One known wart, since you will meet it: where the grid collapses to a single patch, `tilings(..., allow_overlap=True)` still emits one spec per stride value. `tilings((1, 28, 28), allow_overlap=True)` returns 100 specs, of which 28 are the whole-image tiling and 27 of those are labelled `overlap=True`. A single patch overlaps nothing. The arithmetic is right and the label is not useful.

## LR and HR pairs

`pair` extracts the same image region at two resolutions, with metadata that locates every patch in LR pixel coordinates. `paired_tilings` enumerates the geometries that are legal on both sides at once, and `scale_factor` reports the integer ratio or `None` when the two shapes are not an integer multiple apart.

```python
import torch
from patchcraft import pair, paired_tilings, scale_factor

torch.manual_seed(0)
lr, hr = torch.rand(1, 14, 14), torch.rand(1, 28, 28)
assert scale_factor(lr.shape, hr.shape) == 2            # None when the ratio is not an integer

for spec in paired_tilings(lr.shape, hr.shape):
    print(spec.lr.patch_size, "->", spec.hr.patch_size, "grid", spec.lr.num_patches)

pairs = pair(lr, hr, lr_patch_size=7, scale_factor=2, stride=7)
assert len(pairs) == 4
print(tuple(pairs.lr_patches.shape), tuple(pairs.hr_patches.shape))
print(pairs.metas[1])
```

```
(2, 2) -> (4, 4) grid (7, 7)
(7, 7) -> (14, 14) grid (2, 2)
(14, 14) -> (28, 28) grid (1, 1)
(4, 1, 7, 7) (4, 1, 14, 14)
PatchMeta(patch_index=1, row=0, col=7, lr_patch_size=(7, 7), hr_patch_size=(14, 14), image_id=None)
```

*`row` and `col` already have the stride applied: patch 1 sits at LR column 7, which is HR column 14. Multiplying them by the stride again lands on the wrong patch, which is the most common mistake with this structure and the one an external reviewer hit on first contact with this API.*

## The public API, 19 symbols

```python
import patchcraft

assert patchcraft.__version__ == "0.2.1"
assert len(patchcraft.__all__) == 19
assert all(hasattr(patchcraft, name) for name in patchcraft.__all__)
print(patchcraft.__all__)
```

```
['Cache', 'PairedTilingSpec', 'PatchMeta', 'PatchPair', 'Patchify', 'TilingSpec', 'WeightKind', 'extract', 'num_patches', 'pair', 'paired_tilings', 'patch_metrics', 'per_patch_mse', 'per_patch_psnr', 'reconstruct', 'resize', 'scale_factor', 'stitch', 'tilings']
```

| Symbol | Takes | Returns | What it allocates |
|---|---|---|---|
| `extract` | `image`, `patch_size`, `stride`, `dilation` | `(L, C, ph, pw)` | the whole patch stack, `L * C * ph * pw` values |
| `Patchify` | the same geometry, at construction | a callable returning `(L, C, ph, pw)` | nothing beyond the geometry, no cache and no buffer |
| `reconstruct` | `patches`, `image_shape`, `stride`, `dilation` | `(C, H, W)` | one image plus one count map |
| `stitch` | the same, plus keyword-only `weight=`, default `"uniform"` | `(C, H, W)` | one image, one weight map, one `(ph, pw)` kernel |
| `WeightKind` | nothing, it is a `Literal` type alias | `"uniform"`, `"hann"` or `"gaussian"` | nothing |
| `pair` | `lr_image`, `hr_image`, `lr_patch_size`, `scale_factor`, `stride` | `PatchPair` | both patch stacks plus `L` metadata records |
| `PatchPair`, `PatchMeta` | frozen dataclasses | patches and grid coordinates | metadata stays on CPU, always |
| `num_patches` | `image_shape`, `patch_size`, `stride` | `(num_h, num_w)` | nothing, it is arithmetic |
| `tilings` | `image_shape`, keyword-only `allow_overlap=` | `list[TilingSpec]` | one spec per legal geometry, no pixels |
| `scale_factor` | LR and HR shapes | `int`, or `None` when the ratio is not an integer | nothing |
| `paired_tilings` | LR and HR shapes | `list[PairedTilingSpec]` | one spec per legal pair, no pixels |
| `patch_metrics` | two same-shape tensors | `{mae, mse, max_abs, psnr_db}` | accumulates in float64, returns Python floats |
| `per_patch_mse`, `per_patch_psnr` | two `(L, C, h, w)` stacks | `(L,)` in float64 | one value per patch |
| `resize` | one tensor or `PIL.Image`, `target_size`, `backend` | the type it received | one resized image |
| `Cache` | `root`, `namespace`, `version` | a content-addressed store | files on disk, zstd payloads when installed |

```python
import tempfile

import torch
from patchcraft import Cache, extract, patch_metrics, per_patch_psnr, resize

torch.manual_seed(0)
hr = torch.rand(1, 64, 64)
lr = resize(hr, (32, 32), backend="torch")          # PIL or torch backend, type preserved
assert tuple(lr.shape) == (1, 32, 32) and lr.dtype == hr.dtype

reference = extract(hr, patch_size=16, stride=16)
noisy = reference + 0.01
print({name: round(value, 6) for name, value in patch_metrics(reference, noisy).items()})
print(tuple(per_patch_psnr(reference, noisy).shape))

with tempfile.TemporaryDirectory() as directory:
    cache = Cache(root=directory, namespace="demo", version=1)
    key = cache.key_for("hr.png", (16, 16), 16)     # SHA-256 over namespace, version, parts
    assert cache.get(key) is None
    cache.put(key, b"payload")
    assert cache.get(key) == b"payload"
```

```
{'mae': 0.01, 'mse': 0.0001, 'max_abs': 0.01, 'psnr_db': 40.000006}
(16,)
```

The per-symbol contract, including every condition each function rejects, is [docs/THEORY.md](https://github.com/LeoPR/PatchCraft/blob/main/docs/THEORY.md) §9.

## Status and maturity

**0.2.1, pre-1.0, and no external project has consumed it yet.** That second half is the honest headline.

What is verified: 346 tests pass and 5 GPU tests are deselected in the current local full run of `pytest -m "not gpu"`, in about seven seconds on this machine; run `pytest` for the number in your environment. CI runs the same suite plus `ruff check` and `mypy --strict` on {ubuntu-latest, windows-latest} x {Python 3.12, 3.13}, and all four cells are green. Releases reach PyPI through Trusted Publishing on a tag push. The package is typed and ships `py.typed`.

Five things this project does not claim.

1. **No consumer.** The gate in [docs/ROADMAP.md](https://github.com/LeoPR/PatchCraft/blob/main/docs/ROADMAP.md) is that a real project consumes `patchcraft` v0.2 as published, and that has not happened. The one external review that did happen found a defect in the `pair` docstring on first contact, which is the sample size this API has been tested against by someone other than its author.
2. **No CUDA, anywhere.** The torch build here is CPU only and both workflows run `pytest -m "not gpu"`, so the CUDA paths of `extract`, `reconstruct`, `stitch` and `resize` have never executed. Device preservation is implemented and unmeasured on device, and every measurement on this page is CPU.
3. **The no-zstandard cache path runs in no environment.** `Cache` falls back to uncompressed payloads when `zstandard` is absent, and every configuration here and in CI installs the extra. A plain `pip install patchcraft` takes precisely the branch nothing exercises.
4. **Nothing executes the examples on this page.** They were run by hand against 0.2.1 and pasted verbatim, and no test in the suite runs them. A test that executes every fenced block and checks the figures quoted in prose is the next piece of work on this file. [docs/USAGE.md](https://github.com/LeoPR/PatchCraft/blob/main/docs/USAGE.md) is in the same position and is still captured against 0.2.0.
5. **Pre-1.0 means output values can change in a minor release.** 0.2.1 rewrote the hann window, so `stitch(..., weight="hann")` returns different values than 0.2.0 for every geometry. The round-trip contract did not change. [CHANGELOG.md](https://github.com/LeoPR/PatchCraft/blob/main/CHANGELOG.md) records each change together with the measurement that motivated it.

## Repository layout

The wheel is `src/patchcraft/` and nothing else. Everything beside it exists to prove that code works.

```
PatchCraft/
├── src/patchcraft/         the package, one image at a time
│   ├── extract.py          extract, Patchify (F.unfold)
│   ├── reconstruct.py      the inverse (F.fold plus count map)
│   ├── stitch.py           weighted reassembly for modified patches
│   ├── geometry.py         num_patches, tilings, scale_factor, paired_tilings
│   ├── pair.py             LR/HR pairing, PatchPair, PatchMeta
│   ├── metrics.py          patch_metrics, per_patch_mse, per_patch_psnr
│   ├── resize.py           PIL and torch backends
│   ├── cache.py            content-addressed disk cache
│   └── py.typed
├── tests/                  346 tests, the contract from THEORY §9
├── docs/
│   ├── USAGE.md            walkthrough of every public symbol
│   ├── SCOPE.md            what belongs here and what belongs to your pipeline
│   ├── THEORY.md           §0 binding scope, §9 per-function contract
│   ├── ROADMAP.md          milestones and the consumer gate
│   ├── AUXILIARY.md        test fixtures and off-tree conventions
│   ├── ADR/                architecture decision records
│   └── STUDIES/            measured investigations that feed the ADRs
├── lab/                    ephemeral experiments, results kept off-tree
├── .github/workflows/      test.yml (matrix CI), release.yml (Trusted Publishing)
├── pyproject.toml
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

## Contributing and running the tests

```
git clone https://github.com/LeoPR/PatchCraft.git
cd PatchCraft
pip install -e ".[dev,cache]"

pytest -m "not gpu"
ruff check src tests
mypy --strict src
```

Those three commands are what CI runs. New behaviour arrives as a hypothesis measured in `lab/`, becomes a test in `tests/` when the measurement holds, and is recorded in an ADR when it changes a contract. [CONTRIBUTING.md](https://github.com/LeoPR/PatchCraft/blob/main/CONTRIBUTING.md) carries the full layout, the validation conventions and the release procedure. Issues and pull requests are welcome at [github.com/LeoPR/PatchCraft/issues](https://github.com/LeoPR/PatchCraft/issues). A report is most useful with the version, the platform, the dtype and the geometry, because those four facts decide almost everything in this library.

## How to cite

There is no DOI and no `CITATION.cff` in the repository yet, so GitHub shows no "Cite this repository" button. Until there is one:

```bibtex
@software{souza_patchcraft_2026,
  author  = {Souza, Leonardo Marques de},
  title   = {PatchCraft: image patch extraction, reconstruction, pairing
             and seam-aware stitching},
  version = {0.2.1},
  year    = {2026},
  url     = {https://github.com/LeoPR/PatchCraft}
}
```

## Where to read next

| If you want | Open |
|---|---|
| A walkthrough of every public symbol with REPL output | [docs/USAGE.md](https://github.com/LeoPR/PatchCraft/blob/main/docs/USAGE.md), captured against 0.2.0 |
| The line between this library and your pipeline | [docs/SCOPE.md](https://github.com/LeoPR/PatchCraft/blob/main/docs/SCOPE.md) |
| The math, the design decisions, the per-function contract | [docs/THEORY.md](https://github.com/LeoPR/PatchCraft/blob/main/docs/THEORY.md) |
| Why the API looks like this | [docs/ADR/](https://github.com/LeoPR/PatchCraft/tree/main/docs/ADR) |
| Where the exactness boundary is being written down | [docs/ADR/0003-reversibility-classes.md](https://github.com/LeoPR/PatchCraft/blob/main/docs/ADR/0003-reversibility-classes.md), Proposed |
| What changed in each release, with the measurements | [CHANGELOG.md](https://github.com/LeoPR/PatchCraft/blob/main/CHANGELOG.md) |
| Milestones and the consumer gate | [docs/ROADMAP.md](https://github.com/LeoPR/PatchCraft/blob/main/docs/ROADMAP.md) |
| To clone, test and contribute | [CONTRIBUTING.md](https://github.com/LeoPR/PatchCraft/blob/main/CONTRIBUTING.md) |

## License

MIT. See [LICENSE](https://github.com/LeoPR/PatchCraft/blob/main/LICENSE). Copyright 2026 Leonardo Marques de Souza.