<!-- l10n: doc_id=patchcraft-guide · lang=en · canonical -->

# PatchCraft: the guide

This is the manual. [The README](../README.md) is the call page, and it answers whether PatchCraft is worth your time at all; this page answers how to use it well, and it shows the measurements behind every claim the call page makes without proof.

You do not have to read it from the top. Each section stands on its own, so jump into the one that matches the problem in front of you, and follow the links out to [THEORY.md](THEORY.md) when you want the contract rather than the demonstration.

**Provenance.** Every fenced output block on this page is verbatim printed output of the code shown directly above it, run against `patchcraft` 0.2.1 on CPU, with Python 3.13.13 and torch 2.13.0+cpu. Figures quoted in prose are read off those blocks, or are arithmetic on them. Two families of number name their own source instead: the test-suite counts in [section 8](#8-what-this-project-does-not-claim), and the file and line references, which point at the repository as of 0.2.1.

## Contents

1. [Why not `unfold` and `fold` directly](#1-why-not-unfold-and-fold-directly)
2. [The patch stack and the count map](#2-the-patch-stack-and-the-count-map)
3. [Dtypes and files](#3-dtypes-and-files)
4. [When the round trip is bit for bit](#4-when-the-round-trip-is-bit-for-bit)
5. [Seams: `reconstruct` against `stitch`, measured](#5-seams-reconstruct-against-stitch-measured)
6. [Planning the geometry before you allocate](#6-planning-the-geometry-before-you-allocate)
7. [The 19 symbols and what each allocates](#7-the-19-symbols-and-what-each-allocates)
8. [What this project does not claim](#8-what-this-project-does-not-claim)
9. [Install details and citation](#9-install-details-and-citation)

## 1. Why not `unfold` and `fold` directly

Because three defects wait there, all three silent, and you meet the first two within the first hour of writing the code yourself.

### The reshape after `unfold` scrambles the pixels

`F.unfold` returns a tensor of shape `(1, C*ph*pw, L)`. The intuitive next step is to reshape it into `(L, C, ph, pw)`, which is the shape you actually want. That reshape produces the right shape, raises nothing, and hands back the wrong pixels, because the column layout of `unfold` puts the patch index last and the intuitive view reads it first.

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

Both tensors have shape `(4, 1, 4, 4)`, so the shape tells you nothing. Only the values tell you which one is your image, and if the patch goes straight into a model you will never look at the values.

### A stride that does not cover the image leaves pixels at zero

With `patch=32` and `stride=20` on a 128 pixel axis, the grid is 5 by 5 and the last patch ends at pixel 112. A hand-rolled `fold` then divides by a count map that is zero on the remaining band, and it returns a partly black image without raising anything.

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

That is 23 percent of the image coming back zeroed with no exception raised. `reconstruct` refuses the geometry instead, and its message names the covered extent against the requested one, so the number you need in order to pick a legal geometry is already in the error.

### `extract` does not enforce the same rule, and that asymmetry is real

The coverage guard lives on the way back and not on the way out. `extract` will happily tile a 130 pixel axis with 32 pixel patches, discard the last two rows and columns, and say nothing about it.

```python
import torch
from patchcraft import extract, reconstruct

torch.manual_seed(0)
image = torch.rand(3, 130, 130)                       # 130 is not a multiple of 32
patches = extract(image, patch_size=32, stride=32)    # no error, and no warning
print(tuple(patches.shape), "covers", 4 * 32, "of 130 pixels on each axis")

try:
    reconstruct(patches, image.shape, stride=32)
except ValueError as error:
    print("reconstruct:", error)
```

```
(16, 3, 32, 32) covers 128 of 130 pixels on each axis
reconstruct: patch grid leaves pixels uncovered (partial coverage forbidden): image_shape=torch.Size([3, 130, 130]), patch_size=(32, 32), stride=(32, 32) covers (128, 128) of (130, 130). Choose a geometry with exact coverage (see patchcraft.tilings).
```

If you extract, run a model and reconstruct, the guard catches you. If you extract, run a model and never reconstruct, which is what a patch dataset does, then you lose the border in silence.

`Patchify` behaves the same way, because it calls the same code. Until this is decided one way or the other, call `num_patches` or `tilings` before you extract, and [section 6](#6-planning-the-geometry-before-you-allocate) shows how. The asymmetry is recorded as blocker B6 in [FOCO-1.0.md](FOCO-1.0.md).

### The whole job, written both ways

Here is the work the library actually replaces: tiling one image with overlap, running a per-patch model, and blending the result back.

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

Seventeen non-blank lines against three, and the two results are the same tensor to the last bit rather than merely close.

The by-hand version above is also the correct one, since it already has the coverage check, the right permutation and the strictly positive window. Getting to that version is the work.

### Two honest caveats

PatchCraft computes nothing that torch cannot. It is the same `unfold`, the same `fold` and the same count map underneath, and it is not faster. What you are buying is that the geometry is checked before the arithmetic runs, in one place, with tests around it.

The second caveat is that the coverage defect shown above shipped inside PatchCraft itself. Version 0.2.0 validated the patch count and never the coverage, so it returned partly black images until 0.2.1 added the guard, which [CHANGELOG.md](../CHANGELOG.md) records with the measurements that found it. The argument for the library is that the guard is written once and regression tested, and not that it was ever obvious.

## 2. The patch stack and the count map

`extract` turns one `(C, H, W)` image into one `(L, C, ph, pw)` stack. The order is row-major, there is no padding, and there is no batch axis.

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

More generally, with a grid of `nw` patches per row, patch `k` starts at row `k // nw * stride_h` and column `k % nw * stride_w`. You can read that order straight off the corner values.

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

### The count map

The one object to carry into every section below is the **count map**, which says how many patches cover each pixel. `reconstruct` builds it internally and divides the folded sum by it, while `stitch` divides by the folded window instead.

It is not a public symbol, so the block below rebuilds it with `F.fold` in order to make the arithmetic visible, taking the grid size from `num_patches`.

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

Corners are covered once, edges twice and the interior four times. When `stride == patch_size` the map is all ones and the division is a no-op, which is why that geometry is the easy case.

Which values appear in this map is what decides whether the round trip comes back bit for bit, and specifically whether every one of them is a power of two. That is the subject of [section 4](#4-when-the-round-trip-is-bit-for-bit). The math behind the fold and the division is in [THEORY.md](THEORY.md) §2.

## 3. Dtypes and files

### Float tensors only

`extract` hands the tensor straight to `F.unfold`, which has no integer kernel, so an 8-bit image fails with torch's own message rather than with one of ours. Convert on the way in, which costs you one division.

```python
import torch
from patchcraft import extract, reconstruct

eight_bit = (torch.rand(1, 64, 64) * 255).to(torch.uint8)   # what read_image or PIL gives you
try:
    extract(eight_bit, patch_size=16, stride=16)
except Exception as error:
    print("extract:", type(error).__name__ + ":", error)

patches = extract(eight_bit.float() / 255, patch_size=16, stride=16)
print(tuple(patches.shape), patches.dtype)

try:
    reconstruct(patches.to(torch.uint8), (1, 64, 64), stride=16)
except Exception as error:
    print("reconstruct:", type(error).__name__ + ":", error)
```

```
extract: NotImplementedError: "im2col_out_cpu" not implemented for 'Byte'
(16, 1, 16, 16) torch.float32
reconstruct: ValueError: reconstruct requires floating-point patches, got dtype=torch.uint8. F.fold is not implemented for integer dtypes; convert with patches.float() first.
```

The two messages are not equally good, and the asymmetry is a rough edge rather than a design. Since 0.2.1 `reconstruct` and `stitch` carry a dtype guard that raises a framed `ValueError` naming the conversion, while `extract` still lets torch's raw `NotImplementedError` through.

The accepted dtypes are `float16`, `float32`, `float64` and `bfloat16`, and they are contract text in [THEORY.md](THEORY.md) §9.1 and §9.2.

### End to end from a file

Nothing here is library surface. Reading a PNG, converting through numpy and writing the result back is pipeline work, and it is shown because it is the shape of the first script most people write.

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

The block writes `example.png` and `roundtrip.png` into the working directory, and the two files hold the same pixels. It uses only `pillow` and `numpy`, both of which PatchCraft already depends on, so nothing extra needs installing.

The round trip is exact here for a reason worth naming, because it does not generalise. A 96 pixel axis with `patch=32` and `stride=16` is a covering geometry whose count map holds only 1, 2 and 4, and every one of those is a power of two.

## 4. When the round trip is bit for bit

`reconstruct` folds the patches and divides by the count map. Whether that returns the original tensor bit for bit is decided by the values in that map, and not by the dtype you chose.

Here is the sweep that shows it, across four float dtypes and three geometries.

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

### The rule

A pixel covered by `k` patches is summed `k` times and then divided by `k`. In binary floating point that division is exact when `k` is a power of two, and it rounds otherwise.

So the round trip in `float32` and `float64` is `torch.equal` exact **when every value in the count map is a power of two**. Two geometries guarantee that without any checking at all: `stride == patch_size` on both axes, where the map is all ones, and `stride == patch_size / 2`, where the map holds only 1, 2 and 4.

### The maximum of the map is not the rule

This is worth showing, because it is the shape the mistake usually takes. Reading the maximum of the count map and finding a power of two feels like enough, and it is not.

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

The maximum is 4, the geometry is legal and fully covering, and the round trip still misses, because a 3 sits in the map. Every value has to pass the test, and not just the largest one.

### Three honest details

**Outside the predicate, exactness is a property of the data rather than of the geometry.** The guarantee runs one way only: all powers of two means exact, and anything else means not guaranteed. Some inputs do come back exact on a mixed map by luck of rounding, so a geometry that survived one image tells you nothing about the next. What does hold either way is the size of the miss, which was `2.384e-07` in float32 and `4.441e-16` in float64 on the geometries above, and that is one ULP territory.

**float64 is not a safe harbour.** The deciding axis is the count map and not the dtype, so float64 misses the round trip at exactly the geometry where float32 misses it. Reaching for a wider float buys you a smaller error, and it never buys you exactness.

**Half precision comes back exact here for a reason worth knowing, and not because of a stronger guarantee.** Version 0.2.1 accumulates `float16` and `bfloat16` in a float32 buffer and rounds once on return, and that final rounding is far coarser than the float32 error, so those rows land back on the value they started from. The promotion exists for a different reason entirely, which is that the folded sum overflows the fp16 finite range before the division ever happens. [THEORY.md](THEORY.md) §9.2 records the measurement that forced it, where a constant fp16 image at value 10000.0 came back as `inf` in 144 of 256 pixels before the fix.

### Where this rule is written down

[ADR 0003](ADR/0003-reversibility-classes.md) is where the exactness boundary is being turned into contract, and it is still **Proposed**, so the wording has not landed across the project yet. Several docstrings and documents still state the overlap round trip as exact with no condition on the count map, and the audit that lists each of them is blocker B1 in [FOCO-1.0.md](FOCO-1.0.md).

Treat this section as the measured truth and treat those other statements as pending corrections, in that order, until ADR 0003 is accepted.

## 5. Seams: `reconstruct` against `stitch`, measured

`reconstruct` is for patches that came out of `extract` untouched, where every patch covering a pixel agrees about that pixel. `stitch` is for patches a model has changed, where they no longer agree, and uniform averaging leaves the whole disagreement sitting on the grid lines, which is exactly what the eye reads as tiling.

The demonstration below uses a perfectly smooth ramp, so every step in the output is an artifact of the stitching and of nothing else.

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

### The error itself, one row across one boundary

The ratio is a summary, so here is the thing being summarised: sixteen pixels of one row crossing one patch boundary, at a geometry small enough to print.

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

The uniform row is constant inside each 16 pixel stride and then drops from `0.037` to `0.021` between two adjacent pixels of a smooth ramp. That is a step of `0.0165` sitting exactly on the grid line, and a step on a smooth ramp is what a seam is.

The hann row glides from `0.020` down to `0.005` and back up to `0.011`, and its largest move between neighbours is `0.0031`, roughly five times smaller. Same patches, same geometry and the same total disagreement, spread across the overlap instead of piled onto one column.

### The ratio is geometry dependent, so never quote it without the geometry

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

The famous 191x is true at 512 pixel images with 128 pixel patches, and it is true nowhere else. At 8 pixel patches hann wins by 1.6x, which is not worth a paragraph, and gaussian beats hann outright in that row.

The reason is visible in the uniform column, which does not move at all. Uniform's step stays a one-pixel discontinuity at every scale, while hann spreads the same disagreement across an overlap that widens with the patch, so the gap between them is a function of patch size. If your patches are small, uniform is fine, and if they are large, hann is worth its cost.

### The three windows, and what each costs

- **`"uniform"`** is the default. Every covering patch contributes equally, which is exactly `reconstruct`'s arithmetic, and it puts the whole disagreement on the grid lines.
- **`"hann"`** is the strong seam suppressor and the cheapest to compute. Since 0.2.1 it is the interior of a longer symmetric Hann window, `hann_window(n + 2, periodic=False)[1:-1]`, so it is strictly positive on every sample and never zeroes a pixel. The plain symmetric window, which is exactly zero at both endpoints, was the largest defect the 0.2.0 audit found, and [THEORY.md](THEORY.md) §2.5 records what it did to real images.
- **`"gaussian"`** keeps far more weight at the patch edge than hann does, so it suppresses seams less at the larger patch sizes in the sweep, and it wins at the smallest one. THEORY §2.5 states the tradeoff as weaker seam suppression than Hann in exchange for a flatter window.

**Hann costs fidelity, and the cost is already in the numbers above.** Measured against the model's own patches, uniform keeps 29.60 dB mean and hann keeps 27.14 dB. Hann is trading exactness for smoothness on purpose, so if what you want back is the model's output rather than a pleasant image, uniform is the honest choice.

### A trap that made an earlier version of this demo lie

The result flips if the per-patch error alternates in sign from patch to patch, because then the count map averages the alternation away and uniform comes out perfect.

```python
import torch
from patchcraft import extract, stitch

clean = torch.linspace(0, 1, 512).repeat(512, 1).unsqueeze(0)
patches = extract(clean, patch_size=128, stride=64)


def seam(image):                       # largest second difference along the middle row
    error = (image - clean)[0, 256]
    return (error[2:] - 2 * error[1:-1] + error[:-2]).abs().max().item()


generator = torch.Generator().manual_seed(7)
independent = torch.rand(len(patches), 1, 1, 1, generator=generator) * 0.2 - 0.1
alternating = torch.tensor([0.1 if k % 2 == 0 else -0.1 for k in range(len(patches))])
alternating = alternating.reshape(-1, 1, 1, 1)

for name, offset in (("independent", independent), ("alternating", alternating)):
    u = seam(stitch(patches + offset, clean.shape, stride=64, weight="uniform"))
    h = seam(stitch(patches + offset, clean.shape, stride=64, weight="hann"))
    winner = "hann" if h < u else "uniform"
    print(f"{name:>12}: uniform {u:.6f}, hann {h:.6f}, {winner} wins")
```

```
 independent: uniform 0.018617, hann 0.000097, hann wins
 alternating: uniform 0.000000, hann 0.000355, uniform wins
```

Read the two rows carefully, because the conclusion reverses between them. On an alternating error uniform scores a clean `0.000000` and looks unbeatable, which is an artifact of the demo rather than a property of uniform.

One independent level per patch, which is the first row, is what a real model's disagreement actually looks like, and every other measurement in this section uses it. So if you build your own seam benchmark, check what your synthetic error does across neighbouring patches before you trust the number it gives you.

## 6. Planning the geometry before you allocate

`reconstruct` and `stitch` both require exact coverage, so the legal geometries are worth knowing before you extract anything at all. `tilings` enumerates them from the shape alone and returns a list of `TilingSpec`.

```python
from patchcraft import tilings

for spec in tilings((1, 28, 28)):          # every full-coverage geometry, no image needed
    print(spec)

print(len(tilings((1, 28, 28))), "exact tilings,",
      len(tilings((1, 28, 28), allow_overlap=True)), "once overlap is allowed")
```

```
TilingSpec(patch_size=(2, 2), stride=(2, 2), dilation=(1, 1), num_patches=(14, 14), total_patches=196, overlap=False)
TilingSpec(patch_size=(4, 4), stride=(4, 4), dilation=(1, 1), num_patches=(7, 7), total_patches=49, overlap=False)
TilingSpec(patch_size=(7, 7), stride=(7, 7), dilation=(1, 1), num_patches=(4, 4), total_patches=16, overlap=False)
TilingSpec(patch_size=(14, 14), stride=(14, 14), dilation=(1, 1), num_patches=(2, 2), total_patches=4, overlap=False)
TilingSpec(patch_size=(28, 28), stride=(28, 28), dilation=(1, 1), num_patches=(1, 1), total_patches=1, overlap=False)
5 exact tilings, 100 once overlap is allowed
```

Nothing is read and nothing is allocated here. Enumeration is arithmetic on the shape, so it is cheap enough to run before you have decided anything.

### When your image does not tile

The obvious geometry for an HD frame, which is 256 pixel patches at 50 percent overlap, is not legal. Seven rows of stride 128 reach pixel 1024 and the frame is 1080 tall, so `reconstruct` refuses it exactly as it refused the geometry in [section 1](#1-why-not-unfold-and-fold-directly).

The recovery is a filter over the enumeration rather than padding, because padding would synthesize pixels you never had.

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

240 at stride 120 is the largest 50 percent overlap tiling that covers a 1080p frame exactly. Because its stride is half its patch size, its count map holds only powers of two, so by the rule in [section 4](#4-when-the-round-trip-is-bit-for-bit) it also round-trips bit for bit.

The alternatives to picking a legal geometry are cropping the frame to a covering extent or resizing it. Both of those are your call and not this library's, since both of them change the pixels the caller asked about.

### The memory arithmetic

Enumeration costs nothing, and the `extract` it describes is where the memory goes. That difference is worth measuring once, before you send anything to a GPU.

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

`num_patches` answers with a tuple what `extract` needs 79 MiB to discover. At 50 percent overlap the patch stack is more than three times the size of the image it came from, and that multiple grows as the stride shrinks, so the plan is worth making before the allocation.

### One known wart in the enumeration

Where the grid collapses to a single patch, `tilings(..., allow_overlap=True)` still emits one spec per stride value, and it labels almost all of them as overlapping.

```python
from patchcraft import tilings

whole = [s for s in tilings((1, 28, 28), allow_overlap=True) if s.total_patches == 1]
print(len(whole), "specs describe the whole-image tiling,",
      sum(s.overlap for s in whole), "of them labelled overlap=True")
print(whole[0])
```

```
28 specs describe the whole-image tiling, 27 of them labelled overlap=True
TilingSpec(patch_size=(28, 28), stride=(28, 28), dilation=(1, 1), num_patches=(1, 1), total_patches=1, overlap=False)
```

A single patch overlaps nothing, so the arithmetic is right and the label is not useful. The cause is the `nh > 1 or nw > 1` guard in `geometry.py`, it is recorded as blocker B6 in [FOCO-1.0.md](FOCO-1.0.md), and that is where the decision for 1.0 will be made. Filter on `total_patches > 1` if the duplicates get in your way.

## 7. The 19 symbols and what each allocates

The public surface is 19 names, and `__all__` is what fixes it.

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

The table below is the one place that records what each call allocates, which is the question that decides whether a geometry fits in memory. For a walkthrough of any single symbol, open [USAGE.md](USAGE.md), and for the conditions each function rejects, open [THEORY.md](THEORY.md) §9.

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

Four of those symbols sit off to the side of the main loop, and together they fit in one screen.

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

The LR and HR pairing symbols, which are `pair`, `paired_tilings` and `scale_factor`, are walked through in [USAGE.md](USAGE.md) §7 and §11. One warning about them belongs here too, because an external reviewer hit it on first contact with the API: in `PatchMeta`, `row` and `col` already have the stride applied, so multiplying them by the stride again lands you on the wrong patch.

## 8. What this project does not claim

**Version 0.2.1 is pre-1.0, and no external project has consumed it yet.** That second half is the honest headline, and everything below is detail underneath it.

What is verified is this. The full local run of `pytest -m "not gpu"` passes 346 tests and deselects 5 GPU tests, in under seven seconds on this machine, so run `pytest` yourself for the number in your environment. CI runs the same suite plus `ruff check` and `mypy --strict` on Ubuntu and Windows against Python 3.12 and 3.13, and all four cells are green. Releases reach PyPI through Trusted Publishing on a tag push. The package is typed and it ships `py.typed`.

Five things this project does not claim.

**1. No consumer.** The gate in [ROADMAP.md](ROADMAP.md) is that a real project consumes `patchcraft` v0.2 as published, and that has not happened yet. The one external review that did happen found a defect in the `pair` docstring on first contact, which is the sample size this API has been tested against by somebody other than its author.

**2. No CUDA, anywhere.** The torch build here is CPU only, and both workflows run `pytest -m "not gpu"`, so the CUDA paths of `extract`, `reconstruct`, `stitch` and `resize` have never executed. Device preservation is implemented and unmeasured on device, and every measurement on this page is CPU.

**3. The no-zstandard cache path runs in no environment.** `Cache` falls back to uncompressed payloads when `zstandard` is absent, and every configuration here and in CI installs the extra. So a plain `pip install patchcraft` takes precisely the branch that nothing exercises.

**4. Nothing executes the examples on this page.** They were run by hand against 0.2.1 and pasted verbatim, and no test in the suite runs them. A test that executes every fenced block and checks the figures quoted in prose is the next piece of work on this file, and [USAGE.md](USAGE.md) is in the same position while still being captured against 0.2.0.

**5. Pre-1.0 means output values can change in a minor release.** Version 0.2.1 rewrote the hann window, so `stitch(..., weight="hann")` returns different values than 0.2.0 does for every geometry. The round-trip contract did not change. [CHANGELOG.md](../CHANGELOG.md) records each change together with the measurement that motivated it.

The backlog that closes these, along with the wording corrections named in [section 4](#4-when-the-round-trip-is-bit-for-bit) and the label wart in [section 6](#6-planning-the-geometry-before-you-allocate), is [FOCO-1.0.md](FOCO-1.0.md).

## 9. Install details and citation

One line installs the package, and the distribution name and the import name are both `patchcraft`.

```
pip install patchcraft
```

Runtime dependencies are `torch>=2.6`, `numpy>=1.26` and `pillow>=10`. The cache extra adds `zstandard`, which compresses `Cache` payloads.

```
pip install "patchcraft[cache]"     # adds zstandard, compresses Cache payloads
```

From source, for development:

```
git clone https://github.com/LeoPR/PatchCraft.git
cd PatchCraft
pip install -e ".[dev,cache]"
```

**Python versions.** CI tests 3.12 and 3.13 on Ubuntu and Windows, which is also what the classifiers advertise. `requires-python` is `>=3.12` with no ceiling, so pip will install this on 3.14 as well, where nothing has been measured.

**GPU.** Install a matching torch wheel first, following [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/). Read [section 8](#8-what-this-project-does-not-claim) before you do, because no CUDA path in this library has ever been executed.

**Contributing.** The three commands CI runs are `pytest -m "not gpu"`, `ruff check src tests` and `mypy --strict src`. New behaviour arrives as a hypothesis measured in `lab/`, becomes a test in `tests/` when the measurement holds, and is recorded in an ADR when it changes a contract. [CONTRIBUTING.md](../CONTRIBUTING.md) carries the full layout, the validation conventions and the release procedure.

**Citation.** There is no DOI and no `CITATION.cff` in the repository yet, so GitHub shows no "Cite this repository" button. Until there is one, this entry is the reference:

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
