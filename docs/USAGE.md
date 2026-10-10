# PatchCraft: usage walkthrough

> **This page is executed, not transcribed.** Every `>>>` below runs in the
> test suite as a doctest, and the output shown is compared against the output
> produced. A stale line here fails CI rather than misleading a reader.

Run it yourself against your install:

```
pytest --doctest-glob='*.md' docs/USAGE.md
```

This is the tour of the public surface, one symbol at a time. For the
argument-by-argument manual with the measurements, read [GUIDE.md](GUIDE.md);
for what each function accepts and rejects, [THEORY.md](THEORY.md) §9 is the
arbiter.

Most sections start from the same tiny image, 4×4 with one channel, whose
pixels hold 0 to 15 in reading order. Every value names its own position, so
you can follow each pixel through each call:

```
 0  1  2  3
 4  5  6  7
 8  9 10 11
12 13 14 15
```

---

## 0. Setup

```python
>>> import patchcraft
>>> len(patchcraft.__all__)
20
>>> sorted(patchcraft.__all__)
['Cache', 'PairedTilingSpec', 'PatchMeta', 'PatchPair', 'Patchify',
 'TilingSpec', 'WeightKind', 'accel_available', 'extract', 'num_patches',
 'pair', 'paired_tilings', 'patch_metrics', 'per_patch_mse',
 'per_patch_psnr', 'reconstruct', 'resize', 'scale_factor', 'stitch',
 'tilings']

```

Twenty public symbols, frozen by `tests/test_public_api.py`. Functions are
lowercase; classes, dataclasses and NamedTuples are PascalCase. The version is
not shown here on purpose: it comes from the git tag, so any literal would be
wrong the moment it is written.

---

## 1. `num_patches`: pre-flight, no allocation

The shape formula from [`THEORY.md`](THEORY.md) §1, exposed as a
function. Takes `(H, W)` or `(C, H, W)`; channels are ignored. Returns
`(num_h, num_w)` without touching a tensor.

```python
>>> from patchcraft import num_patches
>>> num_patches((28, 28), patch_size=7, stride=7)
(4, 4)
>>> num_patches((28, 28), patch_size=4, stride=2)
(13, 13)
>>> num_patches((3, 32, 32), patch_size=8, stride=8)  # accepts (C, H, W) too
(4, 4)
>>> num_patches((4, 4), patch_size=8, stride=1)        # patch too big -> (0, 0)
(0, 0)

```

Use it for memory planning before allocating, for shape assertions,
or to fill in a progress bar.

---

## 2. `tilings`: every full-coverage geometry for an image

```python
>>> from patchcraft import tilings
>>> for t in tilings((28, 28)):
...     print(t)
TilingSpec(patch_size=(2,  2), stride=(2,  2), dilation=(1, 1), num_patches=(14, 14), total_patches=196, overlap=False)
TilingSpec(patch_size=(4,  4), stride=(4,  4), dilation=(1, 1), num_patches=(7,  7),  total_patches=49,  overlap=False)
TilingSpec(patch_size=(7,  7), stride=(7,  7), dilation=(1, 1), num_patches=(4,  4),  total_patches=16,  overlap=False)
TilingSpec(patch_size=(14, 14), stride=(14, 14), dilation=(1, 1), num_patches=(2,  2),  total_patches=4,   overlap=False)
TilingSpec(patch_size=(28, 28), stride=(28, 28), dilation=(1, 1), num_patches=(1,  1),  total_patches=1,   overlap=False)

```

Exact tilings only. Divisors of 28 that are `>= 2` are
`{2, 4, 7, 14, 28}`, so five entries, matching what the function
emits. Every spec here gives a bit-exact `extract` + `reconstruct`
round-trip (see §5).

```python
>>> len(tilings((28, 28), allow_overlap=True))
73

```

With `allow_overlap=True` the function also emits `stride < patch_size`
geometries where `(H - p) % s == 0` (clean-edge overlap; single-patch
grids are skipped, since one patch makes the stride unobservable).
Useful when you want training data with stride < ph; whether the
round-trip stays bit-exact then depends on the count map (see §5).

---

## 3. `extract`: patches from one `(C, H, W)` image

```python
>>> import torch
>>> from patchcraft import extract
>>> tiny = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
>>> tiny_patches = extract(tiny, patch_size=2, stride=2)
>>> tiny_patches.shape   # (L, C, ph, pw): four 2x2 patches of one channel
torch.Size([4, 1, 2, 2])
>>> tiny_patches[:, 0]   # top-left, top-right, bottom-left, bottom-right
tensor([[[ 0.,  1.],
         [ 4.,  5.]],
<BLANKLINE>
        [[ 2.,  3.],
         [ 6.,  7.]],
<BLANKLINE>
        [[ 8.,  9.],
         [12., 13.]],
<BLANKLINE>
        [[10., 11.],
         [14., 15.]]])

```

The patches come in reading order: across the first row of the grid, then
the next. With a stride smaller than the patch the window moves less than its
own width, so neighbouring patches share pixels. At stride 1 there are nine
patches, and pixel 5 is in the first two below and in two more further on:

```python
>>> extract(tiny, patch_size=2, stride=1)[:3, 0]
tensor([[[0., 1.],
         [4., 5.]],
<BLANKLINE>
        [[1., 2.],
         [5., 6.]],
<BLANKLINE>
        [[2., 3.],
         [6., 7.]]])

```

The same call on a 28×28 image, the size of an MNIST digit:

```python
>>> img = torch.arange(28 * 28, dtype=torch.float32).reshape(1, 28, 28)
>>> img.shape
torch.Size([1, 28, 28])
>>> patches = extract(img, patch_size=7, stride=7)
>>> patches.shape   # (L, C, ph, pw); L = 4*4 = 16
torch.Size([16, 1, 7, 7])
>>> patches.dtype
torch.float32

```

Truncation is the only boundary policy. If the geometry fits no
patch, `extract` returns `Tensor[0, C, ph, pw]` instead of raising,
callers decide whether that's an error.

---

## 4. `Patchify`: callable companion for `transforms.Compose`

```python
>>> from patchcraft import Patchify
>>> patchify = Patchify(patch_size=4, stride=2)
>>> patchify
Patchify(patch_size=(4, 4), stride=(2, 2), dilation=(1, 1))
>>> patchify(img).shape   # 13*13 = 169 overlapping patches
torch.Size([169, 1, 4, 4])
>>> torch.equal(Patchify(patch_size=2, stride=2)(tiny), tiny_patches)   # extract, frozen
True

```

Eager validation: a bad geometry fails at `Patchify(...)`, not at the
first `__call__`. `__slots__`-bound, with no cache, no fixed image_size,
no surprise state. See [ADR 0002](ADR/0002-patchify-transform.md) for
the rationale.

---

## 5. `reconstruct`: bit-exact when `stride == patch_size`

```python
>>> from patchcraft import reconstruct
>>> recon = reconstruct(patches, image_shape=img.shape, stride=7)
>>> recon.shape
torch.Size([1, 28, 28])
>>> torch.equal(recon, img)
True

```

When `stride == patch_size` every pixel is covered exactly once, so
`reconstruct` only rearranges the patches: no fold, no count map and no
arithmetic. Filling patch `k` with the number `k` shows where each one lands,
here six 2×2 patches on a 4×6 image:

```python
>>> labels = torch.arange(6.).view(6, 1, 1, 1).expand(6, 1, 2, 2)
>>> reconstruct(labels, image_shape=(1, 4, 6), stride=2)[0]
tensor([[0., 0., 1., 1., 2., 2.],
        [0., 0., 1., 1., 2., 2.],
        [3., 3., 4., 4., 5., 5.],
        [3., 3., 4., 4., 5., 5.]])

```

### Overlap: weighted, exact under the count-map rule

```python
>>> ps_overlap = extract(img, patch_size=4, stride=2)  # 169 overlapping patches
>>> recon_overlap = reconstruct(ps_overlap, image_shape=img.shape, stride=2)
>>> torch.equal(recon_overlap, img)
True

```

Each pixel covered by *k* patches; each contribution is the original
value; sum is `k * value`; division by the count map gives back
`value`. Bit-exact when every count in the map is a power of two, so
`stride == patch_size` always qualifies and `stride == patch_size / 2`
does too (counts 1, 2, 4, as above). Otherwise the error is bounded
per pixel by `(k+1)·eps·|v|`,
whatever the dtype: the deciding axis is the geometry, not the
precision.

When the patches disagree, which is what a model's output does, each pixel
gets the mean of what its patches say. Below, nine overlapping patches cover
the tiny image; the middle one says 1 and the other eight say 0:

```python
>>> votes = extract(torch.zeros(1, 4, 4), patch_size=2, stride=1)
>>> votes[4] = 1.0
>>> reconstruct(votes, image_shape=(1, 4, 4), stride=1)[0]
tensor([[0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.2500, 0.2500, 0.0000],
        [0.0000, 0.2500, 0.2500, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000]])

```

Each of the four pixels the middle patch covers is covered by four patches in
all, so it comes back as the mean of one 1 and three 0s. That mean is the
least-squares answer, which [`THEORY.md`](THEORY.md) §2.6 derives.

`reconstruct` rejects `dilation != 1` and `stride > patch_size`
(partial coverage forbidden, see [`THEORY.md`](THEORY.md) §9.2).

---

## 6. `stitch`: blend modified patches back with a window kernel

`reconstruct` is the exact inverse of `extract` under the
count-map rule of §5, so it is bit-exact when every count is a power
of two. When patches have been *modified* (model output, denoised,
super-resolved)
uniform averaging makes the disagreement between adjacent patches
show up as visible seams. `stitch` weights each patch by a 2-D
window kernel (`uniform`, `hann`, `gaussian`) so each pixel "trusts"
patches whose center is closer to it more than patches whose edge
falls on it.

Visual: the kernel shape applied to each patch for `patch_size=4`:

```
   weight="uniform"     weight="hann"        weight="gaussian"
   (== reconstruct)     centers > edges      centers >> edges
                        (never 0)            (never 0)

   + + + +              . X X .              . o o .
   + + + +              X X X X              o X X o
   + + + +              X X X X              o X X o
   + + + +              . X X .              . o o .
```

Hann uses the interior of a longer symmetric window
(`hann_window(n + 2, periodic=False)[1:-1]`), so the edge weight is small but
never zero, where a plain symmetric window would be exactly `0` at both edges
and zero the pixels covered only by patch edges. Gaussian uses per-axis
`sigma = max(1, ph / 4)` / `max(1, pw / 4)` so the edge weight is small
but never zero.

```python
>>> from patchcraft import stitch, reconstruct, extract
>>> img_small = torch.full((1, 8, 8), 0.5)  # uniform gray
>>> patches_small = extract(img_small, patch_size=4, stride=4)  # 4 patches

>>> # weight="uniform" is mathematically equivalent to reconstruct.
>>> torch.equal(
...     stitch(patches_small, image_shape=img_small.shape, stride=4, weight="uniform"),
...     reconstruct(patches_small, image_shape=img_small.shape, stride=4),
... )
True

>>> # weight="hann" blends overlap seams away; with unmodified patches the
>>> # round-trip is exact, corners included, because the window never hits 0.
>>> out_hann = stitch(patches_small, image_shape=img_small.shape,
...                    stride=4, weight="hann")
>>> out_hann[0, 0, 0].item()   # corner: small edge weight, still recovered
0.5
>>> out_hann[0, 1, 1].item()   # interior of patch: full weight, recovered
0.5

>>> # weight="gaussian" likewise has no zero-weight edges.
>>> out_gauss = stitch(patches_small, image_shape=img_small.shape,
...                     stride=4, weight="gaussian")
>>> out_gauss[0, 0, 0].item()
0.5

```

Where `stitch` earns its place is where patches disagree. Two 1×4 patches,
one saying 0 and the other 1, overlap on columns 2 and 3 of a 1×6 strip:

```python
>>> two = torch.stack([torch.zeros(1, 1, 4), torch.ones(1, 1, 4)])
>>> for weight in ("uniform", "hann", "gaussian"):
...     out = stitch(two, image_shape=(1, 1, 6), stride=(1, 2), weight=weight)
...     print(f"{weight:9s}", out[0, 0].round(decimals=2))
uniform   tensor([0.0000, 0.0000, 0.5000, 0.5000, 1.0000, 1.0000])
hann      tensor([0.0000, 0.0000, 0.2800, 0.7200, 1.0000, 1.0000])
gaussian  tensor([0.0000, 0.0000, 0.2700, 0.7300, 1.0000, 1.0000])

```

Uniform splits the difference into a step. Hann and gaussian let each patch
dominate near its own centre, so the step becomes a ramp, and a ramp is what
hides a seam.

Use `stitch` for "I have model output and want a single image";
use `reconstruct` for "I have unmodified patches and want my
original image back, exactly." `stitch` accepts only floating-point
patches (window kernels are float-valued; integer patches would
silently quantize). See [`THEORY.md`](THEORY.md) §2.5 for the math
and §9.9 for the full contract.

---

## 7. `pair`: LR / HR correspondences

The HR image below is the tiny image with every pixel blown up into a 2×2
block, so the two sides show the same content at two scales:

```python
>>> from patchcraft import pair
>>> lr = tiny
>>> hr = lr.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)   # 1x8x8
>>> result = pair(lr, hr,
...               lr_patch_size=2, scale_factor=2, stride=2,
...               image_id='demo-0')
>>> result.lr_patches.shape, result.hr_patches.shape
(torch.Size([4, 1, 2, 2]), torch.Size([4, 1, 4, 4]))
>>> len(result)
4
>>> result.lr_patches[1, 0]
tensor([[2., 3.],
        [6., 7.]])
>>> result.hr_patches[1, 0]
tensor([[2., 2., 3., 3.],
        [2., 2., 3., 3.],
        [6., 6., 7., 7.],
        [6., 6., 7., 7.]])
>>> result.metas[1]
PatchMeta(patch_index=1, row=0, col=2,
          lr_patch_size=(2, 2), hr_patch_size=(4, 4),
          image_id='demo-0')

```

Patch 1 is the top-right one on both sides: the LR patch holds pixels 2, 3,
6 and 7, and the HR patch holds the same four as 2×2 blocks.

LR coords (`row`, `col`) are in pixel space; multiply by
`scale_factor` to get HR coords. `image_id` is opaque metadata
forwarded as-is.

LR and HR must share `C`, dtype, and device. `scale_factor` must be
a positive `int`. HR shape must equal `scale_factor * lr.shape` on
both spatial axes.

---

## 8. `resize`: output type matches input

```python
>>> from patchcraft import resize
>>> from PIL import Image
>>> import numpy as np

>>> pil_img = Image.fromarray((np.arange(16*16*3) % 256)
...                            .astype(np.uint8).reshape(16, 16, 3),
...                            mode="RGB")
>>> pil_out = resize(pil_img, target_size=(8, 8), backend="pil")
>>> type(pil_out).__name__, pil_out.size, pil_out.mode
('Image', (8, 8), 'RGB')

>>> tensor_img = torch.rand(3, 16, 16)
>>> tensor_out = resize(tensor_img, target_size=(8, 8), backend="torch")
>>> type(tensor_out).__name__, tuple(tensor_out.shape), tensor_out.dtype
('Tensor', (3, 8, 8), torch.float32)

```

On the tiny image, going down by 2 with the torch backend's default
bilinear averages each 2×2 block, and going back up with `"nearest"` repeats
each value as a block:

```python
>>> down = resize(tiny, target_size=(2, 2), backend="torch")
>>> down[0]
tensor([[ 2.5000,  4.5000],
        [10.5000, 12.5000]])
>>> resize(down, target_size=(4, 4), backend="torch", resample="nearest")[0]
tensor([[ 2.5000,  2.5000,  4.5000,  4.5000],
        [ 2.5000,  2.5000,  4.5000,  4.5000],
        [10.5000, 10.5000, 12.5000, 12.5000],
        [10.5000, 10.5000, 12.5000, 12.5000]])

```

The round trip does not return `tiny`: a resize discards what it averages,
which is why ADR 0003 counts it among the inexact operations. The averaging
holds at a factor of 2; at larger factors the torch backend samples rather
than averages, because it does not antialias, while the PIL backend does.

PIL in → PIL out. Tensor in → Tensor out. Cross-backend
(tensor + `backend="pil"`, or PIL + `backend="torch"`) is supported
via a float32 [0, 1] / uint8 hop. CUDA tensors are accepted only
with `backend="torch"`.

---

## 9. `Cache`: bytes in, bytes out, atomic, version-aware

```python
>>> import tempfile
>>> from patchcraft import Cache
>>> tmp = tempfile.mkdtemp()
>>> c = Cache(tmp, namespace="demo", version=1)
>>> c
Cache(root=..., namespace='demo', version=1)

>>> config = {"target_size": (8, 8), "resample": "lanczos"}
>>> k = c.key_for("image-fingerprint", config)
>>> k[:16]                       # short prefix used as filename
'80f1beaae1321b83'

>>> c.put(k, b"some pickled payload")
>>> c.get(k)
b'some pickled payload'

>>> c.get(c.key_for("missing"))  # absent key -> None
>>> # (None)
```

Bytes-in/bytes-out; the caller picks the serialization (`torch.save`,
`pickle`, raw bytes, whatever). Atomic write via `*.tmp` plus
`os.replace`; retries transient `PermissionError` (OneDrive,
antivirus, indexer) up to five times with exponential backoff.

### Version bump invalidates by construction

```python
>>> c2 = Cache(tmp, namespace="demo", version=2)
>>> k2 = c2.key_for("image-fingerprint", config)
>>> k == k2
False
>>> c2.get(k2)
>>> # (None); different key by construction; old entries unreachable
```

Bumping `version` produces a different SHA-256 for the same parts,
so old entries become transparently unreachable. No delete needed,
no migration code, no race between reader and "is this still valid?"
logic.

---

## 10. `scale_factor`: integer scale between two image shapes

```python
>>> from patchcraft import scale_factor
>>> scale_factor((14, 14), (28, 28))
2
>>> scale_factor((28, 28), (28, 28))      # identity
1
>>> scale_factor((14, 14), (27, 27))      # non-integer ratio
>>> # (None)
>>> scale_factor((10, 10), (20, 30))      # anisotropic
>>> # (None)
```

Pre-flight check before calling `pair`. Returns `None` when no positive
integer `k` exists such that `hr == k * lr`. Accepts `(H, W)` or
`(C, H, W)` (channels ignored).

---

## 11. `paired_tilings`: aligned LR↔HR geometries

```python
>>> from patchcraft import paired_tilings
>>> for p in paired_tilings((14, 14), (28, 28)):
...     print(f"lr={p.lr.patch_size} stride={p.lr.stride} | "
...           f"hr={p.hr.patch_size} stride={p.hr.stride} | "
...           f"total={p.lr.total_patches} | sf={p.scale_factor}")
lr=(2, 2)   stride=(2, 2)   | hr=(4, 4)   stride=(4, 4)   | total=49 | sf=2
lr=(7, 7)   stride=(7, 7)   | hr=(14, 14) stride=(14, 14) | total=4  | sf=2
lr=(14, 14) stride=(14, 14) | hr=(28, 28) stride=(28, 28) | total=1  | sf=2

```

Every entry is guaranteed by construction to:

- have identical `total_patches` on both sides (no off-by-one),
- have patch `k` cover the same image region on LR and HR
  (HR coords = LR coords × `scale_factor`),
- be safe to feed straight into `pair(lr_image, hr_image,
  lr_patch_size=p.lr.patch_size[0], scale_factor=p.scale_factor,
  stride=p.lr.stride[0])`.

This is the answer to the question "for MNIST 14×14 ↔ 28×28, what
patch geometries align with the same `k`?", so three options, take
your pick based on how big a patch you want vs how many examples.

---

## 12. `patch_metrics` / `per_patch_psnr`: canonical comparisons

```python
>>> import torch
>>> from patchcraft import extract, patch_metrics, per_patch_psnr
>>> img = torch.arange(28 * 28, dtype=torch.float32).reshape(1, 28, 28) / 784.0
>>> patches_a = extract(img, patch_size=7, stride=7)
>>> patches_b = patches_a + 0.01  # uniform +0.01 noise

>>> patch_metrics(patches_a, patches_a)        # identical
{'mae': 0.0, 'mse': 0.0, 'max_abs': 0.0, 'psnr_db': inf}

>>> patch_metrics(patches_a, patches_b)        # +0.01 across the board
{'mae': 0.00999..., 'mse': 9.99998...e-05, 'max_abs': 0.0100000..., 'psnr_db': 40.0000...}

>>> per_patch_psnr(patches_a, patches_b)
tensor([40.0000, 40.0000, ..., 40.0000, 40.0000], dtype=torch.float64)

```

One value per patch is what finds the patch that went wrong. On the tiny
image, only the bottom-right patch is changed:

```python
>>> from patchcraft import per_patch_mse
>>> a = extract(tiny / 16, patch_size=2, stride=2)
>>> b = a.clone()
>>> b[3] += 0.1
>>> per_patch_mse(a, b)
tensor([0.0000, 0.0000, 0.0000, 0.0100], dtype=torch.float64)

```

- `patch_metrics` reduces over the whole tensor and returns a dict
  of Python floats. Internal accumulation is `float64`; identical
  inputs give `psnr_db == +inf`.
- `per_patch_psnr(a, b)` and `per_patch_mse(a, b)` keep the leading
  axis, returning one value per patch. Useful for "which patches
  did my model handle worst?".

Both reject shape, dtype, and device mismatches.

---

## 13. End-to-end pre-flight for an LR/HR training set (combining 7, 10, 11, 12)

```python
>>> # Goal: train a model that maps 14x14 patches to corresponding 28x28 patches.
>>> # Step 1: enumerate viable LR/HR geometries with same patch count.
>>> for p in paired_tilings((14, 14), (28, 28)):
...     print(p.lr.patch_size, '<-->', p.hr.patch_size,
...           f'({p.lr.total_patches} patches each)')
(2, 2)   <--> (4, 4)   (49 patches each)
(7, 7)   <--> (14, 14) (4 patches each)
(14, 14) <--> (28, 28) (1 patches each)

>>> # Step 2: pick a spec and call pair() with the LR/HR pixel data:
>>> # from patchcraft import pair
>>> # result = pair(lr_img, hr_img,
>>> #               lr_patch_size=2, scale_factor=2, stride=2,
>>> #               image_id='mnist-0')

>>> # Step 3: train.

>>> # Step 4: measure per-patch error so you know which patches the
>>> #         model handles worst.
>>> # err = per_patch_psnr(model(result.lr_patches), result.hr_patches)
```

This is the loop a super-resolution training pipeline runs.
PatchCraft covers steps 1, 2, and 4. Step 3 is the consumer's job.

---

## 14. Composing in a `torch` pipeline

PatchCraft is built to drop into someone else's pipeline. `Patchify`
is the integration point.

```python
from patchcraft import Patchify
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.GaussianBlur(kernel_size=3),
    Patchify(patch_size=4, stride=2),  # PatchCraft as one step
])

dataset = MNIST(root="...", transform=transform)
loader = DataLoader(dataset, num_workers=4, batch_size=...)
```

The `DataLoader` parallelizes over images for free (worker processes
each apply the full `transform` to one image at a time). PatchCraft
itself remains one-image-at-a-time; multi-image throughput is the
pipeline's job, not the lib's. See [`SCOPE.md`](SCOPE.md) for the
full responsibilities table.

---

## What this page deliberately does not show

- **Dataset loading** stays out: `tests/_datasets.py::mnist_subset` is the dev
  fixture; see [`AUXILIARY.md`](AUXILIARY.md). PatchCraft core never
  touches a dataset.
- **Training loops** are out of scope. PatchCraft is infrastructure.
- **Multi-image batching** is the caller's job: use `torch.vmap`, a Python loop, or a
  `DataLoader`. The lib stays one-image-at-a-time on purpose.
- **GPU details**: `extract`, `Patchify`, `reconstruct`, `pair`, and
  `resize(..., backend="torch")` all preserve device. `resize(...,
  backend="pil")` requires CPU input.
