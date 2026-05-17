# PatchKit — usage walkthrough

Every snippet below was captured from a live REPL session against
`patchkit==0.1.0`. Outputs are real, not pseudocode. Re-run the
script behind it at any time:

```
python lab/usage_demo.py
```

The script is intentionally not part of the wheel — see
[`AUXILIARY.md`](AUXILIARY.md) for why bench scripts live in
`lab/` and stay out of the shipped package.

---

## 0. Setup

```python
>>> import patchkit
>>> patchkit.__version__
'0.1.0'
>>> sorted(patchkit.__all__)
['Cache', 'PairedTilingSpec', 'PatchMeta', 'PatchPair', 'Patchify',
 'TilingSpec', 'extract', 'num_patches', 'pair', 'paired_tilings',
 'patch_metrics', 'per_patch_mse', 'per_patch_psnr', 'reconstruct',
 'resize', 'scale_factor', 'stitch', 'tilings']
```

Eighteen public symbols on the v0.2-track main branch (the wheel
tagged `v0.1.0` shipped eleven; `__version__` will bump to `0.2.0`
once the additions validate in real use). Functions are lowercase;
classes / dataclasses / NamedTuples are PascalCase.

---

## 1. `num_patches` — pre-flight, no allocation

The shape formula from [`THEORY.md`](THEORY.md) §1, exposed as a
function. Takes `(H, W)` or `(C, H, W)`; channels are ignored. Returns
`(num_h, num_w)` without touching a tensor.

```python
>>> from patchkit import num_patches
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

## 2. `tilings` — every full-coverage geometry for an image

```python
>>> from patchkit import tilings
>>> for t in tilings((28, 28)):
...     print(t)
TilingSpec(patch_size=(2,  2), stride=(2,  2), dilation=(1, 1), num_patches=(14, 14), total_patches=196, overlap=False)
TilingSpec(patch_size=(4,  4), stride=(4,  4), dilation=(1, 1), num_patches=(7,  7),  total_patches=49,  overlap=False)
TilingSpec(patch_size=(7,  7), stride=(7,  7), dilation=(1, 1), num_patches=(4,  4),  total_patches=16,  overlap=False)
TilingSpec(patch_size=(14, 14), stride=(14, 14), dilation=(1, 1), num_patches=(2,  2),  total_patches=4,   overlap=False)
TilingSpec(patch_size=(28, 28), stride=(28, 28), dilation=(1, 1), num_patches=(1,  1),  total_patches=1,   overlap=False)
```

Exact tilings only. Divisors of 28 that are `>= 2` are
`{2, 4, 7, 14, 28}` — five entries, matching what the function
emits. Every spec here gives a bit-exact `extract` + `reconstruct`
round-trip (see §5).

```python
>>> len(tilings((28, 28), allow_overlap=True))
100
```

With `allow_overlap=True` the function also emits `stride < patch_size`
geometries where `(H - p) % s == 0` (clean-edge overlap). Useful when
you want training data with stride < ph but reconstruction must still
be exact.

---

## 3. `extract` — patches from one `(C, H, W)` image

```python
>>> import torch
>>> from patchkit import extract
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
patch, `extract` returns `Tensor[0, C, ph, pw]` instead of raising —
callers decide whether that's an error.

---

## 4. `Patchify` — callable companion for `transforms.Compose`

```python
>>> from patchkit import Patchify
>>> patchify = Patchify(patch_size=4, stride=2)
>>> patchify
Patchify(patch_size=(4, 4), stride=(2, 2), dilation=(1, 1))
>>> patchify(img).shape   # 13*13 = 169 overlapping patches
torch.Size([169, 1, 4, 4])
```

Eager validation: a bad geometry fails at `Patchify(...)`, not at the
first `__call__`. `__slots__`-bound — no cache, no fixed image_size,
no surprise state. See [ADR 0002](ADR/0002-patchify-transform.md) for
the rationale.

---

## 5. `reconstruct` — bit-exact when `stride == patch_size`

```python
>>> from patchkit import reconstruct
>>> recon = reconstruct(patches, image_shape=img.shape, stride=7)
>>> recon.shape
torch.Size([1, 28, 28])
>>> torch.equal(recon, img)
True
```

`F.fold` plus a same-geometry fold-of-ones count map. When
`stride == patch_size` every pixel is covered exactly once and
reconstruction is a cheap copy (count is all-ones; division is no-op).

### Overlap: weighted, still exact

```python
>>> ps_overlap = extract(img, patch_size=4, stride=2)  # 169 overlapping patches
>>> recon_overlap = reconstruct(ps_overlap, image_shape=img.shape, stride=2)
>>> torch.allclose(recon_overlap, img)
True
```

Each pixel covered by *k* patches; each contribution is the original
value; sum is `k * value`; division by the count map gives back
`value`. Bit-exact for `float64`; within `~1 ULP` for `float32`.

`reconstruct` rejects `dilation != 1` and `stride > patch_size`
(partial coverage forbidden — see [`THEORY.md`](THEORY.md) §9.2).

---

## 6. `stitch` — blend modified patches back with a window kernel

`reconstruct` is the bit-exact inverse of `extract`. When patches
have been *modified* — model output, denoised, super-resolved —
uniform averaging makes the disagreement between adjacent patches
show up as visible seams. `stitch` weights each patch by a 2-D
window kernel (`uniform`, `hann`, `gaussian`) so each pixel "trusts"
patches whose center is closer to it more than patches whose edge
falls on it.

```python
>>> from patchkit import stitch, reconstruct, extract
>>> img_small = torch.full((1, 8, 8), 0.5)  # uniform gray
>>> patches_small = extract(img_small, patch_size=4, stride=4)  # 4 patches

>>> # weight="uniform" is mathematically equivalent to reconstruct.
>>> torch.equal(
...     stitch(patches_small, image_shape=img_small.shape, stride=4, weight="uniform"),
...     reconstruct(patches_small, image_shape=img_small.shape, stride=4),
... )
True

>>> # weight="hann" zeros image corners that fall on Hann's edge-weight-zero
>>> # positions (documented artifact). Interior pixels are preserved.
>>> out_hann = stitch(patches_small, image_shape=img_small.shape,
...                    stride=4, weight="hann")
>>> out_hann[0, 0, 0].item()   # corner: covered only at relative (0,0) where hann=0
0.0
>>> out_hann[0, 1, 1].item()   # interior of patch: hann>0 there, recovered
0.5

>>> # weight="gaussian" has no zero-weight edges, so no corner artifact.
>>> out_gauss = stitch(patches_small, image_shape=img_small.shape,
...                     stride=4, weight="gaussian")
>>> out_gauss[0, 0, 0].item()
0.5
```

Use `stitch` for "I have model output and want a single image";
use `reconstruct` for "I have unmodified patches and want my
original image back, exactly." `stitch` accepts only floating-point
patches (window kernels are float-valued; integer patches would
silently quantize). See [`THEORY.md`](THEORY.md) §2.5 for the math
and §9.9 for the full contract.

---

## 7. `pair` — LR / HR correspondences

```python
>>> from patchkit import pair
>>> lr = torch.arange(8 * 8,   dtype=torch.float32).reshape(1, 8,  8)
>>> hr = torch.arange(16 * 16, dtype=torch.float32).reshape(1, 16, 16)
>>> result = pair(lr, hr,
...               lr_patch_size=4, scale_factor=2, stride=4,
...               image_id='demo-0')
>>> result.lr_patches.shape
torch.Size([4, 1, 4, 4])
>>> result.hr_patches.shape
torch.Size([4, 1, 8, 8])
>>> len(result)
4
>>> result.metas[0]
PatchMeta(patch_index=0, row=0, col=0,
          lr_patch_size=(4, 4), hr_patch_size=(8, 8),
          image_id='demo-0')
>>> result.metas[-1]
PatchMeta(patch_index=3, row=4, col=4,
          lr_patch_size=(4, 4), hr_patch_size=(8, 8),
          image_id='demo-0')
```

LR coords (`row`, `col`) are in pixel space; multiply by
`scale_factor` to get HR coords. `image_id` is opaque metadata
forwarded as-is.

LR and HR must share `C`, dtype, and device. `scale_factor` must be
a positive `int`. HR shape must equal `scale_factor * lr.shape` on
both spatial axes.

---

## 8. `resize` — output type matches input

```python
>>> from patchkit import resize
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

PIL in → PIL out. Tensor in → Tensor out. Cross-backend
(tensor + `backend="pil"`, or PIL + `backend="torch"`) is supported
via a float32 [0, 1] / uint8 hop. CUDA tensors are accepted only
with `backend="torch"`.

---

## 9. `Cache` — bytes in, bytes out, atomic, version-aware

```python
>>> import tempfile
>>> from patchkit import Cache
>>> tmp = tempfile.mkdtemp()
>>> c = Cache(tmp, namespace="demo", version=1)
>>> c
Cache(root='.../tmpXXXXXXXX/demo', namespace='demo', version=1)

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
>>> # (None) — different key by construction; old entries unreachable
```

Bumping `version` produces a different SHA-256 for the same parts,
so old entries become transparently unreachable. No delete needed,
no migration code, no race between reader and "is this still valid?"
logic.

---

## 10. `scale_factor` — integer scale between two image shapes

```python
>>> from patchkit import scale_factor
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

## 11. `paired_tilings` — aligned LR↔HR geometries

```python
>>> from patchkit import paired_tilings
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
patch geometries align with the same `k`?" — three options, take
your pick based on how big a patch you want vs how many examples.

---

## 12. `patch_metrics` / `per_patch_psnr` — canonical comparisons

```python
>>> import torch
>>> from patchkit import extract, patch_metrics, per_patch_psnr
>>> img = torch.arange(28 * 28, dtype=torch.float32).reshape(1, 28, 28) / 784.0
>>> patches_a = extract(img, patch_size=7, stride=7)
>>> patches_b = patches_a + 0.01  # uniform +0.01 noise

>>> patch_metrics(patches_a, patches_a)        # identical
{'mae': 0.0, 'mse': 0.0, 'max_abs': 0.0, 'psnr_db': inf}

>>> patch_metrics(patches_a, patches_b)        # +0.01 across the board
{'mae': 0.009999...,
 'mse': 9.9999e-05,
 'max_abs': 0.01000...,
 'psnr_db': 40.0000...}    # 10 * log10(1 / 0.0001) = 40 dB

>>> per_patch_psnr(patches_a, patches_b)
tensor([40.0000, 40.0000, 40.0000, ..., 40.0000])  # one per patch (16 total)
```

- `patch_metrics` reduces over the whole tensor and returns a dict
  of Python floats. Internal accumulation is `float64`; identical
  inputs give `psnr_db == +inf`.
- `per_patch_psnr(a, b)` and `per_patch_mse(a, b)` keep the leading
  axis, returning one value per patch. Useful for "which patches
  did my model handle worst?".

Both reject shape, dtype, and device mismatches.

---

## 13. End-to-end QPatchSR-style pre-flight (synthesizing 10, 11, 12)

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
>>> # from patchkit import pair
>>> # result = pair(lr_img, hr_img,
>>> #               lr_patch_size=2, scale_factor=2, stride=2,
>>> #               image_id='mnist-0')

>>> # Step 3: train.

>>> # Step 4: measure per-patch error so you know which patches the
>>> #         model handles worst.
>>> # err = per_patch_psnr(model(result.lr_patches), result.hr_patches)
```

This is the loop QPatchSR (and any similar consumer) will run.
PatchKit covers steps 1, 2, and 4. Step 3 is the consumer's job.

---

## 14. Composing in a `torch` pipeline

PatchKit is built to drop into someone else's pipeline. `Patchify`
is the integration point.

```python
from patchkit import Patchify
from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.GaussianBlur(kernel_size=3),
    Patchify(patch_size=4, stride=2),  # PatchKit as one step
])

dataset = MNIST(root="...", transform=transform)
loader = DataLoader(dataset, num_workers=4, batch_size=...)
```

The `DataLoader` parallelizes over images for free (worker processes
each apply the full `transform` to one image at a time). PatchKit
itself remains one-image-at-a-time; multi-image throughput is the
pipeline's job, not the lib's. See [`SCOPE.md`](SCOPE.md) for the
full responsibilities table.

---

## What this page deliberately does not show

- **Dataset loading** — `tests/_datasets.py::mnist_subset` is the dev
  fixture; see [`AUXILIARY.md`](AUXILIARY.md). PatchKit core never
  touches a dataset.
- **Training loops** — out of scope. PatchKit is infrastructure.
- **Multi-image batching** — use `torch.vmap`, a Python loop, or a
  `DataLoader`. The lib stays one-image-at-a-time on purpose.
- **GPU details** — `extract`, `Patchify`, `reconstruct`, `pair`, and
  `resize(..., backend="torch")` all preserve device. `resize(...,
  backend="pil")` requires CPU input.
