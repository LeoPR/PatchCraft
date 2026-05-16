# PatchKit — scope and responsibilities

This document maps every concern in the patch-extraction pipeline to
**whose job it is**: the PatchKit core, the caller's pipeline, or a
grey area that warrants discussion. It also explains where
parallelism happens and where PatchKit explicitly *does not* try to
help.

The short version lives in [`THEORY.md`](THEORY.md) §0; this is the
long-form companion.

---

## 1. Responsibilities matrix

| Concern | PatchKit core | Caller / pipeline | Grey area |
|---|---|---|---|
| Extract patches from **one** image | ✓ (`extract`, `Patchify`) | | |
| Reconstruct an image from its patches | ✓ (`reconstruct`) | | |
| Pair LR ↔ HR patches | ✓ (`pair`, `PatchPair`, `PatchMeta`) | | |
| Resize one image | ✓ (`resize`, PIL or torch) | | |
| Compute patch geometry without allocating | ✓ (`num_patches`, `tilings`) | | |
| Cache derived bytes on disk | ✓ (`Cache`) | | discussed in §4 below |
| Drop-in to `transforms.Compose([...])` | ✓ (`Patchify`) | | |
| **Loop over many images** | ✗ — caller writes the loop | ✓ (`for`, `vmap`, `DataLoader`) | |
| **Download / cache a dataset** | ✗ — never | ✓ (`torchvision.datasets`, custom) | |
| Shuffle / sample / batch a dataset | ✗ | ✓ (`DataLoader`, `Sampler`) | |
| Training loop / loss / optimizer | ✗ | ✓ | |
| Data augmentation other than resize | ✗ | ✓ (`torchvision.transforms`) | |
| GPU device management | ✗ — preserves whatever you give it | ✓ (`.cuda()`, `.to(device)`) | |
| Mixed-precision dtype management | ✗ — preserves dtype | ✓ (`amp.autocast`, `.to(dtype)`) | |
| Worker-process parallelism | ✗ | ✓ (`DataLoader(num_workers=...)`) | |
| Distributed training | ✗ | ✓ (`DistributedDataParallel`, etc.) | |
| Quantization (color / vector) | ✗ — v0.1 deferred | ✓ (caller pre-processes) | future companion package |
| Channels-last layout | ✗ — v0.1 channels-first only | ✓ (caller converts) | open question (THEORY §8) |
| Label-stratified subset selection | ✗ — moved out of core | (`tests/_datasets.py::label_subset` for dev) | |
| Logging / metrics / progress | ✗ | ✓ | |
| Pretty pictures / visualization | ✗ | ✓ (matplotlib, torchvision.utils) | |

Heuristic: anything that needs more than one image to make sense, or
that needs to talk to disk for reasons other than caching derived
bytes, is the caller's job.

---

## 2. Parallelization: where it happens, who owns it

PatchKit's primitives are one-image-at-a-time. That sounds slow
until you realize that **`F.unfold` and `F.fold` are already
parallel** inside one call, and that the standard way to scale
"many images" in PyTorch is parallelism at the pipeline level, not
inside the primitive.

### 2.1 Inside `extract` — torch already does it

`extract(image, ...)` reduces to a single `torch.nn.functional.unfold`
call. There is no Python loop over patches. The work happens in:

- **CPU**: `im2col` — vectorized C++ implementation; uses SIMD where
  applicable. Operates on the whole `(C, H, W)` tensor at once.
- **CUDA**: a single `im2col` kernel launch. Patches are produced in
  parallel across thousands of CUDA threads.

You get the parallelism for free; PatchKit does not need to and does
not try to.

### 2.2 Across many images — caller's pipeline

For "I have N images and want patches from each", the standard
torch idioms apply unchanged. PatchKit does not add a batch API
because none of the available options is unambiguously better:

| Pattern | When to reach for it | Where parallelism comes from |
|---|---|---|
| `for img in images: extract(img, ...)` | Few images, ad hoc scripts | Sequential — fine if N is small |
| `torch.vmap(extract_fn)(batch)` | Same-size images, same geometry, want a single kernel launch | Torch's vmap engine fuses the per-image calls; works on CPU and CUDA |
| `Dataset` + `DataLoader(num_workers=K)` with `Patchify` in the transform | The general case for training pipelines | K worker processes, each calling `Patchify(image)` on one image at a time. Pipeline overlap (load + transform + train) handled by torch |
| Manual `multiprocessing.Pool` / `concurrent.futures` | When `DataLoader` doesn't fit (e.g., offline preprocessing into a Cache) | OS process pool; each process imports patchkit independently |

The pattern PatchKit cares about is the third one — `Patchify`
exists *exactly* to slot into that pipeline ([ADR
0002](ADR/0002-patchify-transform.md)).

### 2.3 What PatchKit will *not* do

- **No internal threading or multiprocessing.** Threads inside a
  primitive surprise callers who already use a `DataLoader` or
  `Pool`; multiprocessing breaks notebook UX. The lib stays
  pure-Python single-threaded at the call site; parallelism is
  provided by the torch operators and by the pipeline above.
- **No batched `extract(images: (B, C, H, W))`.** Different images
  can have different shapes (and therefore different `L`), so the
  output would need padding or a list-of-tensors — both leak
  complexity into a primitive that does not need it. `vmap` works
  when shapes match; the loop works always.
- **No `Patchify` as `nn.Module`.** It would imply parameter
  registration, `.to(device)`, training/eval modes, gradient hooks —
  none of which `Patchify` honors. The callable class is what
  `torchvision.transforms.v2` itself does for stateless transforms.

### 2.4 Performance triage when extract is "slow"

If a profiling pass shows `extract` is hot, the order to investigate
is:

1. Are you re-extracting from the same image multiple times?
   → cache the result with `Cache` (§4 below).
2. Are you on CPU with float32 and large `(H, W)`?
   → run on CUDA. `extract` preserves device.
3. Are you calling `extract` in a Python loop on many images of the
   same size?
   → switch to `torch.vmap` or push the loop into a
   `DataLoader(num_workers=...)` pipeline with `Patchify`.
4. Is `unfold` itself the bottleneck?
   → that is a torch-level concern; profile the call and report
   upstream if necessary. PatchKit cannot beat the kernel it
   delegates to.

---

## 3. What "one image at a time" really means

The binding scope (THEORY §0) reads "one image at a time", which is
easy to misread as "one call at a time". The precise statement:

- Every public function takes **one** `image: Tensor[C, H, W]` (or
  one `image: PIL.Image`, or one `(lr, hr)` pair).
- The output may be a tensor *batch* — `extract` returns
  `Tensor[L, C, ph, pw]` — but that batch is patches *of that single
  image*, not images.
- Multiple images are someone else's problem.

The wheel ships nothing that walks a directory, indexes a file, or
opens a `Dataset`. If a downstream consumer wants a
`PatchExtractionDataset`, they compose `Patchify` with their own
`Dataset` in roughly five lines.

---

## 4. Grey areas, discussed

### 4.1 `Cache` — is it really core?

**Argument it isn't:** caching is a generic concern, the cache only
handles bytes, and any user could plug `joblib.Memory` or `shelve`
or a custom `dict + pickle.dump` instead.

**Argument it is:** the cache solves a problem the rest of PatchKit
creates (re-running the same `resize` or `extract` over an immutable
image is wasted work), it is < 250 lines, it ships with
content-addressing semantics designed around the lib's other
primitives (key includes the version, parts can be image
fingerprints), and it handles the OneDrive / antivirus write-race
that PatchKit's own development surfaced.

**Decision:** core, but explicitly bytes-only. PatchKit does not
provide `cached_resize` or `cached_extract` helpers — those would
re-introduce orchestration concerns. If you want cached resize, two
lines:

```python
key = c.key_for("resize", image_bytes_or_fingerprint, target_size, backend)
cached = c.get(key)
if cached is None:
    out = resize(image, target_size, backend)
    c.put(key, torch.save(out, ...))  # caller picks serialization
```

The cache does not know about images, tensors, or PIL. That ignorance
is the feature.

### 4.2 `Patchify` — is it surface area creep over `extract`?

It is a 30-line callable wrapper around `extract` with eager
validation and a `__repr__`. Without it, every Compose user writes
`lambda img: extract(img, 4, 2)`. Two ways to do the same thing — one
of which is a function for one-off scripts, the other a class for
pipelines — is a real surface area cost. We accepted it because the
class-in-a-pipeline pattern is the dominant integration mode for any
real consumer ([ADR 0002](ADR/0002-patchify-transform.md)).

### 4.3 `num_patches` and `tilings` — test helpers in disguise?

They were initially proposed because the test suite needed to
parametrize over valid geometries. They survived as public API
because:

- `num_patches` is just the formula from §1, and callers were
  already re-deriving it inline ("how many patches will I get?").
- `tilings` answers a question every new user asks ("what patch
  sizes fit my 28×28 image without overlap and without truncation?")
  and the cost is one tiny module.

They take no tensors, allocate nothing, and have no I/O. They
fit the "one image at a time, no datasets" charter because they take
shape ints, not images.

### 4.4 `image_id` in `PatchMeta`

A grey case in the opposite direction. PatchKit does not manage
images, so it does not name them. `image_id` is a free-form opaque
string the caller passes in and gets back in every `PatchMeta`. It
crosses the lib's boundary only to make round-tripping debugging
information convenient. PatchKit treats it as metadata and never
parses, validates, or persists it.

---

## 5. Pointers

- [`THEORY.md`](THEORY.md) §0 — binding scope (the short version).
- [`THEORY.md`](THEORY.md) §9 — the per-API condition contract
  (Accepts / Rejects / Out of scope), one row per primitive.
- [`USAGE.md`](USAGE.md) — usage walkthrough with real outputs.
- [`AUXILIARY.md`](AUXILIARY.md) — documents the bench tooling
  (`tests/_datasets.py`, `lab/`, dataset / output conventions on
  `Z:\`). Never mixed with the core docs.
- [`ADR/0001-patch-extraction-api.md`](ADR/0001-patch-extraction-api.md)
  — `extract` as a pure function (rejection of class-based extractor).
- [`ADR/0002-patchify-transform.md`](ADR/0002-patchify-transform.md)
  — `Patchify` as a callable wrapper for `Compose`.
