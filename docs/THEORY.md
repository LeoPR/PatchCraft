# PatchForge — Theory Notes

Working document. The goal is to distill the useful theory from the reference implementations in [`../archive/`](../archive/) into a single place **before** writing new code. Keep this document as the source-of-truth for design decisions.

> Every section should end with a paragraph titled **Design decision** stating what PatchForge will actually implement and why.

## 0. Scope (binding)

PatchForge operates on **one image at a time**: a `(C, H, W)` tensor in, a derived tensor out. No batching across images, no dataset abstractions, no dataloader integration.

The lib is the *car*; this repo also contains the *track and pit crew* (`tests/`, `lab/`, `tests/_datasets.py`) that prove the car works. Anything that needs a dataset, a labeled subset, a download, or batching belongs to the track, not the car. The wheel that ships via `pip install patchforge` contains only `src/patchforge/`.

The car must be **acoplável** to other people's torch pipelines: `Patchify` (see §1) is a `torchvision.transforms`-style callable so callers can drop PatchForge into `Compose([..., Patchify(...), ...])` and get `DataLoader` worker parallelism for free. The lib gives them the primitive; they own the pipeline.

Multi-image batching is **out of scope**, not under-specified: callers use a Python loop, `torch.vmap`, or a `DataLoader`. The grey-area "maybe if trivially cheap" was resolved against batching — patch counts vary per image (different `H`, `W` give different `L`), so any batched API would need padding or list-of-tensor outputs, both of which leak complexity into a primitive that has no reason to carry it.

## 1. Patch extraction

**Topic.** Given an image of shape `(C, H, W)`, a patch size `(ph, pw)`, a stride `(sh, sw)` and an optional dilation `d`, define:
- `num_patches_h`, `num_patches_w` as functions of the inputs.
- The mapping `(patch_index) → (row, col)` and its inverse.
- The extraction operator as a single call to `torch.nn.functional.unfold`.

**Counts and indexing.**
- `num_patches_h = floor((H − d·(ph − 1) − 1) / sh) + 1`; analogous for width.
- Row-major ordering: patch `k` is at `(row, col) = (k // num_patches_w, k % num_patches_w)`.
- The top-left pixel of patch `k` is at `(row · sh, col · sw)` in the original image.

**Boundary conditions.** Truncation, never padding. Pixels in the trailing rows/columns that do not fit a full patch are dropped. PatchForge does not synthesize data; callers who need full coverage should pad the image themselves before extracting.

**Memory layout.** `F.unfold(x.unsqueeze(0), (ph,pw), dilation=d, stride=(sh,sw))` yields shape `(1, C·ph·pw, L)`. Reshape as `.view(C, ph, pw, L).permute(3, 0, 1, 2).contiguous()` to obtain `(L, C, ph, pw)`. Device and dtype are preserved from the input.

**Design decision.** `patchforge.extract(image, patch_size, stride, dilation=1) -> Tensor[L, C, ph, pw]` is a pure function built on `F.unfold`. `image` must be `(C, H, W)`; batching is explicit (call in a loop or `vmap`) and is not part of the v0.1 signature. `patch_size` and `stride` accept `int` (square) or `(int, int)`. Truncation is the only boundary policy. When no patch fits, we return an empty tensor `Tensor[0, C, ph, pw]` rather than raising — callers decide. No caching inside the function; caching is a separate concern (see §4). Dilation is supported here even though reconstruction rejects it (see §2). See also [ADR 0001](ADR/0001-patch-extraction-api.md) for the API rationale.

**Composability with torch transforms (Patchify).** ADR 0002 adds a thin callable companion: `patchforge.Patchify(patch_size, stride, dilation=1)` is a class with `__call__(image) -> Tensor[L, C, ph, pw]` that delegates to `extract`. It exists to slot into `torchvision.transforms.Compose([..., Patchify(4, 2), ...])` without forcing callers to write a lambda — lambdas are not repr-friendly and they skip eager validation. `Patchify` validates the geometry in `__init__` (so a bad `patch_size` fails when the pipeline is built, not when the first batch arrives), and carries **only** the geometry ints (`__slots__`, no `__dict__`, no cache, no buffer). The function is the contract; the class is a convenience. Same shape contract, same dtype/device preservation, same truncation policy.

## 1.5 Pre-flight geometry helpers

**Topic.** Before extracting patches, callers often need to answer questions that depend only on the *shape* of the image, never on its pixels:

- "How many patches will I get for this `(H, W, ph, pw, sh, sw, d)`?" — for memory planning, for shape assertions, for filling a progress bar before allocating.
- "What patch sizes tile my image cleanly, with no overlap and no truncation?" — for picking a geometry whose round-trip is bit-exact by construction (§2 exact regime).
- "Which `(p, s)` pairs cover my image without truncation but with overlap?" — for picking a geometry whose round-trip is weighted-exact (§2 overlap regime).

The naïve way is to materialize patches with `extract` and check `.shape[0]`, or to try every plausible geometry by brute force. Both waste cycles on operations whose answers are arithmetic.

**Counts.** `num_patches(image_shape, patch_size, stride, dilation=1) -> (num_h, num_w)` is the formula from §1, exposed as a function. Accepts `(H, W)` or `(C, H, W)` (channels are not used). Returns `(0, *)` or `(*, 0)` when the effective patch does not fit on that axis — same boundary behavior as `extract` returning `Tensor[0, C, ph, pw]`.

**Enumeration.** `tilings(image_shape, *, allow_overlap=False, min_patch_size=2, max_patch_size=None) -> list[TilingSpec]` walks the square patch sizes from `min_patch_size` to `max_patch_size` and emits every geometry that **fully covers** the image:

- *Exact tiling* (always emitted): `patch_size == stride` and `H % p == 0` and `W % p == 0`. The clean grid case, bit-exact round-trip.
- *Overlap with clean edges* (emitted when `allow_overlap=True`): `stride < patch_size` with `(H - p) % s == 0` and `(W - p) % s == 0`. The last patch's last pixel lands on the image edge; full coverage with shared pixels in the middle.

Truncated geometries (where the last patch falls short of the edge) are deliberately not emitted — the function answers "which geometries are sound by construction?", not "which geometries `extract` will accept" (`extract` accepts any positive `(p, s, d)`).

**Worked example (28×28 MNIST).** Divisors of 28 ≥ 2 are `{2, 4, 7, 14, 28}`, so `tilings((28, 28))` returns exactly 5 specs: `(p=2, total=196)`, `(p=4, total=49)`, `(p=7, total=16)`, `(p=14, total=4)`, `(p=28, total=1)`. With `allow_overlap=True` the set grows to 100 specs.

**Dilation.** `tilings` always emits `dilation=(1, 1)` in v0.1 — dilated full-coverage tilings are a non-trivial enumeration (the patch footprint has gaps; you can interleave multiple dilated patches per pixel) and no consumer has asked for them yet. `num_patches` does honor `dilation` because the formula already does.

**Design decision.** Two pure functions and one `NamedTuple`: `num_patches`, `tilings`, `TilingSpec`. No tensors, no images, no dataset abstractions — only ints in, ints out. They live in `src/patchforge/geometry.py` and are re-exported from `patchforge`. The `TilingSpec` fields are `patch_size`, `stride`, `dilation`, `num_patches`, `total_patches`, `overlap` — destructurable for tests, sortable, hashable. Square-only enumeration in v0.1; rectangular comes when a real consumer asks (the function signature accommodates it by always returning `(int, int)` tuples).

**Cross-resolution geometry (`scale_factor`, `paired_tilings`, `PairedTilingSpec`).** Super-resolution consumers (and any multi-resolution patch consumer) ask one more pre-flight question: "given two image shapes — the low-resolution input and the high-resolution target — what `(p, s)` on each side produces the same number of patches with corresponding image regions?". The same arithmetic as `tilings`, layered with the integer-scale invariant from §3:

- `scale_factor(lr_shape, hr_shape) -> int | None`: returns `k` such that `hr.shape[-2:] == (k * lr.shape[-2], k * lr.shape[-1])`, or `None` when no such positive integer exists. Lets a consumer discover the scale factor from data instead of hard-coding it before calling `pair`.
- `paired_tilings(lr_shape, hr_shape, *, allow_overlap=False, ...)`: for each LR tiling that `tilings(lr_shape)` would emit, derives the matching HR tiling by multiplying patch size and stride by the scale factor. Returns a list of `PairedTilingSpec(lr, hr, scale_factor)`. By construction every entry has identical `total_patches` on both sides and aligned per-`k` regions — feed straight into `pair` with confidence.
- Worked example: `paired_tilings((14, 14), (28, 28))` returns three pairs corresponding to LR patch sizes `{2, 7, 14}` (the divisors of 14 with `min_patch_size=2`), each paired with HR patches twice the size, all preserving patch count: `(p_lr=2, p_hr=4, total=49)`, `(p_lr=7, p_hr=14, total=4)`, `(p_lr=14, p_hr=28, total=1)`.

These helpers live alongside `tilings` and are tensor-free; everything they do reduces to shape arithmetic.

## 1.6 Patch comparison metrics

**Topic.** Once `extract` + `reconstruct` round-trips or `pair` produces aligned LR/HR tensors, consumers need to *measure* how close two patch tensors are. The set of useful pixel-level metrics for this is tiny and stable:

- MAE — `(a - b).abs().mean()`
- MSE — `((a - b) ** 2).mean()`
- max-abs-diff — `(a - b).abs().max()`
- PSNR (dB) — `10 * log10(max_value² / mse)`

Every consumer either re-implements these or imports them from some scattered utility module. Different consumers often pick slightly different reductions (axis choice, dtype promotion, what to do when MSE is zero) — divergence breeds bugs. PatchForge ships the canonical reductions so that "did the model do better?" has one answer at the lib level.

**Per-patch vs over-the-stack.** Both are useful. A model trainer wants the scalar PSNR over an entire batch (early stopping, logging); a researcher wants per-patch PSNR (rank patches by reconstruction quality, identify failure modes). Two distinct shapes; two functions.

**Dtype handling.** Internally `patch_metrics` promotes to `float64` for the scalar accumulation (one number per call, cost is irrelevant, precision matters). `per_patch_mse` and `per_patch_psnr` keep input dtype (per-patch values are themselves a tensor; consumer chooses precision via input). PSNR returns `+inf` when MSE is zero — mathematically correct; no clamp tricks that produce a finite "very large" value the user has to reverse-engineer.

**What this section deliberately does not ship.** SSIM, MS-SSIM, LPIPS, FID, any windowed or learned metric. Each depends on parameters (window size, data range, pre-trained network) that PatchForge cannot pick on behalf of consumers, and mature standalone packages exist (`pytorch-msssim`, `lpips`). Adding them here would force PatchForge to pull bigger deps and to bless one parameterization over others.

**Design decision.** Three pure functions in `src/patchforge/metrics.py`, re-exported from `patchforge`:
- `patch_metrics(a, b, *, max_value=1.0) -> dict[str, float]` — scalar reduction over the whole tensor, dtype-promoted internally, returns plain Python floats so the result is JSON-serializable.
- `per_patch_mse(a, b) -> Tensor[L]` — strict `(L, C, h, w)` inputs, returns one value per leading-axis entry.
- `per_patch_psnr(a, b, *, max_value=1.0) -> Tensor[L]` — same as MSE shape; identical patches yield `+inf` (`torch.where(mse == 0, inf, ...)`, not a clamp).

Strict input checks: shape mismatch, dtype mismatch, and device mismatch all raise `ValueError`. The lib does not coerce — caller normalizes upstream. The functions take patches, but the math doesn't care whether the inputs are patches or full images; the type hint mentions patches because that is the canonical use case.

## 2. Reconstruction

**Topic.** Inverse of extraction via `torch.nn.functional.fold`. Two regimes:
- **Exact** — when `sh == ph` and `sw == pw` and `d == 1`: patches tile the image; reconstruction is a cheap copy (weights are all ones).
- **Weighted overlap** — when `sh < ph` or `sw < pw`: patches overlap; reconstruction must divide by the overlap count map (from `fold` of an all-ones tensor with the same geometry).

**Worked example (`ph = pw = 4, sh = sw = 2`).** Every interior pixel is covered by 4 patches (2 row overlaps × 2 col overlaps); edge pixels are covered by 2; corners by 1. The count map produced by `fold(ones_like(patches_flat), output_size=(H,W), kernel=(4,4), stride=2)` encodes exactly these weights. Dividing the summed pixel contributions by this map recovers the original image bit-exactly for fractional pixel values, and within one ULP for float32 noise.

Visualizing the count map along one axis (image cols 0..7, `ph=4`, `sh=2`):

```
   col:     0  1  2  3  4  5  6  7
   patch0:  x  x  x  x
   patch1:        x  x  x  x
   patch2:              x  x  x  x
   count:   1  1  2  2  2  2  1  1   <-- 2-D fold = outer(this, this)
```

**Dilation.** `F.fold` does not support dilation in the same way as `F.unfold`: for `d > 1` the patch footprint skips pixels, and `fold` would deposit the sparse contributions into a canvas that is not the image. Rather than building a custom scatter, we refuse `dilation != 1` at reconstruction time with a clear `ValueError`. Callers who extracted with dilation are expected to consume patches directly (e.g., as features) and not round-trip.

**Stride maior que patch (lacunas).** Quando `sh > ph` ou `sw > pw`, o grid pula pixels: a soma `fold(...)` tem zeros nessas posições, e qualquer divisão (incluindo pelo clamp `min=1e-6`) produz pixels com valor arbitrário — síntese de dado, exatamente o que §1 proíbe ao escolher truncamento como única política de borda. Recusamos a condição com `ValueError` logo na entrada de `reconstruct`. Consumidores que querem features esparsas (kernels, classificadores) usam apenas `extract`, onde `stride > patch_size` é aceito sem objeções, e não tentam round-trip.

**Design decision.** `patchforge.reconstruct(patches, image_shape, stride, dilation=1) -> Tensor[C, H, W]` uses `F.fold` followed by division by the overlap-count map (computed once, same-geometry fold of ones). Raises `ValueError` when `dilation != 1` **ou quando `sh > ph` ou `sw > pw`** (cobertura parcial é proibida). `image_shape` is `(C, H, W)` and must match the geometry implied by the patch grid; inconsistent shapes raise `ValueError`. The count map clamp (`min=1e-6`) existe apenas para absorver ruído float em pixels totalmente cobertos — nunca para mascarar buracos de cobertura. Output dtype matches input.

## 2.5 Stitching modified patches

**Topic.** `reconstruct` answers "given the original patches, give me back the image." It is the inverse of `extract`, and it makes a strong implicit assumption: every patch covering a pixel agrees on that pixel's value (true for patches that came straight from `extract` on the same image). When patches have been *modified* — model output, denoised, super-resolved, hand-edited — that assumption fails, and uniform averaging shows the disagreement as visible seams along patch boundaries.

The standard trick is to weight each patch's contribution by a 2-D window whose value is large at the patch center and small (or zero) at the patch edges, so that pixels closer to a patch's center "trust" that patch more than pixels far from it. PatchForge ships three window kernels: `uniform`, `hann`, `gaussian`.

**Math.** Let `w(i, j)` be the window kernel of shape `(ph, pw)`. For each output pixel `(x, y)`, summing over every patch `k` covering `(x, y)`:

  `numerator(x, y) = Σ_k  patch_k(i_k, j_k) · w(i_k, j_k)`
  `denominator(x, y) = Σ_k  w(i_k, j_k)`
  `output(x, y) = numerator(x, y) / denominator(x, y)`

For *unmodified* patches, every `patch_k(i_k, j_k)` equals `img(x, y)` (definition of `extract`), so the numerator factors as `img(x, y) · denominator(x, y)`, and the output is `img(x, y)` exactly — round-trip is preserved for any kernel as long as the denominator at `(x, y)` is positive. For *modified* patches, contributions are weighted by how central `(x, y)` is to each contributing patch — seams are attenuated.

**Implementation.** Two `F.fold` calls: one on the weighted patches (numerator), one on the kernel replicated across the `L` patch slots (denominator). Identical geometry to `reconstruct`'s count-map fold; same `clamp(min=1e-6)` on the denominator to absorb float noise.

**Window kernels.**

- **`uniform`** — `w ≡ 1`. Numerator becomes the same as `reconstruct`'s fold; denominator becomes the count map. Mathematically equivalent to `reconstruct`. Provided so the API is one function with a parameter instead of two functions with a hidden choice.
- **`hann`** — separable Hann (`outer(hann(ph), hann(pw))`). Center weight is 1, edge weight is 0. Strong seam suppression; cheapest to compute. **Caveat:** image-corner pixels covered only by patches whose `w` at that relative position is zero have numerator and denominator both ≈ 0 — the `clamp(min=1e-6)` divisor makes them zero in the output. This shows up most obviously at `stride == patch_size`, where the four image corners go black. With overlap (`stride < patch_size`) the artifact shrinks to the outermost pixel only on each side.
- **`gaussian`** — separable Gaussian with `sigma = max(1, min(ph, pw) / 4)` centered at the patch midpoint. Weight is non-zero everywhere, so there is no corner artifact, at the cost of weaker seam suppression than Hann.

Visualizing the three 2-D kernels at `patch_size=4` (`+` = full weight, `X` = high, `o` = medium, `.` = zero or near-zero):

```
   uniform        hann           gaussian
   + + + +        . . . .        . o o .
   + + + +        . X X .        o X X o
   + + + +        . X X .        o X X o
   + + + +        . . . .        . o o .
```

**Why a separate function and not a parameter on `reconstruct`?** Because the contracts are different. `reconstruct` is bit-exact for unmodified patches and rejects anything that would force interpolation; `stitch` accepts modified patches and explicitly does interpolated blending. Adding a `weight=` parameter to `reconstruct` would conflate "I want my image back" with "I have model output and need to blend." Two functions, one charter each, no surprise behavior when a kwarg is forgotten.

**Why float-only?** Window kernels are float-valued by construction (`hann`, `gaussian` produce values in `[0, 1]`). Multiplying integer patches by a float kernel would either silently quantize the output or implicitly promote to float, neither of which is a contract a primitive should carry. Reject non-float input with a clear `ValueError` and let the caller convert.

**Design decision.** `patchforge.stitch(patches, image_shape, stride, *, weight="uniform"|"hann"|"gaussian", dilation=1) -> Tensor[C, H, W]` in `src/patchforge/stitch.py`. Lives next to `reconstruct`; uses the same `F.fold` geometry; shares the same rejections (`dilation != 1`, `stride > patch_size`, ndim check, grid-consistency check). Adds: float-only patches, weight-kind validation. Output dtype and device preserved. The `weight="uniform"` path is mathematically equivalent to `reconstruct` (validated by bit-exact equality test on no-overlap and `allclose` on overlap).

## 3. LR ↔ HR pairing

**Topic.** Given a scale factor `r` and LR/HR shapes related by `HR = r · LR`, define patch sizes and strides on both sides so that the *k*-th LR patch corresponds to the *k*-th HR patch (same `k`). Invariant:
- `ph_hr = r · ph_lr` and `sh_hr = r · sh_lr` (similar for width).
- Then `num_patches_lr == num_patches_hr`, and the coordinates of patch *k* on both sides map to the same image region in pixel space.

**Coverage invariant (sketch).** Given `H_hr = r · H_lr`, `ph_hr = r · ph_lr`, `sh_hr = r · sh_lr`, and dilation 1 everywhere:

  `num_h_lr = floor((H_lr - ph_lr) / sh_lr) + 1`
  `num_h_hr = floor((H_hr - ph_hr) / sh_hr) + 1 = floor((r·H_lr - r·ph_lr) / (r·sh_lr)) + 1 = floor((H_lr - ph_lr) / sh_lr) + 1`

So `num_h_lr == num_h_hr` (and analogously for width) exactly when the scale factor is an integer. The top-left of LR patch `k` is at `(row · sh_lr, col · sw_lr)`; multiply by `r` to get the HR patch origin — same image region, different resolution.

**Truncation.** When `(H_lr − ph_lr) % sh_lr != 0`, the trailing LR rows that don't fit a full patch are dropped. Because of the integer scaling, the same trailing HR rows are also dropped — the two grids stay aligned. The user loses a strip on the right/bottom of both resolutions. If full coverage is required, pad the LR image *before* extracting (and the HR image correspondingly) — PatchForge does not pad implicitly.

**Recommended defaults.** For tiling (no overlap): `stride_lr = ph_lr`, `stride_hr = ph_hr = r · ph_lr`. For 2× overlap training data: `stride_lr = ph_lr // 2`.

**Design decision.** `patchforge.pair(lr_image, hr_image, lr_patch_size, scale_factor, stride, *, image_id=None) -> PatchPair`. `scale_factor` must be a positive `int` (non-integer scale is rejected — irrational alignment has no clean semantics). `lr_patch_size` and `stride` are `int` or `(int, int)`; HR `patch_size` and `stride` are derived as `scale_factor * lr_*`. `dilation=1` is fixed (not exposed) because dilation breaks the reconstruction story.

The return type is `PatchPair`, a frozen `@dataclass(slots=True)` with three fields: `lr_patches` (`Tensor[L, C, ph_lr, pw_lr]`), `hr_patches` (`Tensor[L, C, ph_hr, pw_hr]`), `metas` (`tuple[PatchMeta, ...]` of length `L`). This deviates from the earlier "iterator of tuples" sketch — tensors are what downstream code actually wants (a batch of LR patches and the corresponding batch of HR patches), iteration over the dataclass is one `zip(...)` away, and the materialized form is cheap because both `extract` calls are already eager. `PatchMeta` is `@dataclass(frozen=True, slots=True)` holding `patch_index`, `row`, `col` (in LR coords; multiply by `scale_factor` for HR), `lr_patch_size`, `hr_patch_size`, `image_id` — small, CPU-only, never gets pushed to GPU. See §7 for why a dataclass over a dict/tensor.

LR and HR must agree on dtype and device; mismatch is rejected (caller normalizes upstream — implicit conversion would surprise people training mixed-precision pipelines).

## 4. Cache semantics

**Topic.** Processed datasets (resized / quantized / paired) are expensive to build; cache them on disk keyed by a hash of the configuration. Reference implementations:
- `archive/PatchHub/src/patchhub/cache.py` — mmap-friendly on-disk cache with zstd.
- `archive/QSVM_patchkit/patchkit/processed.py` — cache bundle with labels.

**Cache key.** The key is a SHA-256 over a canonical JSON serialization of every input that affects the output. For a resized tensor that is: the image fingerprint (SHA-256 of raw bytes), target size, backend name, resample filter, and a `version` integer owned by the function. The `version` field is the invalidation lever: when the algorithm changes, bump it and the old cache becomes inaccessible by construction (no delete needed).

**Invalidation.** No TTL, no mtime check. Cache entries are immutable; a changed configuration produces a different key and a fresh entry. Stale entries accumulate on disk — cleanup is a manual maintenance task, not a runtime concern. This is safe only because inputs are content-addressed: the same config and the same image bytes always yield the same result.

**Serialization format.** `torch.save` to a `bytes` buffer, then zstd-compress when `zstandard` is installed (level 3), else write raw. File layout: one file per entry, filename = first 16 hex chars of the key, stored under `<cache_dir>/<prefix>/`. A tiny sidecar JSON per entry stores the full key, version, and content checksum — enough to detect truncated writes without scanning the whole payload.

**Concurrency.** Single-writer, multi-reader. PatchForge does not implement locking. If two processes race on the same key, the second writer wins (atomic rename from a `*.tmp` file). No lockfiles — OneDrive-sync weirdness has already shown that filesystem locks are unreliable here.

**Robustez a write races (OneDrive, antivírus, indexador).** Em diretórios sincronizados com OneDrive — e por extensão qualquer pasta varrida por antivírus ou Windows Search — `os.rename`, `open(..., "wb")` e `os.replace` falham esporadicamente com `PermissionError` (errno 13, Windows error 5) enquanto um agente externo segura o handle por alguns ms. Já vimos esse comportamento em `uv lock` rodando contra `uv.lock` em pasta OneDrive. `Cache.put` envolve o rename atômico (e o `open` do `*.tmp`) num loop de retry com backoff exponencial: até 5 tentativas, esperas de `0.25, 0.5, 1.0, 2.0, 4.0` segundos. Depois disso, a exceção original sobe — falha persistente é problema legítimo (disco cheio, permissões reais, OneDrive offline). `Cache.get` aplica o mesmo wrap em `open(..., "rb")` com 2 tentativas apenas — leitura raramente é o lado bloqueado e falhar rápido aqui é melhor que esconder cache miss real.

**Design decision.** One cache module with one public class: `patchforge.Cache(root, namespace, version=1)` (parâmetro renomeado de `dir` para evitar sombrear o builtin). Methods: `get(key) -> bytes | None`, `put(key, bytes)`, `key_for(*parts) -> str`. `root` inexistente é auto-criado (`mkdir(parents=True, exist_ok=True)`) — ergonomia. Writes envolvem o backoff descrito acima; reads usam variante curta. Higher-level helpers (`cached_resize`, etc.) compose `Cache` with a function — they are thin wrappers, not inheritance. No pluggable backends (PatchHub's `memory | shelve | diskcache | hybrid` matrix is overkill for our use cases). No mmap in v0.1 — deferred until a profile shows memory pressure. The `zstandard` dep is an `optional-extra` (`[cache]`), and the absence of it falls back to uncompressed writes transparently.

## 5. Resize backends

**Topic.** Image resize is surprisingly opinionated:
- **PIL** — battle-tested, supports BICUBIC/LANCZOS/etc., CPU-only, convenient.
- **torch** — `F.interpolate`, runs on GPU, different bicubic filter than PIL.

**Non-identity.** PIL's BICUBIC and torch's `F.interpolate(mode='bicubic')` use different kernel parameters (PIL uses `a = -0.5`, torch uses `a = -0.75` with `align_corners=False`) and produce measurably different output. For a 28×28 MNIST image resized to 14×14 the per-pixel difference is typically within ±3 / 255 but can exceed that at edges. **This matters for reproducibility.** A dataset pre-processed with PIL cannot be compared head-to-head against one pre-processed with torch even when the "same" algorithm name is used.

**When to prefer each.**
- **PIL**: canonical dataset generation (reproducibility is paramount, result stays on CPU anyway).
- **torch**: in-training augmentation where the tensor is already on GPU and the extra filter divergence does not matter.

**Design decision.** One function: `patchforge.resize(image, target_size, backend="pil", resample=None)`. `target_size` is a `(H, W)` tuple — no size-spec DSL. `image` may be `PIL.Image` or `Tensor[C, H, W]`. Output type matches input (PIL in → PIL out, Tensor in → Tensor out), regardless of backend. `resample` defaults: `LANCZOS` for PIL, `bilinear` for torch (chosen because bicubic divergence is nastiest to debug; bilinear's difference from PIL's BILINEAR is small). Cross-backend conversions (PIL ↔ Tensor) go through a normalized float32 `[0, 1]` intermediate. No auto-selection: callers pick the backend explicitly and own the consequences.

## 6. Quantization (optional)

**Topic.** Reducing color depth (binary, k-level) before extraction. Reference: `archive/QSVM_patchkit/patchkit/quantize.py` — uniform, k-means, Otsu, Floyd–Steinberg dither.

**Does it belong here?** PatchForge's charter (see README §Scope) is patch infrastructure. Quantization is a *consumer-specific preprocessing step*: QSVM needs it because small bit-depths stabilize kernel computations; other consumers may not want it, or may want a different family (e.g. vector quantization). Keeping it inside PatchForge means every consumer pays the cost of evaluating whether the in-package implementation matches their needs, and updates force coordinated releases.

**Design decision.** **Out of scope for v0.1.** Quantization is deferred. When a pattern emerges across multiple consumers, revisit as either (a) a companion package (`patchforge-quant`), or (b) a plug-in hook point in the extraction pipeline. For now, callers that need quantization apply it *before* calling `patchforge.extract` — the `(C, H, W)` tensor they pass in is whatever they want patches of. The archive implementations stay in `archive/` as reference material for whoever builds the companion.

## 7. Resolved questions

- **`PatchPairDataset` vs decoupled primitives?** Decoupled. v0.1 ships `extract`, `Patchify`, `reconstruct`, `pair`, `resize`, `Cache` as independent pieces. Consumers that want a `torch.utils.data.Dataset` compose them themselves. This mirrors the M2–M5 ordering in the roadmap and avoids the QSVM_patchkit problem where `SuperResPatchDataset` baked in assumptions (ProcessedDataset, specific caching, specific labels) that the next consumer does not share.
- **`meta` shape — dict, structured tensor, or dataclass?** Dataclass (`PatchMeta`). Dicts lose type info; structured tensors force CPU→GPU transfers for fields that are never used on device (image id, patch index). Metadata stays on CPU; the pixel payload is the only thing that should move to GPU. `@dataclass(frozen=True, slots=True)` keeps it cheap.
- **Minimum torch?** `>=2.6` as per `pyproject.toml`. Sticking with this until a feature we want (e.g., `vmap` improvements) forces a bump.
- **Label-stratified subsets** (was §7). Moved to `tests/_datasets.py::label_subset(labels, n_per_label, seed)` as a pure function over a labels sequence. Not part of the public API: it operates on dataset-level concerns (which §0 declares out of scope for the core), and is used only by tests and `lab/` scripts.

## 8. Remaining open questions

- **Channels-first only, or also channels-last?** Currently every API assumes `(C, H, W)`. A `channels_last=True` flag may be desirable for interop with CV libraries that default to HWC. Defer to a real consumer request.
- **Tensor-vs-PIL in `extract`/`reconstruct`.** Extraction is tensor-only. Should we allow PIL input with an internal conversion? Current lean: no — converting before `extract` is one line, and mixed-type APIs hide cost.
- **Batched extraction.** Resolved against by §0 (one image at a time). Reopen only if a benchmark shows the per-call overhead of `extract` is the bottleneck in a real consumer's loop *and* `torch.vmap` doesn't already solve it.

## 9. Contrato de condições suportadas

Esta seção consolida, por API, as condições que v0.1 deve **aceitar**, **rejeitar** (com `ValueError` e mensagem explícita) e tratar como **fora de escopo** (não implementado, documentar como tal). Serve de fonte do plano de testes: cada item "Aceita" vira teste positivo; cada "Rejeita", teste negativo (`pytest.raises(ValueError)`). Onde esta seção diverge dos parágrafos "Design decision" de §1–§6, §9 é a verdade — ajuste os outros parágrafos no marco que implementar a API correspondente.

### 9.1 `extract(image, patch_size, stride, dilation=1)` and `Patchify(patch_size, stride, dilation=1)`

`Patchify(...)(image)` is the callable form and delegates to `extract` — same contract, plus eager geometry validation at `__init__`.

**Aceita:**
- `image` é `Tensor` 3D `(C, H, W)` com `C ≥ 1` arbitrário; dtype suportado por `torch.nn.functional.unfold` (todos os float — `float16/float32/float64/bfloat16` — e integer ≥ 16 bits no caminho CUDA; em CPU, `uint8` não é suportado por `im2col_cpu` e o caller deve converter pra float antes); CPU ou CUDA (preservados na saída).
- `patch_size`, `stride`, `dilation` como `int` (quadrado) ou `(int, int)` com valores positivos.
- Patch maior que imagem → retorna `Tensor[0, C, ph, pw]` (não levanta).
- `stride > patch_size` (grid esparso com lacunas — válido para features que não fazem round-trip; ver §9.2 para a contrapartida em `reconstruct`).
- `dilation ≥ 1`.
- Tensor não-contíguo (paga `.contiguous()` interno no reshape).

**Rejeita (`ValueError`):**
- `image.ndim != 3` (2D ou 4D).
- Qualquer dimensão de `patch_size`, `stride`, `dilation` ≤ 0.

**Fora de escopo v0.1:**
- Batched input `(B, C, H, W)` — caller faz loop ou usa `torch.vmap`.
- `PIL.Image` ou `numpy.ndarray` — converter antes (uma linha).
- Layout channels-last `(H, W, C)` — converter antes.
- Devices não-CUDA acelerados (MPS, XPU) — provavelmente funcionam via torch, mas não testados.

### 9.2 `reconstruct(patches, image_shape, stride, dilation=1)`

**Aceita:**
- `sh ≤ ph` e `sw ≤ pw` (cobertura total ou overlap).
- `image_shape` igual a `(C, H, W)` consistente com a geometria implícita pelo `patches.shape[0]`.
- Qualquer dtype, **com aviso** de que `float16` perde precisão na divisão pelo count map.

**Rejeita (`ValueError`):**
- `dilation != 1`.
- `sh > ph` ou `sw > pw` (cobertura parcial — síntese de dado proibida).
- `image_shape` inconsistente com `patches.shape[0]` / `stride`.
- `patches.ndim != 4`.

**Fora de escopo v0.1:**
- Promoção automática float16 → float32.
- Output em PIL.

### 9.3 `pair(lr_image, hr_image, lr_patch_size, scale_factor, stride, *, image_id=None)`

Retorna `PatchPair(lr_patches, hr_patches, metas)` (frozen dataclass com `__slots__`).

**Aceita:**
- `lr_image` e `hr_image` são `torch.Tensor` 3D com mesmo `C`, dtype e device.
- `scale_factor ∈ ℤ⁺` (`int`, ≥ 1). `scale_factor=1` é válido (LR == HR; útil em testes).
- `hr_image.shape == (C, scale_factor * H_lr, scale_factor * W_lr)`.
- `lr_patch_size`, `stride` como `int` ou `(int, int)` positivos.
- `image_id: str | None = None` — metadado CPU-only, propagado a todos os `PatchMeta`.
- Patch maior que a imagem LR → ambos os lados retornam `(0, C, …)` e `metas == ()`.

**Rejeita (`ValueError` / `TypeError`):**
- `lr_image` ou `hr_image` não-tensor.
- `lr_image.ndim != 3` ou `hr_image.ndim != 3`.
- `scale_factor` não-int (incluindo `bool`), zero, ou negativo.
- `hr_image.shape` não bate com `scale_factor * lr_image.shape` em qualquer eixo espacial.
- Mismatch de canais entre LR e HR.
- Mismatch de dtype entre LR e HR.
- Mismatch de device entre LR e HR.
- `lr_patch_size` ou `stride` não-positivo / não-int.

**Fora de escopo v0.1:**
- `scale_factor` não-inteiro (alinhamento irracional).
- Pareamento N:1 (várias LRs degradadas pra mesmo HR).
- Channels-last.
- `dilation` customizado (fixo em 1 — round-trip exige).
- Coerção implícita LR→HR de dtype/device (caller normaliza antes).

### 9.4 `resize(image, target_size, backend="pil", resample=None)`

**Aceita:**
- `image` é `PIL.Image` em qualquer mode suportado pelo PIL.
- `image` é `Tensor[C, H, W]` em CPU.
- `image` é `Tensor` em CUDA **apenas se** `backend == "torch"`.
- `target_size = (H, W)` tupla de 2 ints positivos.
- `resample=None` → default por backend (LANCZOS pra PIL, bilinear pra torch).

**Rejeita (`ValueError`):**
- `Tensor` em CUDA com `backend == "pil"` (PIL não enxerga GPU — exigir `.cpu()` explícito do caller).
- `target_size` não é 2-tupla ou contém valor ≤ 0.
- `backend` não em `{"pil", "torch"}`.
- `resample` não suportado pelo backend escolhido (mensagem amigável, não trace interno).

**Fora de escopo v0.1:**
- Auto-seleção de backend.
- DSL para `target_size` (`"50%"`, `"min:256"`).
- Aviso de fator de escala extremo.

### 9.5 `Cache(root, namespace, version=1)`

**Aceita:**
- `root` inexistente → auto-criado (`mkdir(parents=True, exist_ok=True)`).
- Entries de qualquer tamanho que caiba em RAM (sem mmap em v0.1).
- Writes concorrentes (last-writer-wins via rename atômico + retry com backoff — §4).
- Reads concorrentes (sem lock).
- Write race transitório (OneDrive, antivírus, indexador) → retry transparente; só falha após esgotar 5 tentativas.
- Paths com caracteres não-ASCII (smoke-test obrigatório com `Acadêmicos/` no caminho).

**Sinaliza (não exceção):**
- Chave inexistente em `get` → retorna `None`.
- `version` diferente da configurada → cache miss transparente (chave hash é diferente por construção).

**Rejeita / propaga:**
- Sidecar checksum não bate com payload → `IOError` (entrada corrompida; sugere remoção manual).
- Disco cheio, permissão real negada após retries → `OSError` original.

**Fora de escopo v0.1:**
- `mmap` para entries gigantes.
- Locking inter-processo explícito.
- Backends pluggáveis (memory, shelve, diskcache).
- TTL ou limpeza automática.

<!-- §9.6 was `label_subset` until 2026-05-16; moved to tests/_datasets.py. The slot below is now `num_patches` + `tilings` (geometry helpers, §1.5). -->

### 9.6 `num_patches(image_shape, patch_size, stride, dilation=1)` and `tilings(image_shape, *, allow_overlap, min_patch_size, max_patch_size)`

**Aceita (`num_patches`):**
- `image_shape` is a 2-tuple `(H, W)` or 3-tuple `(C, H, W)`; channels ignored.
- `patch_size`, `stride`, `dilation` as `int` or `(int, int)` positive.
- Patch larger than image on any axis → that axis returns 0 (mirror of `extract` empty grid).

**Aceita (`tilings`):**
- `image_shape` as above; `allow_overlap: bool`; `min_patch_size: int >= 1`; `max_patch_size: int >= 1` or `None`.
- Always emits square geometries (`ph == pw`, `sh == sw`) with `dilation=(1, 1)`.
- Always emits *full-coverage* geometries only — exact tilings always; overlap-with-clean-edges if `allow_overlap=True`.
- Returns sorted list of `TilingSpec` (a `NamedTuple`).

**Rejeita (`ValueError`):**
- `image_shape` not a 2-tuple or 3-tuple of positive ints.
- Any of `patch_size`, `stride`, `dilation` non-positive or non-int.
- `min_patch_size <= 0`, `max_patch_size <= 0`, or `min_patch_size > max_patch_size`.

**Fora de escopo v0.1:**
- Rectangular `(ph, pw)` enumeration in `tilings` (`num_patches` already handles it on input).
- Dilated `tilings` (`dilation > 1` enumeration is non-trivial; defer until requested).
- Truncated-coverage enumeration (the contract is full-coverage only).
- Multi-image planning (e.g. "max patch that tiles every image in this list"); compose externally.

### 9.7 `scale_factor(lr_shape, hr_shape)` and `paired_tilings(lr_shape, hr_shape, *, allow_overlap, min_patch_size, max_patch_size)`

**Aceita (`scale_factor`):**
- `lr_shape`, `hr_shape` each a 2-tuple `(H, W)` or 3-tuple `(C, H, W)`.
- Returns `int >= 1` when `hr.shape[-2:] == (k * lr.shape[-2], k * lr.shape[-1])`; returns `None` otherwise (non-divisible, anisotropic, or LR larger than HR).

**Aceita (`paired_tilings`):**
- Same shape inputs as `scale_factor`.
- Same enumeration knobs as `tilings` (`allow_overlap`, `min_patch_size`, `max_patch_size`).
- Returns `list[PairedTilingSpec(lr, hr, scale_factor)]` where every entry has identical `total_patches` on both sides and patch `k` covers the same image region (HR coords are LR coords times `scale_factor`).
- `scale_factor=1` (LR == HR) is accepted; each LR tiling is paired with itself.

**Rejeita (`ValueError`):**
- `scale_factor`: malformed shape input or non-positive dimensions.
- `paired_tilings`: same as `scale_factor` plus `tilings`'s own validation (negative `min_patch_size`, `min > max`, etc.).
- `paired_tilings`: shapes not related by an integer scale factor (delegates to `scale_factor`'s `None` return; surfaces as an explicit `ValueError` because there's nothing useful to enumerate).

**Fora de escopo v0.1:**
- Non-integer (fractional) scale factors — `pair` rejects them too; the math has no clean alignment.
- N:1 enumeration (multiple LR shapes against one HR) — compose externally.
- Rectangular paired tilings — square-only inherited from `tilings`.

### 9.8 `patch_metrics(a, b, *, max_value=1.0)`, `per_patch_mse(a, b)`, `per_patch_psnr(a, b, *, max_value=1.0)`

**Aceita (`patch_metrics`):**
- `a, b` are `torch.Tensor` with **identical shape, dtype, and device**. Any shape works (single patch, patch stack, full image, batch).
- `max_value`: positive finite `float` (or `int`). Used only for PSNR (`10 * log10(max_value² / mse)`).
- Returns `dict[str, float]` with keys `mae`, `mse`, `max_abs`, `psnr_db`. Identical inputs yield `psnr_db == +inf`.
- Internal accumulation in `float64` regardless of input dtype (precision for the scalar reduction).

**Aceita (`per_patch_mse`, `per_patch_psnr`):**
- `a, b` are 4-D `(L, C, h, w)` tensors with identical shape, dtype, device.
- Returns `Tensor[L]` with the metric per patch. Reduction is over `(C, h, w)`; the leading axis is preserved.
- Output dtype matches input. PSNR returns `+inf` element-wise when the per-patch MSE is exactly zero (`torch.where`, not clamp).

**Rejeita (`ValueError` / `TypeError`):**
- Non-tensor input (`TypeError`).
- Shape mismatch, dtype mismatch, or device mismatch (`ValueError`).
- Non-positive, non-finite, or non-numeric `max_value` (`ValueError`).
- `per_patch_*`: input `ndim != 4` (`ValueError`).

**Fora de escopo v0.1:**
- SSIM / MS-SSIM / LPIPS / FID / any windowed or learned metric. Use `pytorch-msssim`, `lpips`, etc. on the caller's side.
- Per-channel reduction variants. Caller slices the tensor and calls these directly.
- Auto-detection of `max_value`. Caller knows the range of their data.
- Normalized cross-correlation, cosine similarity. Out of charter (those compare *whole signals*, not pixel-wise reconstructions).

### 9.9 `stitch(patches, image_shape, stride, *, weight="uniform", dilation=1)`

The blending counterpart to `reconstruct`. Same fold geometry, same rejections; adds a window-kernel weighting so modified patches can be reassembled with attenuated seams.

**Aceita:**
- `patches` is a 4-D `(L, C, ph, pw)` floating-point tensor (`float16`, `bfloat16`, `float32`, `float64`).
- `image_shape == (C, H, W)` consistent with the patch grid implied by `patches.shape[0]`, `stride`, and `(ph, pw)`.
- `stride` as `int` or `(int, int)` with `1 ≤ stride ≤ patch_size` on every axis.
- `weight` ∈ `{"uniform", "hann", "gaussian"}`. `"uniform"` is mathematically equivalent to `reconstruct`.
- `dtype` and `device` preserved on output.

**Sinaliza (não exceção):**
- `weight="hann"` at corners covered only by edge-weight-zero positions → output pixel is 0 (documented artifact, §2.5). Surfaces most visibly at `stride == patch_size`.

**Rejeita (`ValueError` / `TypeError`):**
- `patches` not a tensor (`TypeError`).
- `patches.ndim != 4` (`ValueError`).
- `patches.dtype` not floating-point (`ValueError`, message instructs `patches.float()`).
- `weight` not in the allowed set (`ValueError`).
- `dilation != 1` (`ValueError`).
- `stride > patch_size` on any axis (partial coverage forbidden; same as §9.2).
- `image_shape` malformed (not 3-tuple, non-positive dim, non-int).
- Channel mismatch between `image_shape[0]` and `patches.shape[1]`.
- `patches.shape[0]` inconsistent with the grid implied by `image_shape`, `patch_size`, `stride`.

**Fora de escopo v0.1:**
- Caller-provided custom weight tensor (the three named kernels cover seam-blending; an arbitrary tensor adds surface area without a known use case).
- Per-channel weights.
- Auto-selection of kernel based on stride/patch-size.
- Promotion of integer patches to float (caller normalizes).
- Output in PIL.
