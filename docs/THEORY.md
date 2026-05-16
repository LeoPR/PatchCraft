# PatchKit — Theory Notes

Working document. The goal is to distill the useful theory from the reference implementations in [`../archive/`](../archive/) into a single place **before** writing new code. Keep this document as the source-of-truth for design decisions.

> Every section should end with a paragraph titled **Design decision** stating what PatchKit will actually implement and why.

## 1. Patch extraction

**Topic.** Given an image of shape `(C, H, W)`, a patch size `(ph, pw)`, a stride `(sh, sw)` and an optional dilation `d`, define:
- `num_patches_h`, `num_patches_w` as functions of the inputs.
- The mapping `(patch_index) → (row, col)` and its inverse.
- The extraction operator as a single call to `torch.nn.functional.unfold`.

**Counts and indexing.**
- `num_patches_h = floor((H − d·(ph − 1) − 1) / sh) + 1`; analogous for width.
- Row-major ordering: patch `k` is at `(row, col) = (k // num_patches_w, k % num_patches_w)`.
- The top-left pixel of patch `k` is at `(row · sh, col · sw)` in the original image.

**Boundary conditions.** Truncation, never padding. Pixels in the trailing rows/columns that do not fit a full patch are dropped. PatchKit does not synthesize data; callers who need full coverage should pad the image themselves before extracting.

**Memory layout.** `F.unfold(x.unsqueeze(0), (ph,pw), dilation=d, stride=(sh,sw))` yields shape `(1, C·ph·pw, L)`. Reshape as `.view(C, ph, pw, L).permute(3, 0, 1, 2).contiguous()` to obtain `(L, C, ph, pw)`. Device and dtype are preserved from the input.

**Design decision.** `patchkit.extract(image, patch_size, stride, dilation=1) -> Tensor[L, C, ph, pw]` is a pure function built on `F.unfold`. `image` must be `(C, H, W)`; batching is explicit (call in a loop or `vmap`) and is not part of the v0.1 signature. `patch_size` and `stride` accept `int` (square) or `(int, int)`. Truncation is the only boundary policy. When no patch fits, we return an empty tensor `Tensor[0, C, ph, pw]` rather than raising — callers decide. No caching inside the function; caching is a separate concern (see §4). Dilation is supported here even though reconstruction rejects it (see §2). See also [ADR 0001](ADR/0001-patch-extraction-api.md) for the API rationale.

## 2. Reconstruction

**Topic.** Inverse of extraction via `torch.nn.functional.fold`. Two regimes:
- **Exact** — when `sh == ph` and `sw == pw` and `d == 1`: patches tile the image; reconstruction is a cheap copy (weights are all ones).
- **Weighted overlap** — when `sh < ph` or `sw < pw`: patches overlap; reconstruction must divide by the overlap count map (from `fold` of an all-ones tensor with the same geometry).

**Worked example (`ph = pw = 4, sh = sw = 2`).** Every interior pixel is covered by 4 patches (2 row overlaps × 2 col overlaps); edge pixels are covered by 2; corners by 1. The count map produced by `fold(ones_like(patches_flat), output_size=(H,W), kernel=(4,4), stride=2)` encodes exactly these weights. Dividing the summed pixel contributions by this map recovers the original image bit-exactly for fractional pixel values, and within one ULP for float32 noise.

**Dilation.** `F.fold` does not support dilation in the same way as `F.unfold`: for `d > 1` the patch footprint skips pixels, and `fold` would deposit the sparse contributions into a canvas that is not the image. Rather than building a custom scatter, we refuse `dilation != 1` at reconstruction time with a clear `ValueError`. Callers who extracted with dilation are expected to consume patches directly (e.g., as features) and not round-trip.

**Stride maior que patch (lacunas).** Quando `sh > ph` ou `sw > pw`, o grid pula pixels: a soma `fold(...)` tem zeros nessas posições, e qualquer divisão (incluindo pelo clamp `min=1e-6`) produz pixels com valor arbitrário — síntese de dado, exatamente o que §1 proíbe ao escolher truncamento como única política de borda. Recusamos a condição com `ValueError` logo na entrada de `reconstruct`. Consumidores que querem features esparsas (kernels, classificadores) usam apenas `extract`, onde `stride > patch_size` é aceito sem objeções, e não tentam round-trip.

**Design decision.** `patchkit.reconstruct(patches, image_shape, stride, dilation=1) -> Tensor[C, H, W]` uses `F.fold` followed by division by the overlap-count map (computed once, same-geometry fold of ones). Raises `ValueError` when `dilation != 1` **ou quando `sh > ph` ou `sw > pw`** (cobertura parcial é proibida). `image_shape` is `(C, H, W)` and must match the geometry implied by the patch grid; inconsistent shapes raise `ValueError`. The count map clamp (`min=1e-6`) existe apenas para absorver ruído float em pixels totalmente cobertos — nunca para mascarar buracos de cobertura. Output dtype matches input.

## 3. LR ↔ HR pairing

**Topic.** Given a scale factor `r` and LR/HR shapes related by `HR = r · LR`, define patch sizes and strides on both sides so that the *k*-th LR patch corresponds to the *k*-th HR patch (same `k`). Invariant:
- `ph_hr = r · ph_lr` and `sh_hr = r · sh_lr` (similar for width).
- Then `num_patches_lr == num_patches_hr`, and the coordinates of patch *k* on both sides map to the same image region in pixel space.

**Coverage invariant (sketch).** Given `H_hr = r · H_lr`, `ph_hr = r · ph_lr`, `sh_hr = r · sh_lr`, and dilation 1 everywhere:

  `num_h_lr = floor((H_lr - ph_lr) / sh_lr) + 1`
  `num_h_hr = floor((H_hr - ph_hr) / sh_hr) + 1 = floor((r·H_lr - r·ph_lr) / (r·sh_lr)) + 1 = floor((H_lr - ph_lr) / sh_lr) + 1`

So `num_h_lr == num_h_hr` (and analogously for width) exactly when the scale factor is an integer. The top-left of LR patch `k` is at `(row · sh_lr, col · sw_lr)`; multiply by `r` to get the HR patch origin — same image region, different resolution.

**Truncation.** When `(H_lr − ph_lr) % sh_lr != 0`, the trailing LR rows that don't fit a full patch are dropped. Because of the integer scaling, the same trailing HR rows are also dropped — the two grids stay aligned. The user loses a strip on the right/bottom of both resolutions. If full coverage is required, pad the LR image *before* extracting (and the HR image correspondingly) — PatchKit does not pad implicitly.

**Recommended defaults.** For tiling (no overlap): `stride_lr = ph_lr`, `stride_hr = ph_hr = r · ph_lr`. For 2× overlap training data: `stride_lr = ph_lr // 2`.

**Design decision.** PatchKit exposes `patchkit.pair(lr_image, hr_image, lr_patch_size, scale_factor, stride)` that returns an iterator of `(lr_patch, hr_patch, meta)` tuples. `scale_factor` must be a positive `int` (non-integer scale is rejected — irrational alignment has no clean semantics). `lr_patch_size` and `stride` are `int` or `(int, int)`; HR equivalents are derived. `dilation=1` is fixed (not exposed) because dilation breaks the reconstruction story. `meta` is a small dataclass (`PatchMeta`) holding `image_id`, `patch_index`, `row`, `col`, `lr_patch_size`, `hr_patch_size` — readable, not GPU-resident (metadata stays on CPU). See §8 for why a dataclass over a dict/tensor.

## 4. Cache semantics

**Topic.** Processed datasets (resized / quantized / paired) are expensive to build; cache them on disk keyed by a hash of the configuration. Reference implementations:
- `archive/PatchHub/src/patchhub/cache.py` — mmap-friendly on-disk cache with zstd.
- `archive/QSVM_patchkit/patchkit/processed.py` — cache bundle with labels.

**Cache key.** The key is a SHA-256 over a canonical JSON serialization of every input that affects the output. For a resized tensor that is: the image fingerprint (SHA-256 of raw bytes), target size, backend name, resample filter, and a `version` integer owned by the function. The `version` field is the invalidation lever: when the algorithm changes, bump it and the old cache becomes inaccessible by construction (no delete needed).

**Invalidation.** No TTL, no mtime check. Cache entries are immutable; a changed configuration produces a different key and a fresh entry. Stale entries accumulate on disk — cleanup is a manual maintenance task, not a runtime concern. This is safe only because inputs are content-addressed: the same config and the same image bytes always yield the same result.

**Serialization format.** `torch.save` to a `bytes` buffer, then zstd-compress when `zstandard` is installed (level 3), else write raw. File layout: one file per entry, filename = first 16 hex chars of the key, stored under `<cache_dir>/<prefix>/`. A tiny sidecar JSON per entry stores the full key, version, and content checksum — enough to detect truncated writes without scanning the whole payload.

**Concurrency.** Single-writer, multi-reader. PatchKit does not implement locking. If two processes race on the same key, the second writer wins (atomic rename from a `*.tmp` file). No lockfiles — OneDrive-sync weirdness has already shown that filesystem locks are unreliable here.

**Robustez a write races (OneDrive, antivírus, indexador).** Em diretórios sincronizados com OneDrive — e por extensão qualquer pasta varrida por antivírus ou Windows Search — `os.rename`, `open(..., "wb")` e `os.replace` falham esporadicamente com `PermissionError` (errno 13, Windows error 5) enquanto um agente externo segura o handle por alguns ms. Já vimos esse comportamento em `uv lock` rodando contra `uv.lock` em pasta OneDrive. `Cache.put` envolve o rename atômico (e o `open` do `*.tmp`) num loop de retry com backoff exponencial: até 5 tentativas, esperas de `0.25, 0.5, 1.0, 2.0, 4.0` segundos. Depois disso, a exceção original sobe — falha persistente é problema legítimo (disco cheio, permissões reais, OneDrive offline). `Cache.get` aplica o mesmo wrap em `open(..., "rb")` com 2 tentativas apenas — leitura raramente é o lado bloqueado e falhar rápido aqui é melhor que esconder cache miss real.

**Design decision.** One cache module with one public class: `patchkit.Cache(root, namespace, version=1)` (parâmetro renomeado de `dir` para evitar sombrear o builtin). Methods: `get(key) -> bytes | None`, `put(key, bytes)`, `key_for(*parts) -> str`. `root` inexistente é auto-criado (`mkdir(parents=True, exist_ok=True)`) — ergonomia. Writes envolvem o backoff descrito acima; reads usam variante curta. Higher-level helpers (`cached_resize`, etc.) compose `Cache` with a function — they are thin wrappers, not inheritance. No pluggable backends (PatchHub's `memory | shelve | diskcache | hybrid` matrix is overkill for our use cases). No mmap in v0.1 — deferred until a profile shows memory pressure. The `zstandard` dep is an `optional-extra` (`[cache]`), and the absence of it falls back to uncompressed writes transparently.

## 5. Resize backends

**Topic.** Image resize is surprisingly opinionated:
- **PIL** — battle-tested, supports BICUBIC/LANCZOS/etc., CPU-only, convenient.
- **torch** — `F.interpolate`, runs on GPU, different bicubic filter than PIL.

**Non-identity.** PIL's BICUBIC and torch's `F.interpolate(mode='bicubic')` use different kernel parameters (PIL uses `a = -0.5`, torch uses `a = -0.75` with `align_corners=False`) and produce measurably different output. For a 28×28 MNIST image resized to 14×14 the per-pixel difference is typically within ±3 / 255 but can exceed that at edges. **This matters for reproducibility.** A dataset pre-processed with PIL cannot be compared head-to-head against one pre-processed with torch even when the "same" algorithm name is used.

**When to prefer each.**
- **PIL**: canonical dataset generation (reproducibility is paramount, result stays on CPU anyway).
- **torch**: in-training augmentation where the tensor is already on GPU and the extra filter divergence does not matter.

**Design decision.** One function: `patchkit.resize(image, target_size, backend="pil", resample=None)`. `target_size` is a `(H, W)` tuple — no size-spec DSL. `image` may be `PIL.Image` or `Tensor[C, H, W]`. Output type matches input (PIL in → PIL out, Tensor in → Tensor out), regardless of backend. `resample` defaults: `LANCZOS` for PIL, `bilinear` for torch (chosen because bicubic divergence is nastiest to debug; bilinear's difference from PIL's BILINEAR is small). Cross-backend conversions (PIL ↔ Tensor) go through a normalized float32 `[0, 1]` intermediate. No auto-selection: callers pick the backend explicitly and own the consequences.

## 6. Quantization (optional)

**Topic.** Reducing color depth (binary, k-level) before extraction. Reference: `archive/QSVM_patchkit/patchkit/quantize.py` — uniform, k-means, Otsu, Floyd–Steinberg dither.

**Does it belong here?** PatchKit's charter (see README §Scope) is patch infrastructure. Quantization is a *consumer-specific preprocessing step*: QSVM needs it because small bit-depths stabilize kernel computations; other consumers may not want it, or may want a different family (e.g. vector quantization). Keeping it inside PatchKit means every consumer pays the cost of evaluating whether the in-package implementation matches their needs, and updates force coordinated releases.

**Design decision.** **Out of scope for v0.1.** Quantization is deferred. When a pattern emerges across multiple consumers, revisit as either (a) a companion package (`patchkit-quant`), or (b) a plug-in hook point in the extraction pipeline. For now, callers that need quantization apply it *before* calling `patchkit.extract` — the `(C, H, W)` tensor they pass in is whatever they want patches of. The archive implementations stay in `archive/` as reference material for whoever builds the companion.

## 7. Label-stratified subsets

**Topic.** Given a labeled dataset, pick `n` samples per label with a deterministic seed. Reference: `archive/PatchHub/src/patchhub/subset.py` — `LabelSubset`.

**Functional vs class.** PatchHub's `LabelSubset` was a wrapper class that reimplemented `__getitem__` to forward to the base dataset. That's redundant: `torch.utils.data.Subset` already does exactly this given an index list. So the only real work is producing the index list.

**Deterministic seeding.** Use `numpy.random.Generator(PCG64(seed))` — not the global `numpy.random` state. An explicit `int` seed is required; no "auto" mode. Same seed + same label distribution ⇒ same indices, regardless of process order, thread count, or torch state.

**Label extraction.** Prefer `dataset.targets` (torchvision convention). If absent, try `dataset.labels`. If both absent, iterate once and collect — warn the user because this forces a full pass that may be expensive on large datasets.

**Design decision.** `patchkit.label_subset(dataset, n_per_label, seed, classes=None) -> torch.utils.data.Subset`. Strictly functional — no wrapper class. `n_per_label` is an `int`; no float-percentage overloading (PatchHub's dual semantics for `int|float` was a source of confusion). `classes=None` selects every class; pass a `Sequence[int]` to subset. If a requested class has fewer than `n_per_label` samples available, take all of them and emit a warning (do not raise — partial stratification is almost always what the caller wants).

## 8. Resolved questions

- **`PatchPairDataset` vs decoupled primitives?** Decoupled. v0.1 ships `extract`, `reconstruct`, `pair`, `resize`, `label_subset`, `Cache` as independent pieces. Consumers that want a `torch.utils.data.Dataset` compose them themselves (5 lines). This mirrors the M2–M6 ordering in the roadmap and avoids the QSVM_patchkit problem where `SuperResPatchDataset` baked in assumptions (ProcessedDataset, specific caching, specific labels) that the next consumer does not share.
- **`meta` shape — dict, structured tensor, or dataclass?** Dataclass (`PatchMeta`). Dicts lose type info; structured tensors force CPU→GPU transfers for fields that are never used on device (image id, patch index). Metadata stays on CPU; the pixel payload is the only thing that should move to GPU. `@dataclass(frozen=True, slots=True)` keeps it cheap.
- **Minimum torch?** `>=2.6` as per `pyproject.toml`. Sticking with this until a feature we want (e.g., `vmap` improvements) forces a bump.

## 9. Remaining open questions

- **Channels-first only, or also channels-last?** Currently every API assumes `(C, H, W)`. A `channels_last=True` flag may be desirable for interop with CV libraries that default to HWC. Defer to a real consumer request.
- **Tensor-vs-PIL in `extract`/`reconstruct`.** Extraction is tensor-only. Should we allow PIL input with an internal conversion? Current lean: no — converting before `extract` is one line, and mixed-type APIs hide cost.
- **Batched extraction.** For datasets of many small images, calling `extract` in a Python loop may be slow. Benchmark after M2 and decide whether to add a batched variant or document `torch.vmap` as the pattern.

## 10. Contrato de condições suportadas

Esta seção consolida, por API, as condições que v0.1 deve **aceitar**, **rejeitar** (com `ValueError` e mensagem explícita) e tratar como **fora de escopo** (não implementado, documentar como tal). Serve de fonte do plano de testes: cada item "Aceita" vira teste positivo; cada "Rejeita", teste negativo (`pytest.raises(ValueError)`). Onde esta seção diverge dos parágrafos "Design decision" de §1–§7, §10 é a verdade — ajuste os outros parágrafos no marco que implementar a API correspondente.

### 10.1 `extract(image, patch_size, stride, dilation=1)`

**Aceita:**
- `image` é `Tensor` 3D `(C, H, W)` com `C ≥ 1` arbitrário; dtype suportado por `torch.nn.functional.unfold` (todos os float — `float16/float32/float64/bfloat16` — e integer ≥ 16 bits no caminho CUDA; em CPU, `uint8` não é suportado por `im2col_cpu` e o caller deve converter pra float antes); CPU ou CUDA (preservados na saída).
- `patch_size`, `stride`, `dilation` como `int` (quadrado) ou `(int, int)` com valores positivos.
- Patch maior que imagem → retorna `Tensor[0, C, ph, pw]` (não levanta).
- `stride > patch_size` (grid esparso com lacunas — válido para features que não fazem round-trip; ver §10.2 para a contrapartida em `reconstruct`).
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

### 10.2 `reconstruct(patches, image_shape, stride, dilation=1)`

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

### 10.3 `pair(lr_image, hr_image, lr_patch_size, scale_factor, stride)`

**Aceita:**
- `scale_factor ∈ ℤ⁺` (inteiro positivo).
- `lr_image.shape == (C, H_lr, W_lr)` e `hr_image.shape == (C, scale_factor * H_lr, scale_factor * W_lr)`.
- Mesmo `C` e dtype em LR e HR.
- `lr_patch_size`, `stride` como `int` ou `(int, int)` positivos.

**Rejeita (`ValueError`):**
- `scale_factor` não-inteiro ou ≤ 0.
- `hr_image.shape` não bate com `scale_factor * lr_image.shape`.
- `C_lr != C_hr`.
- LR ou HR não 3D.

**Fora de escopo v0.1:**
- `dilation` customizado (fixo em 1 — reconstrução pareada precisa funcionar).
- Pareamento N:1 (várias LRs degradadas pra mesmo HR).
- Channels-last.
- Mismatch de dtype LR vs HR (caller normaliza antes).

### 10.4 `resize(image, target_size, backend="pil", resample=None)`

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

### 10.5 `Cache(root, namespace, version=1)`

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

### 10.6 `label_subset(dataset, n_per_label, seed, classes=None)`

**Aceita:**
- `n_per_label ∈ ℤ⁺`.
- Dataset com `.targets`, ou `.labels`, ou iterável (warn no terceiro caso por causa do custo do full pass).
- `classes=None` (todas as classes) ou `Sequence[int]` (subset das classes existentes).
- Classe com menos de `n_per_label` amostras → take all + warn (parcial é o desejado).
- Mesmo `seed` + mesmo dataset ⇒ mesmos índices, independente de plataforma, thread count, ou estado do torch.

**Rejeita (`ValueError`):**
- `n_per_label ≤ 0`.
- `seed` não é `int`.
- `classes` contém id que não existe no dataset.

**Fora de escopo v0.1:**
- Datasets multi-label (lista de labels por amostra).
- Estratificação por proporção (`n_per_label` como `float` em `(0, 1]`).
- Reweighting / sampler ponderado.
