# ADR 0002 — `Patchify`: callable wrapper for transform pipelines

- **Status:** Accepted
- **Date:** 2026-05-16
- **Deciders:** Leonardo Marques de Souza
- **Relates to:** [ADR 0001](0001-patch-extraction-api.md), [`THEORY.md`](../THEORY.md) §0, §1, §9.1

## Context

[ADR 0001](0001-patch-extraction-api.md) established `extract(image, patch_size, stride, dilation)` as a pure function and explicitly rejected a class-based replacement to avoid state creep (LRU cache, fixed `image_size`, hidden disk cache — the drift visible in `archive/QSVM_patchkit/patchkit/patches.py::OptimizedPatchExtractor`).

A new use case crystallized in conversation: PatchForge must be **acoplável** ("attachable") to other people's torch pipelines. Concretely, a downstream consumer wants:

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.GaussianBlur(kernel_size=3),
    patchforge_step,                              # ← here
])
dataset = MNIST(..., transform=transform)
loader = DataLoader(dataset, num_workers=4)     # parallelism for free
```

The functional API forces `patchforge_step` to be a `lambda img: extract(img, 4, 2)` or `functools.partial(extract, patch_size=4, stride=2)`. Both work, but:

- **Not introspectable.** `print(transform)` shows `<function <lambda> at 0x...>` — useless for debugging long pipelines.
- **No eager validation.** `lambda img: extract(img, patch_size=-1, stride=-1)` only fails when the first batch hits the worker. With a class, `__init__` catches the config error at pipeline construction.
- **Awkward to mix with the rest of the lib's idiom.** Callers reading the code see `extract` documented as a function and `partial(extract, ...)` ad-hoc-glued in — the lib appears to lack a first-class hook for this very common integration.

## Decision

Add `Patchify` to `src/patchforge/extract.py` and export it from `patchforge`. It is a thin callable companion to `extract`:

```python
class Patchify:
    __slots__ = ("_ph", "_pw", "_sh", "_sw", "_dh", "_dw")

    def __init__(self, patch_size, stride, dilation=1):
        self._ph, self._pw = _as_pair(patch_size, "patch_size")
        self._sh, self._sw = _as_pair(stride, "stride")
        self._dh, self._dw = _as_pair(dilation, "dilation")

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return extract(image, (self._ph, self._pw), (self._sh, self._sw),
                       (self._dh, self._dw))

    def __repr__(self) -> str: ...
```

### Specified behaviour

- **Single delegation.** `__call__` does nothing except forward to `extract`. Same shape contract, same dtype/device preservation, same truncation policy, same `(L, C, ph, pw)` output, same exceptions.
- **Eager validation.** `__init__` calls `_as_pair` on each axis. A bad `patch_size`, `stride`, or `dilation` raises `ValueError` at pipeline construction, not at first inference.
- **No state beyond geometry.** `__slots__` lists exactly six ints. No `__dict__`, no buffer, no cache, no `image_size`, no device, no last-image reference.
- **Reusable across image sizes.** A single instance handles every `(C, H, W)` the user feeds it — same property as the function form.
- **Repr-friendly.** `repr(Patchify(4, 2))` returns `"Patchify(patch_size=(4, 4), stride=(2, 2), dilation=(1, 1))"`, suitable for printing transform pipelines.

### What is deliberately *not* in this class

- **No caching.** Adding caching would re-invite the state creep ADR 0001 forbade. Caching is a `patchforge.Cache` concern, composed externally.
- **No fixed image size.** Same rejection as ADR 0001 alternative A.
- **No `nn.Module` inheritance.** `Patchify` is not a layer: no parameters, no gradient hook, no `.to(device)`, no `.train()`/`.eval()`. Subclassing `nn.Module` would imply semantics PatchForge does not honor.
- **No batch axis.** The output stays `(L, C, ph, pw)` — same as `extract`. Per [`THEORY.md`](../THEORY.md) §0, multi-image batching is out of scope; the `transforms.Compose` pipeline applies one image at a time per worker, which fits.
- **No `inverse` method.** Round-trip is `patchforge.reconstruct` (separate function). Bundling inverse here would tie two milestones (M2, M3) into one class — defeats decoupling.

## Consequences

**Positive.**
- Slots into `torchvision.transforms.Compose([..., Patchify(4, 2), ...])` without callers writing lambdas.
- Config errors fail at construction time (where the stack trace points to user code), not at first batch (where the stack trace points to a DataLoader worker).
- Provides a first-class integration point for the most common use case — patch extraction inside a torch pipeline — without changing the function or moving toward orchestration.
- Tests are trivial: confirm delegation to `extract` and confirm `__init__` rejects bad geometries.

**Negative.**
- Two API surfaces for the same primitive. Documented: use `extract` for one-off calls and ad-hoc scripts; use `Patchify` inside a Compose chain or wherever a callable object is expected.
- A future reader may be tempted to add state ("it's already a class…"). The `__slots__` declaration and the explicit "not in this class" list above are the load-bearing safeguards. Any change adding fields requires a new ADR.

**Neutral.**
- The function form (`extract`) remains the contract per ADR 0001. `Patchify` is a wrapper, not a successor.

## Alternatives considered

### A. Document the `lambda` / `functools.partial` pattern in the README and add nothing to the API

**Rejected.** Loses eager validation and introspectability. Users would re-invent the wrapper in each project, sometimes correctly, sometimes not. The class is 30 lines; spending them once in the lib beats spending them dozens of times in consumer code.

### B. Add `Patchify` and *remove* the `extract` function

**Rejected.** Contradicts ADR 0001. The function is the contract. The class is a convenience.

### C. Subclass `torch.nn.Module` so `Patchify` is a `nn.Module`-compatible transform

**Rejected.** `nn.Module` carries semantics PatchForge does not implement (parameter registration, device migration via `.to`, training-mode switching, autograd hooks). Inheriting from it implies those features work; they would silently no-op or fail in surprising ways. The plain callable class is what `torchvision.transforms.v2` itself uses internally for stateless transforms.

### D. Return a single random patch per call so `Patchify` becomes a per-sample augmentation

**Rejected.** That would make `Patchify` non-deterministic and would mismatch the function's shape contract. Per-sample patch augmentation is a higher-level concern that belongs to consumer pipelines or to a future `tests/_datasets.py` helper — not to the core primitive.

## Status after M2

`Patchify` ships with M2 (extract). When M3 lands (reconstruct), a sibling `Unpatchify` may be considered under a future ADR if a real consumer asks for it; until then, callers use `reconstruct` directly.
