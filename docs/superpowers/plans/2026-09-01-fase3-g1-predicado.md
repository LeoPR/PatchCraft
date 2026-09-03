# Fase 3 — G1: predicado bit-exato e a suíte falsificadora (0.5.0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fechar os bloqueadores B1/B2/B3/B6 do FOCO-1.0: o predicado correto ("bit-exato sse todo valor do count map é potência de dois; fora, erro por pixel ≤ `(k+1)·eps·|v|`") escrito em todo lugar, uma suíte capaz de falsificá-lo, a superfície pública de 20 nomes fixada por teste, a enumeração de `tilings` sem specs degenerados, e release 0.5.0 com retratação pública.

**Architecture:** Só testes e texto mudam o comportamento observável, com duas exceções pontuais: a guarda `nh == 1 and nw == 1` em `geometry.py` (D1) e o bump de versão. Os helpers auditados de `tests/_rng.py` (geração de ruído no dtype alvo, comparação bitwise NaN-safe, predicado em duas formas independentes, bound por pixel) sustentam a suíte nova (`test_exactness.py`, `test_reference.py`, `test_public_api.py`) e a reescrita dos round-trips existentes.

**Tech Stack:** Python ≥3.12, torch ≥2.6, numpy ≥1.26 (já deps), pytest; sem novas dependências. Spec de origem: `docs/superpowers/specs/2026-09-01-fase3-g1-predicado-design.md` (inclui o Amendment A: o bound "1 ULP" foi medido falso e é substituído pelo bound por pixel).

## Global Constraints

- Branch de trabalho: `feat/0.5.0-g1-predicado` (criada na Task 1). Commits por task. **Nunca** `git push`; merge local para `main` só na Task 10.
- Contrato numérico escrito (D5/Amendment A), vale para todo texto e teste: dentro do predicado (todo valor do count map é potência de dois) o round-trip é **bit-exato**; fora, o erro por pixel é limitado por `(k+1)·eps·|v|` (k = contagem de cobertura do pixel). **Nunca** escrever "1 ULP" sem qualificação — a frase foi medida falsa (até 19 ULP com k=81).
- Superfície pública congelada (D4): exatamente os 20 nomes atuais de `patchcraft.__all__`.
- Estilo: `from __future__ import annotations` após o docstring em todo módulo novo; testes importam helpers via `from tests._rng import ...` (precedente: `tests/_datasets.py`); ruff `line-length = 100`, select `E,F,W,I,N,UP,B,SIM,RUF`; mypy `--strict` cobre só `src/`, mas o código de teste novo vai tipado mesmo assim.
- Comandos rodam via `uv run ...` no Git Bash. Suite: `uv run pytest -m "not gpu"`. Marcadores registrados: `slow`, `gpu` (`--strict-markers` ativo).
- A suite deve passar nos dois modos: `PATCHCRAFT_ACCEL=0` (puro) e com o acelerador ativo (se o wheel estiver montado; `_accel.py` faz fallback silencioso caso contrário).
- Cargo (só verificação, crate intocado): `CARGO_TARGET_DIR=C:/Users/leona/.cache/patchcraft-target cargo test` dentro de `accel/`; pular se a toolchain não existir.
- D2: comportamento de `extract` (trunca) e `reconstruct` (recusa) **não** muda; só se documenta a assimetria.
- D3: `WeightKind` é conjunto aberto; o teste fixa os três valores atuais como ponto de partida.
- Números já verificados empiricamente (não re-medir): espaço legal = 126.736 geometrias; amostra seed 20260901 de 256 ⇒ 67 dentro / 189 fora do predicado; receita ≡ count map com **0 divergências no espaço completo** (12s com o count map por difference arrays); `tilings((28,28), allow_overlap=True)`: 100 → 73; `paired_tilings((14,14),(28,28), allow_overlap=True)`: 40 → 27; referência naive dentro do bound fora do predicado: 0 violações em 3 geometrias × 2 dtypes × 20 seeds.

## File Structure

- Create `tests/_rng.py` — helpers auditados (dados, comparação, predicado ×2, bound).
- Create `tests/test_rng.py` — auto-testes dos helpers.
- Create `tests/test_exactness.py` — suíte falsificadora (duas metades + receita×predicado + NaN + sweep completo).
- Create `tests/test_reference.py` — referência naive com loops puros (substituto do consumer gate, D6).
- Create `tests/test_public_api.py` — congelamento da superfície (B3/D4).
- Modify `tests/test_reconstruct.py`, `tests/test_geometry.py`, `tests/test_stitch.py` — round-trips sobre ruído real + bound por pixel.
- Modify `tests/test_extract.py` — docstring da política de trunca (D2).
- Modify `tests/test_cache.py` — ramo sem zstandard.
- Modify `src/patchcraft/geometry.py` — guarda D1 + docstring de `TilingSpec`.
- Modify `src/patchcraft/reconstruct.py`, `src/patchcraft/stitch.py` — docstrings (B1).
- Modify `docs/SCOPE.md`, `docs/THEORY.md`, `docs/GUIDE.md`, `docs/USAGE.md`, `README.md`, `README.pt-BR.md` — predicado qualificado (B1) e assimetria §9.1 (D2).
- Modify `src/patchcraft/__init__.py` (`__version__`), `CHANGELOG.md` — release 0.5.0.

---

### Task 1: `tests/_rng.py` — helpers auditados + branch

**Files:**
- Create: `tests/_rng.py`
- Create: `tests/test_rng.py`

**Interfaces:**
- Consumes: nada de tasks anteriores.
- Produces (usado pelas Tasks 2, 3, 4):
  - `rand_image(c: int, h: int, w: int, dtype: torch.dtype, seed: int) -> torch.Tensor`
  - `bit_equal(a: torch.Tensor, b: torch.Tensor) -> bool` (NaN-safe; suporta f16/f32/f64)
  - `exact_axes_pow2(h, w, ph, pw, sh, sw) -> bool` (receita O(H+W))
  - `count_map_pow2(h, w, ph, pw, sh, sw) -> bool` (via `coverage_counts`)
  - `coverage_counts(h, w, ph, pw, sh, sw) -> np.ndarray` — mapa `(h, w)` int64
  - `within_pixel_bound(out: torch.Tensor, img: torch.Tensor, counts: np.ndarray) -> bool`

- [ ] **Step 1: criar a branch**

```bash
git checkout -b feat/0.5.0-g1-predicado
```

- [ ] **Step 2: escrever os testes dos helpers (falham antes do módulo existir)**

Criar `tests/test_rng.py`:

```python
"""Self-tests for the audited helpers in tests/_rng.py (G1, FOCO §4)."""
from __future__ import annotations

import numpy as np
import torch

from tests._rng import (
    bit_equal,
    count_map_pow2,
    coverage_counts,
    exact_axes_pow2,
    rand_image,
    within_pixel_bound,
)


class TestRandImage:
    def test_seeded_reproducible(self) -> None:
        a = rand_image(1, 8, 8, torch.float32, seed=7)
        b = rand_image(1, 8, 8, torch.float32, seed=7)
        assert bit_equal(a, b)

    def test_different_seed_differs(self) -> None:
        a = rand_image(1, 8, 8, torch.float32, seed=7)
        b = rand_image(1, 8, 8, torch.float32, seed=8)
        assert not bit_equal(a, b)

    def test_generated_in_target_dtype_not_widened(self) -> None:
        """The banned shortcut: `.double()` of float32 leaves the low 29
        mantissa bits zero; true float64 noise does not (measured, FOCO §0)."""
        direct = rand_image(1, 16, 16, torch.float64, seed=3)
        widened = direct.float().double()
        assert not bit_equal(direct, widened)
        low29 = (1 << 29) - 1
        assert bool((widened.view(torch.int64) & low29 == 0).all())
        assert bool((direct.view(torch.int64) & low29 != 0).any())


class TestBitEqual:
    def test_nan_safe(self) -> None:
        img = rand_image(1, 4, 4, torch.float32, seed=1)
        img[0, 0, 0] = float("nan")
        clone = img.clone()
        assert not torch.equal(img, clone)  # NaN != NaN
        assert bit_equal(img, clone)  # bits are identical

    def test_one_ulp_differs(self) -> None:
        a = rand_image(1, 4, 4, torch.float32, seed=1)
        b = a.clone()
        b[0, 0, 0] = torch.nextafter(b[0, 0, 0], torch.tensor(2.0))
        assert not bit_equal(a, b)

    def test_dtype_mismatch(self) -> None:
        a = rand_image(1, 4, 4, torch.float32, seed=1)
        assert not bit_equal(a, a.double())


class TestPredicates:
    def test_stride_equals_patch_always_true(self) -> None:
        assert exact_axes_pow2(28, 28, 7, 7, 7, 7)
        assert count_map_pow2(28, 28, 7, 7, 7, 7)

    def test_half_stride_true(self) -> None:
        # counts are 1, 2, 4
        assert exact_axes_pow2(16, 16, 4, 4, 2, 2)
        assert count_map_pow2(16, 16, 4, 4, 2, 2)

    def test_foco_anchor_false(self) -> None:
        # FOCO anchor geometry (1, 4, 14) p=(4, 4) s=(1, 1): counts include 3
        assert not exact_axes_pow2(4, 14, 4, 4, 1, 1)
        assert not count_map_pow2(4, 14, 4, 4, 1, 1)

    def test_recipe_matches_count_map_on_fixed_grid(self) -> None:
        cases = [
            (28, 28, 7, 7, 7, 7),
            (16, 16, 4, 4, 2, 2),
            (4, 14, 4, 4, 1, 1),
            (13, 13, 4, 4, 3, 3),
            (12, 18, 4, 6, 2, 3),
            (9, 9, 4, 4, 1, 1),
            (24, 24, 9, 9, 3, 3),
            (10, 10, 4, 4, 2, 2),
            (7, 7, 3, 3, 2, 2),
        ]
        for g in cases:
            assert exact_axes_pow2(*g) == count_map_pow2(*g), g

    def test_coverage_counts_known_geometry(self) -> None:
        counts = coverage_counts(8, 8, 4, 4, 2, 2)
        assert counts[0, 0] == 1 and counts[7, 7] == 1
        assert counts[3, 3] == 4
        assert set(np.unique(counts).tolist()) == {1, 2, 4}


class TestWithinPixelBound:
    def test_exact_result_passes(self) -> None:
        img = rand_image(1, 9, 9, torch.float32, seed=5)
        counts = coverage_counts(9, 9, 4, 4, 1, 1)
        assert within_pixel_bound(img, img, counts)

    def test_detects_large_error(self) -> None:
        img = rand_image(1, 9, 9, torch.float32, seed=5)
        counts = coverage_counts(9, 9, 4, 4, 1, 1)
        assert not within_pixel_bound(img + 0.5, img, counts)

    def test_zero_image_bound_is_zero_and_exact(self) -> None:
        img = torch.zeros(1, 9, 9)
        counts = coverage_counts(9, 9, 4, 4, 1, 1)
        assert within_pixel_bound(img, img, counts)
```

- [ ] **Step 3: rodar e ver falhar**

Run: `uv run pytest tests/test_rng.py -v`
Expected: FAIL na coleta (`ModuleNotFoundError: No module named 'tests._rng'`).

- [ ] **Step 4: escrever `tests/_rng.py`**

```python
"""Audited data helpers for round-trip tests (G1; docs/FOCO-1.0.md §4).

Three generators of false negatives are banned from round-trip assertions:

- **Integer ramps** (``torch.arange``): small integers are exactly
  representable, so sums and divisions land exactly where generic float data
  does not. Use :func:`rand_image` whenever the *value* matters; ramps remain
  valid where only order/position matters (row-major layout, all-ones count
  maps, coverage guards).
- **Widened float32** (``x.double()``): the low 29 mantissa bits come out
  zero, so the data round-trips where true float64 noise does not. Generate
  directly in the target dtype.
- **``torch.equal`` on NaN data**: not reflexive (NaN != NaN even with
  identical bits). :func:`bit_equal` compares integer views and is NaN-safe.

The exactness predicate (ADR 0003): the ``extract``/``reconstruct`` round
trip is bit-exact iff every value of the overlap count map is a power of
two. :func:`exact_axes_pow2` computes it with an O(H+W) per-axis closed
form; :func:`count_map_pow2` materializes the count map with integer
difference arrays — an independent code path, so cross-checking the two
means something. Both assume exact coverage; without it there is no
round-trip at all (``reconstruct`` raises).
"""
from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "bit_equal",
    "count_map_pow2",
    "coverage_counts",
    "exact_axes_pow2",
    "rand_image",
    "within_pixel_bound",
]

_INT_VIEW: dict[torch.dtype, torch.dtype] = {
    torch.float16: torch.int16,
    torch.float32: torch.int32,
    torch.float64: torch.int64,
}


def rand_image(c: int, h: int, w: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    """Uniform [0, 1) image generated directly in ``dtype`` — never widened
    from a narrower dtype (that would leave half the mantissa zeroed)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.rand(c, h, w, dtype=dtype, generator=gen)


def bit_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bitwise equality via the integer view — NaN-safe, unlike
    ``torch.equal`` (which is not reflexive on NaN)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool((a.view(_INT_VIEW[a.dtype]) == b.view(_INT_VIEW[b.dtype])).all())


def exact_axes_pow2(h: int, w: int, ph: int, pw: int, sh: int, sw: int) -> bool:
    """Closed-form predicate, O(H+W). On one axis (length ``n``, patch ``p``,
    stride ``s``, exact coverage) the number of patches covering pixel ``i``
    is ``hi - lo + 1`` with ``hi = min(i // s, (n - p) // s)`` and
    ``lo = max(0, ceil((i - p + 1) / s))``. True iff every distinct count on
    both axes is a power of two."""
    for n, p, s in ((h, ph, sh), (w, pw, sw)):
        starts = (n - p) // s
        counts: set[int] = set()
        for i in range(n):
            hi = min(i // s, starts)
            lo = max(0, -((-(i - p + 1)) // s))
            counts.add(hi - lo + 1)
        if any(c & (c - 1) != 0 for c in counts):
            return False
    return True


def coverage_counts(h: int, w: int, ph: int, pw: int, sh: int, sw: int) -> np.ndarray:
    """The ``(h, w)`` integer count map, built with per-axis difference
    arrays (place every patch interval, cumsum, outer product) — no closed
    form and no ``F.fold``, so it cross-checks :func:`exact_axes_pow2`
    independently."""
    axes: list[np.ndarray] = []
    for n, p, s in ((h, ph, sh), (w, pw, sw)):
        diff = np.zeros(n + 1, dtype=np.int64)
        starts = np.arange(0, n - p + 1, s)
        np.add.at(diff, starts, 1)
        np.add.at(diff, starts + p, -1)
        axes.append(np.cumsum(diff[:-1]))
    return np.outer(axes[0], axes[1])


def count_map_pow2(h: int, w: int, ph: int, pw: int, sh: int, sw: int) -> bool:
    """Predicate via the materialized count map: every value a power of two."""
    vals = np.unique(coverage_counts(h, w, ph, pw, sh, sw))
    return bool(((vals & (vals - 1)) == 0).all())


def within_pixel_bound(
    out: torch.Tensor, img: torch.Tensor, counts: np.ndarray
) -> bool:
    """Amendment A bound: ``|out - img| <= (k + 1) * eps * |img|`` per pixel,
    with ``k`` the pixel's coverage count. Finite data only (NaN -> False)."""
    k = torch.from_numpy(counts).to(img.dtype).unsqueeze(0).expand_as(img)
    eps = torch.finfo(img.dtype).eps
    err = (out - img).abs()
    bound = (k + 1.0) * eps * img.abs()
    return bool((err <= bound).all())
```

- [ ] **Step 5: rodar e ver passar**

Run: `uv run pytest tests/test_rng.py -v`
Expected: 14 passed.

- [ ] **Step 6: lint + commit**

Run: `uv run ruff check tests/_rng.py tests/test_rng.py`
Expected: clean.

```bash
git add tests/_rng.py tests/test_rng.py
git commit -m "test(g1): audited round-trip helpers (rand_image, bit_equal, predicate x2, pixel bound)"
```

---

### Task 2: reescrita dos round-trips existentes sobre ruído real

**Files:**
- Modify: `tests/test_reconstruct.py`
- Modify: `tests/test_geometry.py` (`TestTilingsRoundtripGuarantee`, :152-170)
- Modify: `tests/test_stitch.py`

**Interfaces:**
- Consumes: `rand_image`, `bit_equal`, `coverage_counts`, `within_pixel_bound`, `count_map_pow2` de `tests/_rng.py` (Task 1).
- Produces: nada novo para outras tasks.

Regra (spec §2.2): onde o **valor** importa, `_ramp`/`arange` sai e entra `rand_image`; onde só **ordem/posição** importa (`TestCountMap` com imagem de uns, guards de cobertura, metas do `pair`), a rampa fica. Asserções novas:
- geometria dentro do predicado ⇒ `bit_equal(out, img)`;
- fora ⇒ `within_pixel_bound(out, img, counts)` (nunca `allclose` frouxo).

`tests/test_pair.py` **não muda** (ver Amendments, A2: `pair`/`extract` não fazem aritmética — é gather puro; a rampa lá testa posição).

- [ ] **Step 1: reescrever `tests/test_reconstruct.py`**

Adicionar ao bloco de imports:

```python
from tests._rng import bit_equal, coverage_counts, rand_image, within_pixel_bound
```

`TestRoundtripExact` inteiro vira (5 casos, todos dentro do predicado pois `stride == patch_size`):

```python
class TestRoundtripExact:
    """`stride == patch_size`: each pixel covered exactly once, bit-exact."""

    def test_basic(self) -> None:
        img = rand_image(3, 32, 32, torch.float32, seed=101)
        patches = extract(img, patch_size=8, stride=8)
        out = reconstruct(patches, image_shape=img.shape, stride=8)
        assert bit_equal(out, img)

    def test_rectangular_geometry(self) -> None:
        img = rand_image(1, 20, 30, torch.float64, seed=102)
        patches = extract(img, patch_size=(4, 6), stride=(4, 6))
        out = reconstruct(patches, image_shape=img.shape, stride=(4, 6))
        assert bit_equal(out, img)

    def test_single_patch_equals_image(self) -> None:
        img = rand_image(2, 8, 8, torch.float32, seed=103)
        patches = extract(img, patch_size=8, stride=8)
        out = reconstruct(patches, image_shape=img.shape, stride=8)
        assert bit_equal(out, img)

    def test_multichannel(self) -> None:
        img = rand_image(7, 16, 16, torch.float32, seed=104)
        patches = extract(img, patch_size=4, stride=4)
        out = reconstruct(patches, image_shape=img.shape, stride=4)
        assert bit_equal(out, img)

    def test_patch_size_1(self) -> None:
        img = rand_image(1, 4, 4, torch.float32, seed=105)
        patches = extract(img, patch_size=1, stride=1)
        out = reconstruct(patches, image_shape=img.shape, stride=1)
        assert bit_equal(out, img)
```

`TestRoundtripOverlap` inteiro vira:

```python
class TestRoundtripOverlap:
    """`stride < patch_size`: exact iff every count-map value is a power of
    two (ADR 0003); otherwise bounded per pixel by (k+1)*eps*|v|."""

    def test_half_overlap_basic(self) -> None:
        """counts are 1, 2, 4 -> inside the predicate -> bit-exact."""
        img = rand_image(1, 16, 16, torch.float64, seed=201)
        patches = extract(img, patch_size=4, stride=2)
        out = reconstruct(patches, image_shape=img.shape, stride=2)
        assert bit_equal(out, img)

    def test_max_overlap_stride_1(self) -> None:
        """p=3 s=1 puts a 3 in the count map -> outside the predicate;
        the per-pixel bound (k+1)*eps*|v| is the contract (Amendment A)."""
        img = rand_image(2, 8, 8, torch.float64, seed=202)
        patches = extract(img, patch_size=3, stride=1)
        out = reconstruct(patches, image_shape=img.shape, stride=1)
        assert within_pixel_bound(out, img, coverage_counts(8, 8, 3, 3, 1, 1))

    def test_asymmetric_overlap(self) -> None:
        """counts on both axes are {1, 2} -> 2-D map in {1, 2, 4} -> exact."""
        img = rand_image(1, 12, 18, torch.float64, seed=203)
        patches = extract(img, patch_size=(4, 6), stride=(2, 3))
        out = reconstruct(patches, image_shape=img.shape, stride=(2, 3))
        assert bit_equal(out, img)

    def test_float32_overlap_within_pixel_bound(self) -> None:
        """Amendment A: outside the predicate the error is bounded per pixel
        by (k+1)*eps*|v| — it grows with the coverage count, so no fixed ULP
        figure applies. (Was: `rtol=1e-5` on a p4 s2 geometry, which is
        *inside* the predicate and therefore bit-exact — the old assertion
        tested nothing about the error regime.)"""
        img = rand_image(1, 16, 16, torch.float32, seed=204)
        patches = extract(img, patch_size=4, stride=1)
        out = reconstruct(patches, image_shape=img.shape, stride=1)
        assert within_pixel_bound(out, img, coverage_counts(16, 16, 4, 4, 1, 1))
```

Em `TestRoundtripPreservation`: manter `test_dtype_preserved` e `test_device_preserved_cpu` com `_ramp` (valor irrelevante). Trocar dados nas duas que comparam valor:

```python
    @pytest.mark.gpu
    def test_cuda_roundtrip(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        img = rand_image(1, 8, 8, torch.float32, seed=106).cuda()
        patches = extract(img, patch_size=4, stride=4)
        out = reconstruct(patches, image_shape=img.shape, stride=4)
        assert out.device.type == "cuda"
        assert bit_equal(out, img)

    def test_accepts_torch_size(self) -> None:
        """torch.Tensor.shape returns torch.Size (a tuple subclass)."""
        img = rand_image(1, 8, 8, torch.float32, seed=107)
        patches = extract(img, patch_size=4, stride=4)
        out = reconstruct(patches, image_shape=img.shape, stride=4)
        assert isinstance(img.shape, torch.Size)
        assert bit_equal(out, img)
```

Em `TestCoverageGuard.test_exact_coverage_boundary_still_accepted` (geometria 10×10 p4 s2 ⇒ contagens {1,2} ⇒ dentro do predicado):

```python
    def test_exact_coverage_boundary_still_accepted(self) -> None:
        """Grid that ends exactly on the image edge must not raise."""
        img = rand_image(1, 10, 10, torch.float32, seed=301)
        patches = extract(img, patch_size=4, stride=2)  # (4-1)*2+4 == 10
        out = reconstruct(patches, image_shape=img.shape, stride=2)
        assert bit_equal(out, img)
```

- [ ] **Step 2: rodar o arquivo**

Run: `uv run pytest tests/test_reconstruct.py -v`
Expected: tudo verde (incluindo os 4 casos reescritos de overlap).

- [ ] **Step 3: reescrever `TestTilingsRoundtripGuarantee` em `tests/test_geometry.py`**

Adicionar import: `from tests._rng import bit_equal, count_map_pow2, coverage_counts, rand_image, within_pixel_bound`

```python
class TestTilingsRoundtripGuarantee:
    """Every spec from tilings() round-trips; the assertion written is the
    contract itself (ADR 0003): bit-exact inside the predicate, per-pixel
    bound outside it. A flat `allclose` cannot tell the two regimes apart."""

    def test_all_exact_tilings_28x28_roundtrip(self) -> None:
        img = rand_image(1, 28, 28, torch.float64, seed=401)
        for spec in tilings((28, 28)):
            patches = extract(img, patch_size=spec.patch_size, stride=spec.stride)
            assert patches.shape[0] == spec.total_patches
            recon = reconstruct(patches, image_shape=img.shape, stride=spec.stride)
            assert bit_equal(recon, img), f"spec {spec} broke bit-exact round-trip"

    def test_overlap_tilings_28x28_roundtrip_predicate_split(self) -> None:
        img = rand_image(1, 28, 28, torch.float64, seed=402)
        for spec in tilings((28, 28), allow_overlap=True):
            ph, pw = spec.patch_size
            sh, sw = spec.stride
            patches = extract(img, patch_size=spec.patch_size, stride=spec.stride)
            recon = reconstruct(patches, image_shape=img.shape, stride=spec.stride)
            if count_map_pow2(28, 28, ph, pw, sh, sw):
                assert bit_equal(recon, img), f"spec {spec} inside predicate, not exact"
            else:
                counts = coverage_counts(28, 28, ph, pw, sh, sw)
                assert within_pixel_bound(recon, img, counts), (
                    f"spec {spec} outside predicate, beyond (k+1)*eps*|v|"
                )
```

- [ ] **Step 4: rodar o arquivo**

Run: `uv run pytest tests/test_geometry.py -v`
Expected: tudo verde.

- [ ] **Step 5: reescrever os casos de valor em `tests/test_stitch.py`**

Adicionar import: `from tests._rng import bit_equal, rand_image`

`TestUniformEqualsReconstruct` vira:

```python
class TestUniformEqualsReconstruct:
    """`weight="uniform"` is mathematically equivalent to `reconstruct`."""

    def test_exact_tiling_bit_exact(self) -> None:
        img = rand_image(1, 16, 16, torch.float32, seed=501)
        patches = extract(img, patch_size=4, stride=4)
        out_stitch = stitch(patches, image_shape=img.shape, stride=4)
        out_recon = reconstruct(patches, image_shape=img.shape, stride=4)
        assert bit_equal(out_stitch, out_recon)

    def test_exact_tiling_recovers_image(self) -> None:
        img = rand_image(2, 12, 12, torch.float32, seed=502)
        patches = extract(img, patch_size=4, stride=4)
        out = stitch(patches, image_shape=img.shape, stride=4, weight="uniform")
        assert bit_equal(out, img)

    def test_overlap_matches_reconstruct(self) -> None:
        """Equivalence is down to floating-point ordering by contract, so
        this stays `allclose` — but on real noise, not an integer ramp."""
        img = rand_image(1, 16, 16, torch.float64, seed=503)
        patches = extract(img, patch_size=4, stride=2)
        out_stitch = stitch(patches, image_shape=img.shape, stride=2)
        out_recon = reconstruct(patches, image_shape=img.shape, stride=2)
        assert torch.allclose(out_stitch, out_recon, rtol=1e-12, atol=1e-12)

    def test_overlap_recovers_image(self) -> None:
        """counts are 1, 2, 4 (powers of two): uniform stitch is bit-exact
        here — the ones-kernel multiply is exact and the cumsum denominator
        reproduces the integer count map exactly."""
        img = rand_image(1, 16, 16, torch.float64, seed=504)
        patches = extract(img, patch_size=4, stride=2)
        out = stitch(patches, image_shape=img.shape, stride=2)
        assert bit_equal(out, img)

    def test_rectangular_geometry(self) -> None:
        img = rand_image(1, 20, 30, torch.float64, seed=505)
        patches = extract(img, patch_size=(4, 6), stride=(4, 6))
        out = stitch(patches, image_shape=img.shape, stride=(4, 6))
        assert bit_equal(out, img)
```

Nos testes de janela (valor importa; tolerância 1e-9 permanece porque janela não-uniforme faz aritmética de verdade), trocar só a linha do `img`:

- `TestHann.test_unmodified_overlap_recovers_image` (:63): `img = rand_image(1, 16, 16, torch.float64, seed=511)`
- `TestHann.test_exact_tiling_recovers_image` (:73): `img = rand_image(1, 12, 12, torch.float64, seed=512) + 1.0` (manter o `+ 1.0`: o caso histórico é de valor longe de zero)
- `TestHann.test_no_zeroed_pixels_with_overlap` (:81): `img = rand_image(1, 13, 13, torch.float64, seed=513) + 1.0`
- `TestHann.test_patch_size_2_not_degenerate` (:89): `img = rand_image(1, 8, 8, torch.float64, seed=514)`
- `TestGaussian.test_unmodified_recovers_full_image` (:122): `img = rand_image(1, 16, 16, torch.float64, seed=521)`

`TestPreservation.test_cuda_roundtrip` e `test_accepts_torch_size` (ambos `stride == patch_size`):

```python
    @pytest.mark.gpu
    def test_cuda_roundtrip(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        img = rand_image(1, 8, 8, torch.float32, seed=506).cuda()
        patches = extract(img, patch_size=4, stride=4)
        out = stitch(patches, image_shape=img.shape, stride=4)
        assert out.device.type == "cuda"
        assert bit_equal(out, img)

    def test_accepts_torch_size(self) -> None:
        img = rand_image(1, 8, 8, torch.float32, seed=507)
        patches = extract(img, patch_size=4, stride=4)
        out = stitch(patches, image_shape=img.shape, stride=4)
        assert isinstance(img.shape, torch.Size)
        assert bit_equal(out, img)
```

Não tocar: `test_center_pixel_weighted_more_than_edge` (posicional, `pytest.approx`), `test_no_corner_artifact_at_exact_tiling` (constante 0.5), kernels, rejects, caracterização 0.3.0 (`_stitch_reference`, já usa `randn` no dtype alvo).

- [ ] **Step 6: rodar o arquivo**

Run: `uv run pytest tests/test_stitch.py -v`
Expected: tudo verde.

- [ ] **Step 7: lint + suite completa + commit**

Run: `uv run ruff check tests/ && uv run pytest -m "not gpu" -q`
Expected: ruff limpo; suite verde.

```bash
git add tests/test_reconstruct.py tests/test_geometry.py tests/test_stitch.py
git commit -m "test(g1): round-trips on seeded full-mantissa noise; per-pixel bound replaces rtol=1e-5"
```

---

### Task 3: `tests/test_exactness.py` — a suíte falsificadora

**Files:**
- Create: `tests/test_exactness.py`

**Interfaces:**
- Consumes: todos os helpers de `tests/_rng.py` (Task 1); `extract`, `reconstruct` de `patchcraft`.
- Produces: nada para outras tasks; a verificação de mutação da Task 10 depende deste arquivo existir.

Criar o arquivo completo:

```python
"""Falsification suite for the bit-exactness predicate (FOCO §5, ADR 0003).

Predicate: the ``extract``/``reconstruct`` round trip is bit-exact iff every
value of the overlap count map is a power of two. Outside it, the per-pixel
error is bounded by ``(k+1)*eps*|v|`` with ``k`` the pixel's coverage count
(Amendment A of docs/superpowers/specs/2026-09-01-fase3-g1-predicado-design.md
— the frozen "1 ULP" wording was measured false).

Strategy: enumerate the legal geometry space *independently* of the
predicate (H, W in 4..24; ph, pw in 2..9; strides with exact coverage —
126,736 geometries), draw a seeded 256-geometry sample (67 inside the
predicate, 189 outside), and try to break both halves:

- inside  -> bit-exact for every seed of a fixed set;
- outside -> at least one of 50 seeds comes back inexact (exactness outside
  the predicate is a property of the data, not of the geometry: measured
  63/300 exact seeds in float32 at the FOCO anchor geometry, so a single
  execution proves nothing) AND every seed stays within the pixel bound.

``PATCHCRAFT_SWEEP_FULL=1`` replaces the sample with the full space — the
local gate before merging (recipe vs count map over all 126,736, plus one
bit-exact round trip per inside-predicate geometry; ~1-2 min).
"""
from __future__ import annotations

import os
import random

import pytest
import torch

from patchcraft import extract, reconstruct
from tests._rng import (
    bit_equal,
    count_map_pow2,
    coverage_counts,
    exact_axes_pow2,
    rand_image,
    within_pixel_bound,
)

Geometry = tuple[int, int, int, int, int, int]  # (h, w, ph, pw, sh, sw)

_SAMPLE_SIZE = 256
_SAMPLE_SEED = 20260901
_POSITIVE_SEEDS = range(5)
_NEGATIVE_SEEDS = range(50)
_DTYPES = [torch.float32, torch.float64]


def _legal_geometries() -> list[Geometry]:
    """Every (h, w, ph, pw, sh, sw) with exact coverage: H, W in 4..24,
    ph, pw in 2..9 (and <= the axis), 1 <= s <= p with (n - p) % s == 0 on
    each axis independently (rectangular included). No hand-picked lists."""
    out: list[Geometry] = []
    for h in range(4, 25):
        for w in range(4, 25):
            for ph in range(2, min(9, h) + 1):
                for pw in range(2, min(9, w) + 1):
                    for sh in range(1, ph + 1):
                        if (h - ph) % sh != 0:
                            continue
                        for sw in range(1, pw + 1):
                            if (w - pw) % sw == 0:
                                out.append((h, w, ph, pw, sh, sw))
    return out


_SPACE = _legal_geometries()
_SAMPLE = random.Random(_SAMPLE_SEED).sample(_SPACE, _SAMPLE_SIZE)
_POSITIVE = [g for g in _SAMPLE if count_map_pow2(*g)]
_NEGATIVE = [g for g in _SAMPLE if not count_map_pow2(*g)]


def _roundtrip(
    g: Geometry, dtype: torch.dtype, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    h, w, ph, pw, sh, sw = g
    img = rand_image(1, h, w, dtype, seed)
    out = reconstruct(
        extract(img, patch_size=(ph, pw), stride=(sh, sw)),
        image_shape=img.shape,
        stride=(sh, sw),
    )
    return out, img


class TestSampleShape:
    def test_space_size(self) -> None:
        assert len(_SPACE) == 126_736

    def test_sample_has_both_halves(self) -> None:
        # Seeded, so stable: 67 inside the predicate, 189 outside.
        assert (len(_POSITIVE), len(_NEGATIVE)) == (67, 189)


class TestRecipeMatchesPredicate:
    @pytest.mark.parametrize("g", _SAMPLE, ids=repr)
    def test_axes_recipe_equals_count_map(self, g: Geometry) -> None:
        assert exact_axes_pow2(*g) == count_map_pow2(*g)


class TestPositiveHalf:
    @pytest.mark.parametrize("g", _POSITIVE, ids=repr)
    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_bit_exact_every_seed(self, g: Geometry, dtype: torch.dtype) -> None:
        for seed in _POSITIVE_SEEDS:
            out, img = _roundtrip(g, dtype, seed)
            assert bit_equal(out, img), (g, dtype, seed)


class TestNegativeHalf:
    @pytest.mark.parametrize("g", _NEGATIVE, ids=repr)
    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_some_seed_inexact_and_all_within_bound(
        self, g: Geometry, dtype: torch.dtype
    ) -> None:
        h, w, ph, pw, sh, sw = g
        counts = coverage_counts(h, w, ph, pw, sh, sw)
        inexact = 0
        for seed in _NEGATIVE_SEEDS:
            out, img = _roundtrip(g, dtype, seed)
            assert within_pixel_bound(out, img, counts), (g, dtype, seed)
            if not bit_equal(out, img):
                inexact += 1
        assert inexact >= 1, (
            f"{g} {dtype}: exact on all 50 seeds — either the predicate grew "
            "or the fixed sample got lucky; investigate before touching this"
        )


class TestNanInsidePredicate:
    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_nan_roundtrips_bit_exact(self, dtype: torch.dtype) -> None:
        """D5 in code: inside the predicate the bits come back even when the
        value is NaN — where torch.equal is not reflexive."""
        img = rand_image(1, 16, 16, dtype, seed=99)
        img[0, 3, 5] = float("nan")
        img[0, 10, 11] = float("nan")
        out = reconstruct(
            extract(img, patch_size=(4, 4), stride=(2, 2)),
            image_shape=img.shape,
            stride=(2, 2),
        )
        assert not torch.equal(out, img)  # NaN != NaN
        assert bit_equal(out, img)


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("PATCHCRAFT_SWEEP_FULL") != "1",
    reason="full 126,736-geometry sweep is a local gate: PATCHCRAFT_SWEEP_FULL=1",
)
class TestFullSweep:
    def test_recipe_matches_count_map_everywhere(self) -> None:
        for g in _SPACE:
            assert exact_axes_pow2(*g) == count_map_pow2(*g), g

    def test_no_counterexample_inside_predicate(self) -> None:
        for g in _SPACE:
            if not count_map_pow2(*g):
                continue
            out, img = _roundtrip(g, torch.float32, seed=1)
            assert bit_equal(out, img), f"counterexample: {g}"
```

- [ ] **Step 1: rodar a suíte nova (modo amostra)**

Run: `uv run pytest tests/test_exactness.py -q`
Expected: ~772 testes, todos verdes, em menos de ~2 min. Se a metade negativa falhar com "exact on all 50 seeds" em alguma geometria, **não** enfraquecer o teste — reportar (é um achado, não um flake).

- [ ] **Step 2: rodar também no modo puro**

Run: `PATCHCRAFT_ACCEL=0 uv run pytest tests/test_exactness.py -q`
Expected: idem.

- [ ] **Step 3: lint + commit**

Run: `uv run ruff check tests/test_exactness.py`

```bash
git add tests/test_exactness.py
git commit -m "test(g1): falsification suite for the count-map predicate (sampled + full sweep gate)"
```

---

### Task 4: `tests/test_reference.py` — referência naive (consumer gate, D6)

**Files:**
- Create: `tests/test_reference.py`

**Interfaces:**
- Consumes: helpers de `tests/_rng.py`; `extract`, `reconstruct` de `patchcraft`.
- Produces: nada para outras tasks.

Criar o arquivo completo:

```python
"""Naive reference implementation of extract+reconstruct (FOCO §2, D6).

The substitute for the lost `hand.py` x `pc.py` consumer gate: pure Python
loops, no `F.fold`, no code shared with the library. If the fast paths and
this reference agree bit for bit inside the predicate — and both stay within
the per-pixel bound outside it — the arithmetic, not just the API, is right.

Runs in both accel modes on purpose: `reconstruct` dispatches internally, so
an active accelerator is exercised here without any extra code (the full
accel x pure equivalence grid lives in tests/test_accel.py).
"""
from __future__ import annotations

import pytest
import torch

from patchcraft import extract, reconstruct
from tests._rng import (
    bit_equal,
    count_map_pow2,
    coverage_counts,
    rand_image,
    within_pixel_bound,
)

Geometry = tuple[int, int, int, int, int, int]  # (h, w, ph, pw, sh, sw)

_GRID: list[Geometry] = [
    (8, 8, 4, 4, 4, 4),     # exact tile
    (16, 16, 4, 4, 2, 2),   # pow2 overlap (counts 1, 2, 4)
    (9, 9, 3, 3, 3, 3),     # exact tile
    (13, 13, 4, 4, 3, 3),   # pow2 overlap, counts {1, 2}
    (10, 10, 4, 4, 2, 2),   # pow2 overlap
    (12, 18, 4, 6, 2, 3),   # pow2 rectangular
    (7, 7, 3, 3, 2, 2),     # pow2, counts {1, 2}
    (14, 14, 4, 4, 1, 1),   # FOCO anchor: outside (counts include 3)
    (9, 9, 4, 4, 1, 1),     # outside: counts {1, 2, 3, 4}
    (24, 24, 9, 9, 3, 3),   # outside: counts include 3
]


def _ref_extract(
    img: torch.Tensor, ph: int, pw: int, sh: int, sw: int
) -> torch.Tensor:
    """Slice every patch out pixel region by pixel region; stack row-major."""
    c, h, w = img.shape
    patches = [
        img[:, y : y + ph, x : x + pw].clone()
        for y in range(0, h - ph + 1, sh)
        for x in range(0, w - pw + 1, sw)
    ]
    return torch.stack(patches)


def _ref_reconstruct(
    patches: torch.Tensor, c: int, h: int, w: int, ph: int, pw: int,
    sh: int, sw: int,
) -> torch.Tensor:
    """Accumulate sum and count per pixel in the input dtype, then divide.
    Row-major ascending patch order, one slice-add per patch."""
    acc = torch.zeros(c, h, w, dtype=patches.dtype)
    cnt = torch.zeros(c, h, w, dtype=patches.dtype)
    k = 0
    for y in range(0, h - ph + 1, sh):
        for x in range(0, w - pw + 1, sw):
            acc[:, y : y + ph, x : x + pw] += patches[k]
            cnt[:, y : y + ph, x : x + pw] += 1
            k += 1
    return acc / cnt


@pytest.mark.parametrize("g", _GRID, ids=repr)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_reference_matches_library(g: Geometry, dtype: torch.dtype) -> None:
    h, w, ph, pw, sh, sw = g
    img = rand_image(3, h, w, dtype, seed=17)

    ref_patches = _ref_extract(img, ph, pw, sh, sw)
    lib_patches = extract(img, patch_size=(ph, pw), stride=(sh, sw))
    # extract is a pure gather: bits must survive identically.
    assert bit_equal(lib_patches, ref_patches)

    ref = _ref_reconstruct(ref_patches, 3, h, w, ph, pw, sh, sw)
    lib = reconstruct(lib_patches, image_shape=img.shape, stride=(sh, sw))
    if count_map_pow2(*g):
        # Any summation order of k identical values with k a power of two is
        # exact, so the two implementations agree bit for bit.
        assert bit_equal(lib, ref)
    else:
        counts = coverage_counts(h, w, ph, pw, sh, sw)
        assert within_pixel_bound(lib, img, counts)
        assert within_pixel_bound(ref, img, counts)
```

- [ ] **Step 1: rodar**

Run: `uv run pytest tests/test_reference.py -v`
Expected: 20 passed. Verde também com `PATCHCRAFT_ACCEL=0`.

- [ ] **Step 2: lint + commit**

```bash
uv run ruff check tests/test_reference.py
git add tests/test_reference.py
git commit -m "test(g1): naive loop-based reference for extract+reconstruct (consumer gate)"
```

---

### Task 5: `tests/test_public_api.py` — superfície congelada (B3/D4)

**Files:**
- Create: `tests/test_public_api.py`

**Interfaces:**
- Consumes: nada de tasks anteriores.
- Produces: o alvo da mutação `__all__` da Task 10.

Assinaturas verificadas contra o head 0.4.0 via `inspect` (os módulos usam `from __future__ import annotations`, então `return_annotation` é string). Criar o arquivo completo:

```python
"""Frozen public surface (B3/D4): 20 names, signatures, carrier fields.

A failure here means the 1.0-freeze surface moved. Changing it on purpose
means updating this file in the same commit and noting it in the CHANGELOG.
"""
from __future__ import annotations

import inspect
import typing
from dataclasses import fields

import patchcraft
from patchcraft import Cache, Patchify

P = inspect.Parameter
POK = P.POSITIONAL_OR_KEYWORD
KO = P.KEYWORD_ONLY
VP = P.VAR_POSITIONAL

EXPECTED_ALL = {
    "Cache",
    "PairedTilingSpec",
    "PatchMeta",
    "PatchPair",
    "Patchify",
    "TilingSpec",
    "WeightKind",
    "accel_available",
    "extract",
    "num_patches",
    "pair",
    "paired_tilings",
    "patch_metrics",
    "per_patch_mse",
    "per_patch_psnr",
    "reconstruct",
    "resize",
    "scale_factor",
    "stitch",
    "tilings",
}
# 20 names: the 19 of FOCO-1.0.md plus `accel_available` (0.4.0).


def _params(fn: object) -> list[tuple[object, ...]]:
    """(name, kind, has_default, default_or_None) per parameter."""
    out: list[tuple[object, ...]] = []
    for p in inspect.signature(fn).parameters.values():  # type: ignore[union-attr]
        out.append((
            p.name,
            p.kind,
            p.default is not P.empty,
            None if p.default is P.empty else p.default,
        ))
    return out


def _returns(fn: object) -> object:
    return inspect.signature(fn).return_annotation  # type: ignore[union-attr]


class TestAll:
    def test_exact_set(self) -> None:
        assert set(patchcraft.__all__) == EXPECTED_ALL

    def test_no_duplicates(self) -> None:
        assert len(patchcraft.__all__) == len(EXPECTED_ALL)

    def test_every_name_reachable(self) -> None:
        for name in EXPECTED_ALL:
            assert getattr(patchcraft, name) is not None


class TestSignatures:
    def test_extract(self) -> None:
        assert _params(patchcraft.extract) == [
            ("image", POK, False, None),
            ("patch_size", POK, False, None),
            ("stride", POK, False, None),
            ("dilation", POK, True, 1),
        ]
        assert _returns(patchcraft.extract) == "torch.Tensor"

    def test_reconstruct(self) -> None:
        assert _params(patchcraft.reconstruct) == [
            ("patches", POK, False, None),
            ("image_shape", POK, False, None),
            ("stride", POK, False, None),
            ("dilation", POK, True, 1),
        ]
        assert _returns(patchcraft.reconstruct) == "torch.Tensor"

    def test_stitch(self) -> None:
        assert _params(patchcraft.stitch) == [
            ("patches", POK, False, None),
            ("image_shape", POK, False, None),
            ("stride", POK, False, None),
            ("weight", KO, True, "uniform"),
            ("dilation", KO, True, 1),
        ]
        assert _returns(patchcraft.stitch) == "torch.Tensor"

    def test_resize(self) -> None:
        assert _params(patchcraft.resize) == [
            ("image", POK, False, None),
            ("target_size", POK, False, None),
            ("backend", POK, True, "pil"),
            ("resample", POK, True, None),
        ]
        assert _returns(patchcraft.resize) == "torch.Tensor | PILImage"

    def test_pair(self) -> None:
        assert _params(patchcraft.pair) == [
            ("lr_image", POK, False, None),
            ("hr_image", POK, False, None),
            ("lr_patch_size", POK, False, None),
            ("scale_factor", POK, False, None),
            ("stride", POK, False, None),
            ("image_id", KO, True, None),
        ]
        assert _returns(patchcraft.pair) == "PatchPair"

    def test_num_patches(self) -> None:
        assert _params(patchcraft.num_patches) == [
            ("image_shape", POK, False, None),
            ("patch_size", POK, False, None),
            ("stride", POK, False, None),
            ("dilation", POK, True, 1),
        ]
        assert _returns(patchcraft.num_patches) == "tuple[int, int]"

    def test_tilings(self) -> None:
        assert _params(patchcraft.tilings) == [
            ("image_shape", POK, False, None),
            ("allow_overlap", KO, True, False),
            ("min_patch_size", KO, True, 2),
            ("max_patch_size", KO, True, None),
        ]
        assert _returns(patchcraft.tilings) == "list[TilingSpec]"

    def test_paired_tilings(self) -> None:
        assert _params(patchcraft.paired_tilings) == [
            ("lr_shape", POK, False, None),
            ("hr_shape", POK, False, None),
            ("allow_overlap", KO, True, False),
            ("min_patch_size", KO, True, 2),
            ("max_patch_size", KO, True, None),
        ]
        assert _returns(patchcraft.paired_tilings) == "list[PairedTilingSpec]"

    def test_scale_factor(self) -> None:
        assert _params(patchcraft.scale_factor) == [
            ("lr_shape", POK, False, None),
            ("hr_shape", POK, False, None),
        ]
        assert _returns(patchcraft.scale_factor) == "int | None"

    def test_patch_metrics(self) -> None:
        assert _params(patchcraft.patch_metrics) == [
            ("a", POK, False, None),
            ("b", POK, False, None),
            ("max_value", KO, True, 1.0),
        ]
        assert _returns(patchcraft.patch_metrics) == "dict[str, float]"

    def test_per_patch_mse(self) -> None:
        assert _params(patchcraft.per_patch_mse) == [
            ("a", POK, False, None),
            ("b", POK, False, None),
        ]
        assert _returns(patchcraft.per_patch_mse) == "torch.Tensor"

    def test_per_patch_psnr(self) -> None:
        assert _params(patchcraft.per_patch_psnr) == [
            ("a", POK, False, None),
            ("b", POK, False, None),
            ("max_value", KO, True, 1.0),
        ]
        assert _returns(patchcraft.per_patch_psnr) == "torch.Tensor"

    def test_accel_available(self) -> None:
        assert _params(patchcraft.accel_available) == []
        assert _returns(patchcraft.accel_available) == "bool"


class TestCarriers:
    def test_tiling_spec_fields(self) -> None:
        assert patchcraft.TilingSpec._fields == (
            "patch_size", "stride", "dilation",
            "num_patches", "total_patches", "overlap",
        )

    def test_paired_tiling_spec_fields(self) -> None:
        assert patchcraft.PairedTilingSpec._fields == ("lr", "hr", "scale_factor")

    def test_patch_pair_fields(self) -> None:
        assert [f.name for f in fields(patchcraft.PatchPair)] == [
            "lr_patches", "hr_patches", "metas",
        ]

    def test_patch_meta_fields(self) -> None:
        assert [f.name for f in fields(patchcraft.PatchMeta)] == [
            "patch_index", "row", "col",
            "lr_patch_size", "hr_patch_size", "image_id",
        ]


class TestCacheSurface:
    def test_init_signature(self) -> None:
        assert _params(Cache.__init__) == [
            ("self", POK, False, None),
            ("root", POK, False, None),
            ("namespace", POK, False, None),
            ("version", POK, True, 1),
        ]

    def test_method_signatures(self) -> None:
        assert _params(Cache.key_for) == [("self", POK, False, None), ("parts", VP, False, None)]
        assert _returns(Cache.key_for) == "str"
        assert _params(Cache.put) == [
            ("self", POK, False, None),
            ("key", POK, False, None),
            ("data", POK, False, None),
        ]
        assert _params(Cache.get) == [("self", POK, False, None), ("key", POK, False, None)]
        assert _returns(Cache.get) == "bytes | None"

    def test_properties(self) -> None:
        for name in ("root", "namespace", "version"):
            assert isinstance(inspect.getattr_static(Cache, name), property)


class TestPatchifySurface:
    def test_init_signature(self) -> None:
        # positional-or-keyword on purpose (verified at head): Patchify is a
        # transforms-style callable, `Patchify(4, 2)` is idiomatic.
        assert _params(Patchify.__init__) == [
            ("self", POK, False, None),
            ("patch_size", POK, False, None),
            ("stride", POK, False, None),
            ("dilation", POK, True, 1),
        ]


class TestWeightKind:
    def test_current_values(self) -> None:
        # D3: the set is open (new windows are a compatible addition); this
        # pins the starting point, not an exhaustive contract.
        assert typing.get_args(patchcraft.WeightKind) == ("uniform", "hann", "gaussian")
```

- [ ] **Step 1: rodar**

Run: `uv run pytest tests/test_public_api.py -v`
Expected: ~25 passed.

- [ ] **Step 2: lint + commit**

```bash
uv run ruff check tests/test_public_api.py
git add tests/test_public_api.py
git commit -m "test(g1): freeze the 20-name public surface with signatures and carrier fields"
```

---

### Task 6: B6 — enumeração sem specs degenerados + assimetria escrita

**Files:**
- Modify: `src/patchcraft/geometry.py` (:27-33 docstring, :178-190 guarda)
- Modify: `tests/test_geometry.py` (novos testes em `TestTilingsOverlap` e `TestPairedTilingsAccepts`)
- Modify: `docs/THEORY.md` §9.1 (após :261)
- Modify: `tests/test_extract.py` (:113-118, só docstring)
- Modify: `docs/USAGE.md` (:84-86, saída `100` → `73`)

**Interfaces:**
- Consumes: nada de tasks anteriores.
- Produces: comportamento novo de `tilings`/`paired_tilings` com `allow_overlap=True` (verificado pelos testes novos).

- [ ] **Step 1: testes primeiro (falham antes da guarda)**

Adicionar em `TestTilingsOverlap` (`tests/test_geometry.py`):

```python
    def test_28x28_overlap_count_after_degenerate_removal(self) -> None:
        """0.5.0 (D1): single-patch overlap specs (27 of the old 100) had no
        observable stride and duplicated the exact tile; no longer emitted."""
        assert len(tilings((28, 28), allow_overlap=True)) == 73

    def test_no_single_patch_overlap_specs(self) -> None:
        for shape in [(28, 28), (20, 30), (7, 7), (9, 12)]:
            for t in tilings(shape, allow_overlap=True):
                assert not (t.total_patches == 1 and t.overlap)

    def test_every_spec_covers_exactly(self) -> None:
        for t in tilings((28, 28), allow_overlap=True):
            (ph, pw), (sh, sw) = t.patch_size, t.stride
            nh, nw = t.num_patches
            assert (nh - 1) * sh + ph == 28
            assert (nw - 1) * sw + pw == 28
```

Adicionar em `TestPairedTilingsAccepts`:

```python
    def test_overlap_count_after_degenerate_removal(self) -> None:
        """LR side carried 13 degenerate single-patch specs (p == 14, s < 14):
        40 -> 27 pairs with overlap enabled (inherits the tilings fix)."""
        assert len(paired_tilings((14, 14), (28, 28), allow_overlap=True)) == 27
```

Run: `uv run pytest tests/test_geometry.py -k "degenerate or single_patch or covers_exactly" -v`
Expected: FAIL (`test_28x28_overlap_count_after_degenerate_removal`: 100 != 73).

- [ ] **Step 2: guarda em `geometry.py`**

No ramo `allow_overlap` de `tilings` (:178-190), pular o caso degenerado:

```python
        if allow_overlap:
            for s in range(1, p):
                if (h - p) % s == 0 and (w - p) % s == 0:
                    nh = (h - p) // s + 1
                    nw = (w - p) // s + 1
                    if nh == 1 and nw == 1:
                        # Degenerate single-patch geometry: with one patch the
                        # stride is unobservable and the spec is a semantic
                        # duplicate of the exact tile (p, p)/(p, p), always
                        # emitted for the same p. Skip it (0.5.0, D1).
                        continue
                    results.append(TilingSpec(
                        patch_size=(p, p),
                        stride=(s, s),
                        dilation=(1, 1),
                        num_patches=(nh, nw),
                        total_patches=nh * nw,
                        overlap=True,
                    ))
```

E o docstring de `TilingSpec` (:27-33) vira:

```python
class TilingSpec(NamedTuple):
    """One valid patch geometry for an image.

    ``overlap=False`` means an *exact tile*: ``patch_size == stride`` and the
    image is divided into a clean grid with no overlap and no waste.
    ``overlap=True`` means ``stride < patch_size`` **with more than one
    patch**, so adjacent patches share pixels while the grid still covers the
    image exactly. (A single-patch grid has no observable stride; emitting it
    as ``overlap=True`` duplicated the exact tile, so it is skipped.)
    """
```

- [ ] **Step 3: rodar e ver passar**

Run: `uv run pytest tests/test_geometry.py -v`
Expected: tudo verde (incl. os dois testes novos e a suíte de round-trip da Task 2).

- [ ] **Step 4: docstring da trunca em `tests/test_extract.py`**

`test_truncation_drops_trailing` (:113-118) ganha a referência à política escrita (D2 não muda comportamento):

```python
    def test_truncation_drops_trailing(self) -> None:
        """Image 10x10, patch 4, stride 3 → trailing column dropped.

        Truncation is the written boundary policy (ADR 0001, THEORY §9.1
        "Assimetria ida/volta"): `extract` drops silently, `reconstruct`
        raises on the same uncovered grid (§9.2). This test pins the silent
        side; do not "fix" it into a raise without an ADR."""
        img = _ramp(1, 10, 10)
        # num_h = num_w = (10 - 4) // 3 + 1 = 3
        out = extract(img, patch_size=4, stride=3)
        assert out.shape == (9, 1, 4, 4)
```

- [ ] **Step 5: assimetria no THEORY §9.1**

Após o bloco "**Fora de escopo v0.1:**" de §9.1 (fecha em :261, "- Devices não-CUDA acelerados..."), adicionar o parágrafo:

```markdown
**Assimetria ida/volta (extract trunca, reconstruct recusa).** Na ida, uma geometria que não cobre a imagem inteira **trunca**: as linhas/colunas finais que não fecham um patch são descartadas sem aviso — é a política de borda do ADR 0001, e o caso de uso principal (extração para treino sobre imagens de tamanho arbitrário) depende dela. Na volta, a mesma situação é `ValueError` (§9.2): remontar com cobertura parcial sintetizaria pixels, que é o que a biblioteca recusa por princípio. Quem precisa de cobertura exata confere antes com `num_patches`/`tilings` (§9.6); quem precisa de tamanho arbitrário com round-trip faz padding explícito no caller (o helper `pad` é candidata nomeada da 1.1, FOCO §3).
```

- [ ] **Step 6: USAGE.md — contagem da enumeração**

`docs/USAGE.md` :83-86. O exemplo mostra a contagem antiga:

```
>>> len(tilings((28, 28), allow_overlap=True))
100
```

vira `73`, e o texto logo abaixo (:88-91) ganha a nota de patch único:

```
With `allow_overlap=True` the function also emits `stride < patch_size`
geometries where `(H - p) % s == 0` (clean-edge overlap; single-patch
grids are skipped, since one patch makes the stride unobservable).
Useful when you want training data with stride < ph; whether the
round-trip stays bit-exact then depends on the count map (see §5).
```

(A troca "reconstruction must still be exact" → frase condicionada já é B1; fica aqui porque a mesma região está sendo editada — ver Amendments, A7.)

- [ ] **Step 7: lint + mypy + suite + commit**

Run: `uv run ruff check src tests && uv run mypy src && uv run pytest -m "not gpu" -q`
Expected: limpo e verde.

```bash
git add src/patchcraft/geometry.py tests/test_geometry.py tests/test_extract.py docs/THEORY.md docs/USAGE.md
git commit -m "fix(g1): skip degenerate single-patch overlap specs in tilings (B6/D1); write the extract/reconstruct asymmetry (D2)"
```

---

### Task 7: B1 — o predicado qualificado nos pontos restantes

**Files:**
- Modify: `src/patchcraft/reconstruct.py` (:40-42, docstring)
- Modify: `src/patchcraft/stitch.py` (:3-5, :123-124, docstrings)
- Modify: `docs/SCOPE.md` (:229-230)
- Modify: `docs/THEORY.md` (:100, :153, :157)
- Modify: `docs/GUIDE.md` (:433, :441-443)
- Modify: `docs/USAGE.md` (:149, :158-163)
- Modify: `README.md` (:86), `README.pt-BR.md` (:86)

**Interfaces:**
- Consumes: nada de código; a redação segue D5/Amendment A.
- Produces: o alvo do grep de verificação (Step final).

Todas as edições abaixo são `old` → `new` exatos (verificar o contexto com Read antes de editar; números de linha são do head 0.4.0).

- [ ] **Step 1: `src/patchcraft/reconstruct.py` docstring** (entra por Amendment A — o ponto não constava da lista de auditoria da spec, ver Amendments A4)

old:
```
    Outside the rule the error is about 1 ULP, measured at 2.4e-07 in float32 on
    a 16x16 image with ``patch_size=4, stride=1``. Widening the dtype does not
    help, because the deciding axis is the geometry rather than the precision.
```
new:
```
    Outside the rule the per-pixel error is bounded by ``(k + 1) * eps * |v|``,
    where ``k`` is that pixel's coverage count — the error grows with the
    overlap, so there is no fixed ULP figure (measured up to 19 ULP at k=81 in
    float32). Widening the dtype does not help, because the deciding axis is
    the geometry rather than the precision.
```

- [ ] **Step 2: `src/patchcraft/stitch.py` docstrings**

old (:3-5):
```
Where :func:`patchcraft.reconstruct` is a bit-exact inverse of ``extract``,
``stitch`` is intended for *modified* patches, meaning patches that have been
```
new:
```
Where :func:`patchcraft.reconstruct` inverts ``extract`` exactly under the
count-map rule (every coverage count a power of two — always true at
``stride == patch_size``), ``stitch`` is intended for *modified* patches,
meaning patches that have been
```

old (:123-124):
```
    super-resolved). Use :func:`patchcraft.reconstruct` when patches came
    straight from ``extract`` and you want a bit-exact inverse with no
    extra arithmetic.
```
new:
```
    super-resolved). Use :func:`patchcraft.reconstruct` when patches came
    straight from ``extract`` and you want the exact inverse — bit-exact
    under the count-map rule — with no extra arithmetic.
```

- [ ] **Step 3: `docs/SCOPE.md`**

old (:229-230):
```
- **Contract divergence.** `reconstruct` is bit-exact for unmodified
  patches; `stitch` is explicitly an interpolated blend that assumes
  patches were modified.
```
new:
```
- **Contract divergence.** `reconstruct` is bit-exact for unmodified
  patches under the count-map rule (exact iff every coverage count is a
  power of two — always true at `stride == patch_size`; THEORY §2);
  `stitch` is explicitly an interpolated blend that assumes
  patches were modified.
```

- [ ] **Step 4: `docs/THEORY.md` — três pontos**

:100, old:
```
and a geometry that puts a 3 or a 9 in the map does not, where the error is ~1 ULP (measured 2.4e-07 in float32 on a 16×16 image with `patch_size=4, stride=1`). Widening the dtype does not help, because the deciding axis is the geometry rather than the precision.
```
new:
```
and a geometry that puts a 3 or a 9 in the map does not. Outside the rule the per-pixel error is bounded by `(k+1)·eps·|v|`, with `k` the pixel's coverage count — the error grows with the overlap count (measured up to 19 ULP at k=81 in float32), so no fixed ULP figure applies. Widening the dtype does not help, because the deciding axis is the geometry rather than the precision.
```

:153, old:
```
`reconstruct` is bit-exact for unmodified patches under the count-map rule of §2 (exact when every count is a power of two, ~1 ULP otherwise) and rejects anything that would force interpolation;
```
new:
```
`reconstruct` is bit-exact for unmodified patches under the count-map rule of §2 (exact when every count is a power of two, per-pixel error ≤ `(k+1)·eps·|v|` otherwise) and rejects anything that would force interpolation;
```

:157, old:
```
inherits the same count-map rule (bit-exact when every count is a power of two, ~1 ULP otherwise; validated by bit-exact equality test on no-overlap and `allclose` on overlap).
```
new:
```
inherits the same count-map rule (bit-exact when every count is a power of two, per-pixel error ≤ `(k+1)·eps·|v|` otherwise; pinned by the two-half falsification suite in `tests/test_exactness.py`).
```

- [ ] **Step 5: `docs/GUIDE.md` — dois pontos**

:433, old (fim do parágrafo "Outside the predicate..."):
```
What does hold either way is the size of the miss, which was `2.384e-07` in float32 and `4.441e-16` in float64 on the geometries above, and that is one ULP territory.
```
new:
```
What does hold either way is the size of the miss: bounded per pixel by `(k+1)·eps·|v|`, with `k` the pixel's coverage count — `2.384e-07` in float32 and `4.441e-16` in float64 on the geometries above, and larger where the count map reaches higher, so there is no fixed ULP figure.
```

:441-443 ("Where this rule is written down"), old:
```
[ADR 0003](../../ADR/0003-reversibility-classes.md) is where the exactness boundary is being turned into contract, and it is still **Proposed**, so the wording has not landed across the project yet. Several docstrings and documents still state the overlap round trip as exact with no condition on the count map, and the audit that lists each of them is blocker B1 in [FOCO-1.0.md](../../FOCO-1.0.md).

Treat this section as the measured truth and treat those other statements as pending corrections, in that order, until ADR 0003 is accepted.
```
new:
```
[ADR 0003](../../ADR/0003-reversibility-classes.md) is where the exactness boundary is being turned into contract, and it is still **Proposed** (acceptance is part of the 1.0 freeze). The wording itself landed across the project in 0.5.0 — docstrings, SCOPE, THEORY, USAGE and the READMEs all state the count-map rule with the per-pixel bound — closing blocker B1 in [FOCO-1.0.md](../../FOCO-1.0.md).

Treat this section as the measured truth, and the ADR as the formal statement of the same rule.
```

- [ ] **Step 6: `docs/USAGE.md` — qualificação mínima** (o corpo segue 0.2.0 com banner de staleness; regeneração completa é G2)

:149, old: `### Overlap: weighted, still exact`
new: `### Overlap: weighted, exact under the count-map rule`

:158-163, old:
```
Each pixel covered by *k* patches; each contribution is the original
value; sum is `k * value`; division by the count map gives back
`value`. Bit-exact when every count in the map is a power of two —
`stride == patch_size` always, `stride == patch_size / 2` (counts 1,
2, 4, as above) — and within ~1 ULP otherwise, whatever the dtype:
the deciding axis is the geometry, not the precision.
```
new:
```
Each pixel covered by *k* patches; each contribution is the original
value; sum is `k * value`; division by the count map gives back
`value`. Bit-exact when every count in the map is a power of two —
`stride == patch_size` always, `stride == patch_size / 2` (counts 1,
2, 4, as above) — and otherwise bounded per pixel by `(k+1)·eps·|v|`,
whatever the dtype: the deciding axis is the geometry, not the
precision.
```

(:80-81, "Every spec here gives a bit-exact round-trip", fala só de tilings exatos — verdade, não tocar. :134, heading "## 5. reconstruct: bit-exact when stride == patch_size" — verdade como está, não tocar.)

- [ ] **Step 7: READMEs — cheat-sheet**

`README.md` :86, old:
```
| The patches back as an image, untouched | `reconstruct` | It is the exact inverse of `extract` |
```
new:
```
| The patches back as an image, untouched | `reconstruct` | Exact inverse of `extract` when every overlap count is a power of two — always true at `stride == patch_size` |
```

`README.pt-BR.md` :86, old:
```
| Os patches de volta como imagem, intocados | `reconstruct` | Ele é o inverso exato do `extract` |
```
new:
```
| Os patches de volta como imagem, intocados | `reconstruct` | Inverso exato do `extract` quando toda contagem de sobreposição é potência de dois — sempre vale com `stride == patch_size` |
```

- [ ] **Step 8: grep de verificação (FOCO §4)**

Run: `grep -rn "bit-exact\|bit a bit\|exato" src/ docs/ README.md README.pypi.md README.pt-BR.md`

Revisar hit a hit. Critério: em `src/`, `README*`, `docs/THEORY.md`, `docs/GUIDE.md`, `docs/SCOPE.md`, `docs/USAGE.md`, `docs/ADR/`, toda frase sobre exatidão do round-trip deve estar **condicionada** à regra do count map. Isentos por desenho: `docs/FOCO-1.0.md` e `docs/superpowers/` (documentos de auditoria/planejamento que citam a redação antiga como histórico), e usos não relacionados a round-trip (ex.: "termine exatamente na borda"). Se sobrar claim incondicionado fora dos isentos, corrigir no mesmo commit e listar no relatório da task.

- [ ] **Step 9: suite + commit**

Run: `uv run pytest -m "not gpu" -q`
Expected: verde (mudança é só texto/docstring; o docstring de `reconstruct` não tem doctest ligado).

```bash
git add src/patchcraft/reconstruct.py src/patchcraft/stitch.py docs/SCOPE.md docs/THEORY.md docs/GUIDE.md docs/USAGE.md README.md README.pt-BR.md
git commit -m "docs(g1): qualify the exactness predicate everywhere; per-pixel bound replaces '1 ULP' (B1)"
```

---

### Task 8: ramo sem zstandard — `tests/test_cache.py`

**Files:**
- Modify: `tests/test_cache.py`

**Interfaces:**
- Consumes: `patchcraft.cache._try_zstandard` (chamado no `Cache.__init__`, :108; ramo em `put`, :146-152; sidecar com `"compressed"`, :160).
- Produces: nada para outras tasks.

- [ ] **Step 1: teste novo**

`tests/test_cache.py` ganha `import json` no bloco de imports e a classe:

```python
class TestNoZstandardFallback:
    """Branch coverage: with zstandard absent the payload is stored raw and
    the sidecar says so (0.5.0; the compressed path is covered by the rest of
    the suite whenever the extra is installed)."""

    def test_put_get_roundtrip_uncompressed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("patchcraft.cache._try_zstandard", lambda: None)
        c = Cache(tmp_path, namespace="t")
        key = c.key_for("raw")
        c.put(key, b"uncompressed bytes")
        assert c.get(key) == b"uncompressed bytes"
        sidecar_path = next((tmp_path / "t").glob("*.json"))
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert sidecar["compressed"] is False
        assert sidecar["key"] == key
```

- [ ] **Step 2: rodar + commit**

Run: `uv run pytest tests/test_cache.py -v`
Expected: tudo verde (novo teste incluso, com ou sem zstandard instalado — o monkeypatch força o ramo).

```bash
git add tests/test_cache.py
git commit -m "test(g1): cover the no-zstandard cache branch (raw payload, compressed=False)"
```

---

### Task 9: release 0.5.0

**Files:**
- Modify: `src/patchcraft/__init__.py` (:20)
- Modify: `CHANGELOG.md` (após `[Unreleased]`)

**Interfaces:**
- Consumes: todas as tasks anteriores mergeadas nesta branch.
- Produces: 0.5.0.

- [ ] **Step 1: bump**

`src/patchcraft/__init__.py`: `__version__ = "0.4.0"` → `__version__ = "0.5.0"`.
(`pyproject.toml` não tem campo de versão — `[tool.hatch.version]` lê de `__init__.py`; ver Amendments, A3.)

- [ ] **Step 2: CHANGELOG**

Inserir logo após a linha `## [Unreleased]`:

```markdown
## [0.5.0] - 2026-09-01

### Changed

- `tilings`/`paired_tilings` no longer emit degenerate single-patch overlap
  specs: with one patch the stride is unobservable and the spec duplicated
  the exact tile. `tilings((28, 28), allow_overlap=True)` now returns 73
  specs instead of 100; `paired_tilings((14, 14), (28, 28), allow_overlap=True)`
  returns 27 instead of 40.

### Fixed

- **Public retraction: the documented exactness boundary was wrong.** The old
  wording (`k_max <= 4`, "`stride == patch_size / 2`", "within ~1 ULP" outside
  the rule) was measured false: the error grows with the pixel's coverage
  count (up to 19 ULP at count 81 in float32). The correct contract
  (ADR 0003): the `extract`/`reconstruct` round trip is bit-exact iff every
  value of the overlap count map is a power of two — always true at
  `stride == patch_size` — and outside the rule the per-pixel error is
  bounded by `(k+1)·eps·|v|`, with `k` the pixel's coverage count.
  Docstrings and docs (SCOPE, THEORY, GUIDE, USAGE, READMEs) now state this
  form; THEORY §9.1 records the extract-truncates / reconstruct-rejects
  asymmetry.
- Test suite: round-trip assertions now run on seeded full-mantissa noise
  (integer ramps and widened-float32 data could mask ULP-level errors); a
  falsification suite (`tests/test_exactness.py`) enumerates the 126,736
  legal geometries, samples 256 (seeded; full sweep via
  `PATCHCRAFT_SWEEP_FULL=1`), and pins both halves of the predicate; a naive
  loop-based reference (`tests/test_reference.py`) cross-checks the fast
  paths; the 20-name public surface is frozen by `tests/test_public_api.py`;
  the no-`zstandard` cache branch is now covered.

### Notes

- No signature changed. The public surface is exactly the 20 names of 0.4.0,
  now pinned by test.
```

- [ ] **Step 3: verificação completa**

Run, em ordem:
1. `uv run ruff check src tests` → limpo
2. `uv run mypy src` → limpo
3. `PATCHCRAFT_ACCEL=0 uv run pytest -m "not gpu" -q` → verde (modo puro)
4. `uv run pytest -m "not gpu" -q` → verde (modo accel, se o wheel estiver montado; senão idêntico ao passo 3)
5. `PATCHCRAFT_SWEEP_FULL=1 uv run pytest tests/test_exactness.py -q` → verde (~1-2 min; 0 divergências receita×count-map, 0 contraexemplos)
6. `cd accel && CARGO_TARGET_DIR=C:/Users/leona/.cache/patchcraft-target cargo test` → 6/6 (pular se a toolchain não existir; o crate não foi tocado)

Se qualquer passo falhar: **não** ajustar teste para passar — investigar e reportar.

- [ ] **Step 4: commit**

```bash
git add src/patchcraft/__init__.py CHANGELOG.md
git commit -m "release: 0.5.0 — count-map predicate retraction, falsification suite, tilings cleanup"
```

---

### Task 10: verificação de mutação + merge local

**Files:** nenhum (mutações são temporárias e revertidas).

- [ ] **Step 1: mutação 1 ULP (FOCO §4, item 1)**

Em `src/patchcraft/reconstruct.py`, logo após o bloco `if numerator is None:` (antes da divisão pelo count map), inserir:

```python
    numerator = torch.nextafter(numerator, torch.full_like(numerator, float("inf")))
```

Run: `uv run pytest tests/test_exactness.py tests/test_reference.py tests/test_reconstruct.py -x -q`
Expected: **FAIL** (a metade positiva da suíte de exatidão deve ficar vermelha).
Reverter: `git checkout -- src/patchcraft/reconstruct.py` e re-rodar o mesmo comando → verde.
Se a mutação passar despercebida, a suíte não presta: parar e reportar, não mergear.

- [ ] **Step 2: mutação `__all__` (FOCO §4, item 2)**

Remover `"resize",` de `__all__` em `src/patchcraft/__init__.py`.
Run: `uv run pytest tests/test_public_api.py -q` → **FAIL**.
Reverter: `git checkout -- src/patchcraft/__init__.py` e re-rodar → verde.

- [ ] **Step 3: merge local**

```bash
git checkout main
git merge --no-ff feat/0.5.0-g1-predicado -m "merge: Fase 3 G1 — predicado bit-exato, suíte falsificadora, superfície congelada (0.5.0)"
git log --oneline -5
```

Sem push, sem tag. Reportar o estado final: commits do merge, saída dos dois modos da suite, resultado do sweep completo.

---

## Amendments (desvios da spec, registrados)

- **A1** — `within_pixel_bound(out, img, counts)` recebe o count map pré-computado (via o 5º helper `coverage_counts`), em vez de recomputar a geometria a cada chamada. Motivo: a metade negativa reusa o mapa em 50 seeds por geometria. A spec §2.1 congela só os outros quatro helpers; `coverage_counts`/`within_pixel_bound` não estavam nomeados lá.
- **A2** — `tests/test_pair.py` não muda. `pair` faz zero aritmética de pixel (`extract` é gather puro); a rampa lá testa ordem/posição, que é exatamente onde a spec §2.2 manda a rampa ficar.
- **A3** — `pyproject.toml` não tem campo `version` (dinâmico via `[tool.hatch.version]` → `src/patchcraft/__init__.py`); o "pyproject idem" da spec §2.9 é no-op.
- **A4** — `src/patchcraft/reconstruct.py:40-42` ("the error is about 1 ULP") entra na lista de B1. A auditoria da spec §1 chamou o predicado de correto ali, mas a frase de ULP viola o Amendment A ("nunca '1 ULP' sem qualificação").
- **A5** — `test_float32_overlap_close` é renomeado para `test_float32_overlap_within_pixel_bound` e a geometria muda de s=2 para s=1: s=p/2 é **dentro** do predicado (bit-exato), então o caso antigo não exercitava o regime de erro que o nome sugeria.
- **A6** — `docs/GUIDE.md:441-443` ("Where this rule is written down") é reescrito: com B1 fechado, "the wording has not landed" viraria falso.
- **A7** — a edição de `docs/USAGE.md:84-91` (contagem 100→73 + nota de patch único + frase condicionada) fica na Task 6, porque a contagem é efeito direto da guarda D1; o restante de USAGE fica na Task 7.
