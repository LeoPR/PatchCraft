# PatchCraft — auxiliary tooling (NOT part of the wheel)

The lib is the car. This document describes the **track**: the
auxiliary tooling in this repo that exists to test, exercise, and
visualize the core, but is deliberately kept out of `pip install
patchcraft`. Nothing here is public API; nothing here is stable;
nothing here ships.

Two boundaries are load-bearing:

1. **Files** in `src/patchcraft/` get shipped. Files anywhere else
   (`tests/`, `lab/`, `examples/` if it ever exists) do not.
2. **Docs** in this file describe the auxiliary side. Docs in
   [`THEORY.md`](THEORY.md), [`USAGE.md`](USAGE.md),
   [`SCOPE.md`](SCOPE.md), and [`ADR/`](ADR/) describe the lib. The
   two sides never mix — when an auxiliary helper does something
   that looks core-shaped, it stays here.

---

## 1. `tests/_datasets.py` — dev fixtures, leading underscore on purpose

PatchCraft core takes one image at a time. The test suite needs *real*
images to drive the primitives through varied conditions, but the
lib must not ship a dataset loader. `tests/_datasets.py` resolves
that tension:

- Lives under `tests/`, so it never ends up in the wheel.
- The filename has a leading underscore — Python convention for "not
  part of any public API of the enclosing package".
- Imports `torchvision` lazily, inside the function. `torchvision` is
  in the `[dev]` extra ([`pyproject.toml`](../pyproject.toml)), never
  a runtime dep of the lib.

### What it exposes

```python
from tests._datasets import label_subset, mnist_subset
```

- **`label_subset(labels, n_per_label, seed=0) -> list[int]`** —
  pure function over a labels sequence. Picks up to `n_per_label`
  indices per distinct label, deterministically seeded.
  *Not* part of the published API; if a downstream consumer needs
  label stratification, they implement it (it is ~15 lines).
  Originally proposed as `patchcraft.label_subset(dataset, ...)`; moved
  here when the binding scope ruled out dataset-shaped APIs in core
  (see [`SCOPE.md`](SCOPE.md) §1, "Label-stratified subset selection"
  row, and [`THEORY.md`](THEORY.md) §7).
- **`mnist_subset(n_per_label=5, seed=0, train=True)
  -> list[tuple[Tensor[1,28,28], int]]`** — downloads MNIST on first
  call into `Z:\caches\datasets\mnist\`; subsequent calls hit the
  cache. Returns balanced `(image, label)` pairs as `float32` in
  `[0, 1]`, ready to feed straight into `patchcraft.extract`.

### Why it stays auxiliary

- A "dataset loader" is multi-image orchestration. PatchCraft core
  refuses to grow that.
- Different consumers want different splits, different transforms,
  different formats. Forcing one in the wheel pleases none of them.
- The test suite needs exactly one cheap fixture: balanced MNIST.
  That fixture is too narrow to be a feature and too useful to throw
  away — `tests/_datasets.py` is its home.

---

## 2. `lab/` — ephemeral bench

```
lab/
├── README.md       (tracked — the rules)
└── .gitignore      (tracked — ignores everything else)
```

Everything inside `lab/` other than `README.md` and `.gitignore` is
local to your working copy. The rules:

1. **Throwaway.** Scripts here are not reviewed, not tested, not
   stable. If something deserves to survive, it migrates: turned
   into a test under `tests/`, or (later) promoted to
   `experiments/<NNN>-slug/`.
2. **Nothing in `lab/` is imported by `src/patchcraft/`.** One-way
   dependency: lab uses the lib, the lib never uses lab.
3. **Outputs go off-tree.** Lab scripts write to
   `Z:\outputs\patchcraft\<YYYY-MM-DD-slug>\`, *never* into the project
   root. The script creates the directory; nothing in this repo
   tracks the artefacts.
4. **Suggested naming:** `YYYY-MM-DD-slug.{py,ipynb}`. Prefix-sorts
   chronologically.
5. **Lifecycle:** if a script sits in `lab/` for two-plus weeks
   without being promoted or deleted, it is a candidate for
   removal. Stale exploration adds noise.

### Example contents (none are committed)

The current local working copy has, at the time of writing:

- `lab/2026-05-16-roundtrip-mnist.py` — the script that validated
  M2 + M3 end-to-end on MNIST (4 geometries, max diff `1.19e-7`
  on `stride=1 float32`). Produced
  `Z:\outputs\patchcraft\2026-05-16-roundtrip-mnist\sample0-digit0-half-overlap.png`.
- `lab/usage_demo.py` + `lab/usage_demo.out` — the script and
  captured output behind [`USAGE.md`](USAGE.md). Re-run if any
  public API signature changes.

Neither is in the repo. If you clone fresh, `lab/` will have only
this README and the `.gitignore`.

---

## 3. Off-tree conventions on `Z:\`

The dev environment uses [`Z:\caches`](../../../caches) and
[`Z:\venvs`](../../../venvs) for everything that should not live
inside a OneDrive-synced project root. Two sub-conventions are
specific to PatchCraft:

| Path | Contents | Created by | Cleanable |
|---|---|---|---|
| `Z:\caches\datasets\<name>\` | Downloaded reference datasets (MNIST so far; CIFAR-10 / DIV2K when needed). | `tests/_datasets.py` on first call | Yes — will re-download |
| `Z:\outputs\patchcraft\<slug>\` | Artefacts from `lab/` scripts: PNGs, JSONs, CSVs, dumps. | Each lab script creates its own subdir. | Yes — these are derived |

Neither path is created by `pip install patchcraft`. Both are created
on demand by the auxiliary tooling. If you move machines and your
new dev environment doesn't have `Z:\`, edit
[`tests/_datasets.py::DATASETS_ROOT`](../tests/_datasets.py) and the
hardcoded paths inside your lab scripts. There is no PatchCraft setting
for these — they belong to the bench, not the lib.

The canonical reference for `Z:\` is the project-external
`dev-environment/README.md` outside this repo. PatchCraft knows nothing
about it; that doc is operator-facing.

---

## 4. What the wheel ships vs. what stays local

| In `dist/patchcraft-0.1.0-py3-none-any.whl` | NOT in the wheel |
|---|---|
| `src/patchcraft/*.py` | `tests/` (including `_datasets.py`) |
| `src/patchcraft/py.typed` | `lab/` |
| `LICENSE` (via wheel `licenses/`) | `docs/` (including this file) |
| `METADATA` (from `pyproject.toml`) | `pyproject.toml` (only metadata is extracted) |
| | `archive/` (gitignored anyway) |
| | `uv.lock`, `.vscode/`, `.python-version` |

If you want to confirm the boundary, after a build:

```
python -m zipfile -l dist/patchcraft-0.1.0-py3-none-any.whl
```

You should see only `patchcraft/` (the seven primitives + `py.typed`)
and the `*.dist-info/` metadata. Nothing else.

---

## 5. Pointers

- [`SCOPE.md`](SCOPE.md) — formal responsibilities table, including
  what counts as auxiliary in row-by-row form.
- [`USAGE.md`](USAGE.md) — the lib's API exercised with real outputs.
  Uses `mnist_subset` only via the auxiliary path, never inside an
  example presented as "the library does this".
- [`lab/README.md`](../lab/README.md) — the bench rules from inside
  the bench.
