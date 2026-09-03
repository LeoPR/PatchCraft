# MAP: finding your way around PatchCraft

> One page. If you know **what** you want, this says **where** it is.
> If you do not know yet, start at [README.md](README.md), which is the call page.

PatchCraft cuts one image into patches and puts it back. The whole library is
20 public names over eight modules; almost everything in this repository is the
evidence that those 20 behave as documented.

## The tree

```
PatchCraft/
├── README.md ................. the call page, canonical, English
│   README.pt-BR.md ........... the same page in Portuguese
│   README.pypi.md ............ the PyPI page; links absolute, because relative ones break there
├── MAP.md .................... this file
├── CHANGELOG.md .............. every release, each change with the measurement behind it
├── CONTRIBUTING.md ........... layout, the two code paths, the versioning rule, releasing
├── CITATION.cff .............. citation metadata; GitHub reads it for the cite button
├── LICENSE ................... MIT
│
├── src/patchcraft/ ........... THE LIBRARY. 20 public names, frozen by test
│   ├── __init__.py ........... re-exports the surface and the generated version
│   ├── extract.py ............ image -> patches; Patchify wraps it for a Compose
│   ├── reconstruct.py ........ patches -> image, the exact inverse for untouched patches
│   ├── stitch.py ............. patches -> image with a window, for patches a model rewrote
│   ├── geometry.py ........... num_patches, tilings, scale_factor, paired_tilings; no pixels
│   ├── pair.py ............... LR/HR pairing; PatchPair, PatchMeta
│   ├── metrics.py ............ patch_metrics, per_patch_mse, per_patch_psnr
│   ├── resize.py ............. PIL and torch backends
│   ├── cache.py .............. content-addressed disk cache
│   ├── _foldgeom.py .......... the coverage guard both reconstruct and stitch call
│   └── _accel.py ............. bridge to the native kernel; falls back silently
│
├── accel/ .................... the Rust kernel, compiled INTO the wheel, not a second package
│   └── src/kernel.rs ......... the gather-fold; cargo-tested without Python
│
├── tests/ .................... the contract, executable
│   ├── test_exactness.py ..... enumerates 126,736 geometries and tries to FALSIFY the predicate
│   ├── test_reference.py ..... a naive loop the fast paths must agree with
│   ├── test_public_api.py .... freezes the 20 names, their signatures and carrier fields
│   └── _rng.py ............... audited data generation; integer ramps hid a real defect once
│
├── docs/
│   ├── GUIDE.md .............. THE MANUAL. every README claim, measured, with its output
│   ├── THEORY.md ............. the math, and §9 is the arbiter of the per-function contract
│   ├── SCOPE.md .............. the line between this library and your pipeline
│   ├── PERFORMANCE.md ........ what the accelerator is worth; machine, versions, date, command
│   ├── USAGE.md .............. a walkthrough, captured against an old release and saying so
│   ├── AUXILIARY.md .......... test fixtures, lab/, the off-tree conventions
│   ├── ROADMAP.md ............ milestones, historical
│   ├── FOCO-1.0.md ........... what 1.0 freezes, and the blockers still in the way
│   ├── ADR/ .................. one file per decision that shaped the API
│   ├── design/ ............... one spec per work phase: alternatives measured, then chosen
│   └── STUDIES/ .............. background reading behind those decisions
│
├── tools/
│   ├── benchmark.py .......... accelerated vs pure, and proves they agree before timing
│   └── check_dist.py ......... release gate: extension present, versions agree, tag respected
│
├── outreach/ ................. material for presenting the project; not shipped
├── lab/ ...................... scratch experiments; only the README is tracked
└── .github/workflows/ ........ test.yml (both code paths) and release.yml (one project, six wheels)
```

## I want to... go here

| I want to | Go to |
|---|---|
| **See what this is, in two minutes** | [README.md](README.md) |
| **Use it**: every argument, measured, with its printed output | [docs/GUIDE.md](docs/GUIDE.md) |
| Know **when the round trip is exact**, as a rule I can evaluate myself | [docs/GUIDE.md §4](docs/GUIDE.md#4-when-the-round-trip-is-bit-for-bit), and [docs/ADR/0003](docs/ADR/0003-reversibility-classes.md) for the contract |
| Know **what each function accepts and rejects**, precisely | [docs/THEORY.md §9](docs/THEORY.md), which is the arbiter where documents disagree |
| Know **whether this library is the right tool** | [docs/SCOPE.md](docs/SCOPE.md) |
| Know **what the native accelerator is worth**, and re-measure it | [docs/PERFORMANCE.md](docs/PERFORMANCE.md), then `python tools/benchmark.py` |
| Know **what the project refuses to claim** | [docs/GUIDE.md §8](docs/GUIDE.md#8-what-this-project-does-not-claim) |
| Understand **why the API looks like this** | [docs/ADR/](docs/ADR/) |
| Understand **why a phase was built the way it was** | [docs/design/](docs/design/) |
| **Contribute**: run the gates, understand the two code paths | [CONTRIBUTING.md](CONTRIBUTING.md) |
| **Release** it | [CONTRIBUTING.md](CONTRIBUTING.md), the "Releasing" section |
| Know **what changed and why** | [CHANGELOG.md](CHANGELOG.md) |
| Know **what is left before 1.0** | [docs/FOCO-1.0.md](docs/FOCO-1.0.md) |
| **Cite** it | [CITATION.cff](CITATION.cff) |

## Two things worth knowing before you read anything else

**The contract is falsifiable, on purpose.** `tests/test_exactness.py` exists to
break the library's own exactness claim, not to confirm it. That claim was
published in a weaker form, measured false and retracted, and the suite is what
came out of it. When a document and a test disagree, the test is right.

**Where documents disagree with each other, [THEORY.md](docs/THEORY.md) §9
wins.** It says so itself, and it is the section to fix first when something is
found wrong.
