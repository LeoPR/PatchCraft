# STATUS: 2026-10-11

> The snapshot. Where things are *right now*, and who each open item waits on.
> Where things **are** is [MAP.md](MAP.md); the rules are [AGENTS.md](AGENTS.md);
> the history is [CHANGELOG.md](CHANGELOG.md). This file does not repeat them.

## Where the project is

**0.6.0 is on PyPI**, published through Trusted Publishing on a tag push: one
sdist, five `cp312-abi3` platform wheels carrying the Rust accelerator, and one
universal `py3-none-any` wheel. There is no extra to enable and no second
package.

0.6.0 added `resize(..., antialias=None)`, keyword-only with the default that
changes no call, and made `stitch` with the uniform window hand an overlapping
grid to `reconstruct`, bit-identical and 1.45 to 1.75 times faster on the pure
path at 512 and 1024. The new parameter is why it is a new `0.y`, by the
README's rule and the first row of CONTRIBUTING's table.

0.5.7 brought integer images: `extract` takes every
dtype, and `reconstruct` takes the five integer dtypes and returns them exactly
in every legal geometry, with disagreeing patches rounded half to even and no
intermediate overflowing. They are ADR 0004's steps 1 and 2, shipped together.
The full suite and the 126,736-geometry sweep, which now includes an integer
round trip, pass on both paths.

0.5.6, the release before it, was a fix release: `torch.vmap` over `extract`,
`reconstruct` and `stitch`, which the documentation offers as a way to batch,
failed on most calls since 0.5.0 and works again, pinned by
`tests/test_vmap.py`; and `tilings` returns `[]` for an image smaller than the
default minimum patch instead of raising. With them came the corrected reason
for the exactness rule, THEORY §2.6 on the operator, the USAGE walk through a
4×4 image, and a CI job at the declared floor, whose first run found a claim
that held only at the top. The full 126,736-geometry sweep passes on both
paths.

**2025 tests pass** under `pytest -m "not gpu"`, with 32 skipped and 5
deselected, plus two full-sweep gates behind `PATCHCRAFT_SWEEP_FULL=1`
that enumerate all 126,736 legal geometries. CI is green on
{Ubuntu, Windows} x {3.12, 3.13, 3.14} on the pure path, plus a two-OS job that
builds the Rust kernel and runs the whole suite through it, all at the newest
torch, and a job at the declared floor (Python 3.12, torch 2.6, numpy 1.26,
pillow 10.1, Rust 1.82) that runs it on both paths. Three of those jobs also
run on `ubuntu-26.04` by name, for as long as the item below holds.

**The surface is 20 names**, pinned by `tests/test_public_api.py` with
`inspect.signature`.

## What is in flight

**On `main` since 0.6.0, unreleased:** nine images in the GUIDE, drawn by
`tools/make_guide_figures.py` and checked by `tests/test_docs_figures.py`, and
the CI arrangement for the Ubuntu move below, and the GitHub Actions pinned to
exact versions, with Dependabot proposing the updates. Nothing that ships in a
wheel has changed; the next sdist carries the images.

**The Ubuntu move is in flight until 2026-11-19.** GitHub moves
`ubuntu-latest` from Ubuntu 24.04 to 26.04 between 2026-10-19 and 2026-11-19,
and a job on that label can land on either image while it moves. `test.yml`
runs a pure cell, the accelerated job and the floor job on `ubuntu-26.04` by
name meanwhile, and `release.yml` builds on `ubuntu-24.04`. Once the move is
over, the 26.04 entries repeat `ubuntu-latest` and go. Moving `release.yml`
is a change of its own, which first shows the manylinux build, the part that
runs in Docker on the host, green on `ubuntu-26.04` without publishing.
**Waits on the date.**

**ADR 0005 is `Proposed`**: padding a grid that does not cover the image, with
the design taken from kornia's auto-padding and MONAI's `padding_mode`, one
keyword with a default that changes no call. Nothing is implemented. The owner
recorded it in the ROADMAP as the 1.1 candidate rather than scheduling it, so it
**waits on a consumer**.

**`docs/PAPER/` is a scaffold** for an article on means that never leave
their domain: outline, formal statements, measurements and references, with
the prose not yet written. It does not ship. Its first open item, a search for
prior art on the integer carry, **waits on the owner**, as does the writing.

## What is blocked, and on whom

**ADR 0003 is accepted**, rescoped first. It ratifies one rule, that an
exactness claim is declared per regime with a condition the caller can evaluate
before calling, and it carries the accumulator amendment in its body: the
predicate is a property of the geometry **and** the accumulator, since over 76
legal geometries a `float32` accumulator is exact on 45, precisely the
power-of-two ones, and a `float64` accumulator on all 76. The three-letter
vocabulary its first draft proposed is deferred until a transform needs it.

**ADR 0004 is `Proposed`, and asks nothing.** It records the precision and
effort parameters as a design that is mostly not built; the status reflects
that `accum_dtype`, `rounding_mode` and the preset are not implemented, not
that a decision is pending. What it records that did ship: its steps 1 and 2,
integers through `extract` and `reconstruct` without a knob, in 0.5.7, and its
third parameter, `resize(..., antialias=None)`, in 0.6.0. The one question it
appeared to hold, whether to add the knobs before the 1.0 freeze or after,
rested on the premise that adding one afterwards is expensive. It is not: each
is a keyword-only argument with a default, so adding one is additive and is a
minor release, and `reconstruct` has no positional-only parameters to make the
order matter. Step A of its plan shipped in 0.5.2. **Waits on nobody.**

**The `Cache` path question is closed, and it was not a decision.** It was
posed as whether to reject a `namespace` containing `..`. The answer is no:
inventing path security is not this library's job, and no library in the
ecosystem does it, verified against `torch.hub.set_dir`, pytest's cacheprovider
and pip. Asking the question found the real defect, which was the opposite
shape: `Cache` was missing the one thing all of them do, so `Cache("~/cache")`
created a directory literally named `~`. Fixed in 0.5.4, with no
validation added. [SECURITY.md](SECURITY.md) and THEORY §9.5 now say the path
is the caller's and that only `~` is expanded.

## 1.0 blockers

**All six are closed.** The last, B4, closed in 0.5.3 by making `docs/USAGE.md`
executable rather than by regenerating it: every `>>>` on the page now runs in
the suite, so it cannot go stale silently again.

**And that leaves no decision before 1.0 either.** ADR 0003 is accepted and
ADR 0004 asks nothing, so what stands between here and 1.0 is the project's own
gate rather than an open question: an external consumer, which it does not have
yet.

| | |
|---|---|
| B1 predicate wrong in fifteen places | closed in 0.5.0 |
| B2 the suite could not falsify B1 | closed in 0.5.0 |
| B3 nothing pinned the public surface | closed in 0.5.0 |
| B4 the two pages a stranger reads | closed in 0.5.3 |
| B5 THEORY §9 contradicted itself on fp16 | closed in 0.5.2 |
| B6 enumeration junk and asymmetric guards | closed in 0.5.0 |

## What the project still declines to claim

No external project has consumed the library, and that consumption is this
project's own stated gate for calling the shape settled. No CUDA path has ever
executed, in CI or outside it. Three of the five accelerated wheels, the two
macOS ones and the aarch64 one, are built and have their contents checked, and
their kernel has never run in CI.

## Also outstanding, outside the 1.0 path

**A DOI.** The Strata conformity review
([docs/STUDIES/2026-09-03-strata-conformity.md](docs/STUDIES/2026-09-03-strata-conformity.md))
found it to be the one item missing from the publishing pattern, and it is the
kind that cannot be applied retroactively: Zenodo mints a DOI per release from
the moment it is connected, and every release before that has none, permanently.
One authorisation. **Waits on the owner.**
