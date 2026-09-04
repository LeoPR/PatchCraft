# STATUS: 2026-09-04

> The snapshot. Where things are *right now*, and who each open item waits on.
> Where things **are** is [MAP.md](MAP.md); the rules are [AGENTS.md](AGENTS.md);
> the history is [CHANGELOG.md](CHANGELOG.md). This file does not repeat them.

## Where the project is

**0.5.4 is on PyPI**, published through Trusted Publishing on a tag push: one
sdist, five `cp312-abi3` platform wheels carrying the Rust accelerator, and one
universal `py3-none-any` wheel. There is no extra to enable and no second
package.

**1619 tests pass** under `pytest -m "not gpu"`, with 32 skipped and 5
deselected, plus two full-sweep gates behind `PATCHCRAFT_SWEEP_FULL=1`
that enumerate all 126,736 legal geometries. CI is green on
{Ubuntu, Windows} x {3.12, 3.13, 3.14} on the pure path, plus a two-OS job that
builds the Rust kernel and runs the whole suite through it.

**The surface is 20 names**, pinned by `tests/test_public_api.py` with
`inspect.signature`.

## What is in flight

**Nothing unreleased.** 0.5.4 closed the nine entries that had accumulated
since 0.5.3: two fixes, `Cache` not expanding a leading `~` and `resize`
raising a raw torch error on 13 of the 25 resample-mode x integer-dtype
combinations, the second of which also uncovered a silent `int32` truncation
older than itself. The other seven are documentation, and they close every
item the 2026-09-04 audit left open, including ADR 0004's step 0 and the
`WeightKind` compatibility sentence.

## What is blocked, and on whom

**ADR 0003 is accepted**, rescoped first. It ratifies one rule, that an
exactness claim is declared per regime with a condition the caller can evaluate
before calling, and it carries the accumulator amendment in its body: the
predicate is a property of the geometry **and** the accumulator, since over 76
legal geometries a `float32` accumulator is exact on 45, precisely the
power-of-two ones, and a `float64` accumulator on all 76. The three-letter
vocabulary its first draft proposed is deferred until a transform needs it.

**ADR 0004 is `Proposed`, and asks nothing.** It records the precision and
effort parameters as a design that is not built; the status reflects that none
of it is implemented, not that a decision is pending. The one question it
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
created a directory literally named `~`. Fixed in the unreleased set, with no
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
