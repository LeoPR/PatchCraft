# STATUS: 2026-09-03

> The snapshot. Where things are *right now*, and who each open item waits on.
> Where things **are** is [MAP.md](MAP.md); the rules are [AGENTS.md](AGENTS.md);
> the history is [CHANGELOG.md](CHANGELOG.md). This file does not repeat them.

## Where the project is

**0.5.2 is on PyPI**, published through Trusted Publishing on a tag push: one
sdist, five `cp312-abi3` platform wheels carrying the Rust accelerator, and one
universal `py3-none-any` wheel. There is no extra to enable and no second
package.

**1557 tests pass**, plus two full-sweep gates behind `PATCHCRAFT_SWEEP_FULL=1`
that enumerate all 126,736 legal geometries. CI is green on
{Ubuntu, Windows} x {3.12, 3.13, 3.14} on the pure path, plus a two-OS job that
builds the Rust kernel and runs the whole suite through it.

**The surface is 20 names**, pinned by `tests/test_public_api.py` with
`inspect.signature`.

## What is in flight

**An unreleased set of nine entries.** All of it documentation, tooling and
packaging: the `bfloat16` correction that closed B5, `CITATION.cff`,
`.pre-commit-config.yaml`, `MAP.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, the
issue forms, `AGENTS.md` and this file. Under the versioning rule none of it
moves a contract, so it is a `0.5.3`.

## What is blocked, and on whom

**ADR 0003 is accepted**, rescoped first. It ratifies one rule, that an
exactness claim is declared per regime with a condition the caller can evaluate
before calling, and it carries the accumulator amendment in its body: the
predicate is a property of the geometry **and** the accumulator, since over 76
legal geometries a `float32` accumulator is exact on 45, precisely the
power-of-two ones, and a `float64` accumulator on all 76. The three-letter
vocabulary its first draft proposed is deferred until a transform needs it.

**ADR 0004 is `Proposed` by design.** It records the precision and effort
parameters and was written to be decided later. Step A of its plan shipped in
0.5.2. **Waits on the owner, deliberately.**

**One decision surfaced by the security audit.** `Cache` validates its
`namespace` only as a non-empty string and joins it into a path, so
`Cache(root, namespace="../elsewhere")` writes outside `root`. THEORY §9.5
never constrained the argument's shape, so tightening it changes behaviour on a
public constructor rather than fixing a violation. Documented as a trust
boundary in [SECURITY.md](SECURITY.md). **Waits on the owner.**

## 1.0 blockers

**All six are closed.** The last, B4, closed in 0.5.3 by making `docs/USAGE.md`
executable rather than by regenerating it: every `>>>` on the page now runs in
the suite, so it cannot go stale silently again.

What that leaves before 1.0 is not a blocker but a decision, and it is ADR 0003,
above.

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
