# PatchCraft: the canonical guide

> **Single source, agent-agnostic.** The branded files ([`CLAUDE.md`](CLAUDE.md))
> point here and do not duplicate content. On any disagreement, this file wins.
>
> **Route**: this guide = *the rules* · [`MAP.md`](MAP.md) = *where things are* ·
> [`CONTRIBUTING.md`](CONTRIBUTING.md) = *how to run the gates and release* ·
> [`STATUS.md`](STATUS.md) = *where things stand now* ·
> [`docs/FOCO-1.0.md`](docs/FOCO-1.0.md) = *what is left before 1.0*.
> Do not repeat between them. Link.

## 0. The invariants

Each of these was earned. Where one names an incident, the incident is real and
is in the changelog.

### I1. Measure before you write it down

Every number in this repository has a command that reproduces it. A sentence
that says "faster", "exact" or "negligible" without one is a defect waiting to
be found, and several have been.

The failures this rule comes from: a predicate published as `k_max <= 4`,
measured false and retracted in public; a benchmark that reported the
byte-native path as nine times slower because `torch`'s `uint8.sum()` silently
promotes to `int64`, so it measured the opposite of what it claimed; a security
note asserting `Cache` runs `torch.load`, which it does not.

Recalling is not measuring. Reading the code is not measuring either, when the
question is numeric.

### I2. `docs/THEORY.md` §9 is the arbiter

It says so itself. When two documents disagree about what a function accepts,
rejects or returns, §9 is the truth and the other document is the bug. When §9
and the code disagree, decide which is wrong before changing either, and say
which in the commit.

§9 has been wrong twice. That is not a reason to trust it less; it is why the
rule exists.

### I3. A fix ships with a test that fails on the old code

Without one it is not a fix, it is a coincidence. The suite already contains
several tests whose only job is to prevent a specific defect from returning,
and each names the measurement in its docstring.

### I4. The exactness suite exists to falsify, not to confirm

`tests/test_exactness.py` enumerates the legal geometry space independently of
the predicate and hunts for both counterexamples: a case inside the rule that
is not exact, and a case outside it that is exact by luck. If you change the
predicate, the enumerator must not learn about the change.

### I5. The public surface is 20 names, frozen by test

`tests/test_public_api.py` pins the names, their signatures and the fields of
the four data carriers. Adding a name is a decision recorded in an ADR, never a
convenience taken mid-task. [`docs/FOCO-1.0.md`](docs/FOCO-1.0.md) says what
1.0 freezes.

### I6. Both code paths stay green

The library has a pure-torch path and, on five of six wheels, a Rust one. They
are bit-identical by test. Any change to the fold, the count map or the
accelerator must run both: `pytest -m "not gpu"` with the extension built, and
again with `PATCHCRAFT_ACCEL=0`.

### I7. The version answers one question

Not "did a returned value change", because a fix changes values by definition.
The question is whether the **contract moved** (`0.Y.0`) or the
**implementation stopped violating it** (`0.y.Z`). The rule and its four worked
examples are in [`CONTRIBUTING.md`](CONTRIBUTING.md), and getting it wrong is
what nearly shipped 0.5.2 as 0.6.0.

### I8. Nothing ephemeral lives in the project folder

Caches, build output and virtual environments are redirected by user-level
environment variables. `accel/target` reached 269 MiB inside the tree before
this was enforced. Never commit a machine path to fix it: that mistake is
already in the history, as a `cache_dir = "Z:\caches\pytest"` that created a
literal directory of that name on Linux.

### I9. A tracked file ships to PyPI unless two files say otherwise

`setuptools-scm`'s file finder offers every git-tracked file to the sdist, so
`MANIFEST.in` is a blocklist over everything git knows about, and
`tools/check_dist.py` holds the matching allowlist. Adding a top-level
directory means updating both. The gate will tell you; do not silence it.

### I10. An ADR carries the choice, not the research behind it

Before writing one, split what is being said into what is **determined** and
what is **chosen**, and put only the second in the ADR.

A name is determined: which identifier a parameter carries is a fact about the
ecosystem the caller already imports, and picking one by taste is a guess made
permanent by a frozen surface. A measurement is determined. A benchmark table is
determined. Those belong in `docs/STUDIES/`, dated, with the ADR linking to
them.

What is chosen is whether to do the thing at all, when, and what it costs the
surface. If that part comes out empty, it was never an ADR: it was a finding,
and a study is where it goes.

Both ADRs written in 2026-09 failed this on the first draft, in the same way
and for the same reason, which is that writing down everything you learned feels
like thoroughness and reads as a lab notebook. ADR 0003 mixed a rule with a
vocabulary for six transforms that never shipped; ADR 0004 mixed the names,
which precedent decides, with whether to spend a frozen surface on them, which
only the owner decides. Both shrank by a third once separated, and both were
caught by the owner asking what the document was for.

### I11. Do not invent what already exists

Check for an official or established solution before writing one. The
accelerator's packaging, the versioning rule, the citation format and the
outreach layout were all taken from prior art rather than designed here, and
each is better for it. Writing something new is fine when nothing fits; writing
it without looking is not.

## 1. Writing

Prose in this repository has no em-dashes. Replace the semantic role with a
sentence that carries it, never with another symbol. Keep articles and
connectives; one idea per paragraph. `README.md` is a call page, not a manual:
the manual is [`docs/GUIDE.md`](docs/GUIDE.md), and complex examples belong
there.

Documents that state a version go stale. There is exactly one version string
left in the documentation, the BibTeX entry in GUIDE section 9, plus
[`CITATION.cff`](CITATION.cff). Do not add a third.

## 2. Before you touch anything

```
uv sync --locked --extra cache --extra dev
ruff check src tests tools
mypy --strict src
pytest -m "not gpu"
```

For a change to the fold, the predicate or the accelerator, add:

```
PATCHCRAFT_SWEEP_FULL=1 pytest tests/test_exactness.py
cargo test --manifest-path accel/Cargo.toml
```

The same gates run as a pre-commit hook. On a machine with a global
`core.hooksPath`, that hook is installed but inert;
[`CONTRIBUTING.md`](CONTRIBUTING.md) says how to tell.

## 3. What this project is not

No batch axis, no dataset, no dataloader, no model, no padding to make an
awkward geometry fit. Those are out by decision and the line is drawn in
[`docs/SCOPE.md`](docs/SCOPE.md) and [`docs/THEORY.md`](docs/THEORY.md) §0.
A request to add one of them is a scope conversation, not a feature.
