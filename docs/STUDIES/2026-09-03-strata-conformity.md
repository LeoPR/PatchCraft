# Strata conformity review of PatchCraft

Date: 2026-09-03. Method: Strata **v1.2.2**, canonical English source at
`Acadêmicos/Methodologies/recipe/knowledge-architecture.en.md`, last changed by
commit `3e92195`.

## How this review was bounded

Strata §9 constrains the review before it constrains the project, in two ways
that matter more than any individual finding.

**Name the genre and apply its standard.** §9 says what counts as
well-organised depends on the kind of artifact, and it names the case directly:
*a reference work calls for an index, cross-references, errata; in software, a
library: tests, packaging, CI*. PatchCraft is a library, so that is the
standard applied here. Demanding a research project's apparatus of it would be,
in the method's own words, "the same §9 excess along another axis", and low
conformance to a foreign standard is not a defect.

**The default verdict is no change.** §9: *the default verdict of an honest
evaluation is no change, unless a real defect pays for the fix. Inventing work
where nothing is broken is §9 excess on the action axis. Name the defect first;
if there is none, the deliverable is the statement that there is none.*

That is this review's shape. It found one real gap and one piece of debris.

## What was checked, and what it found

| Section | What it asks | Here |
|---|---|---|
| §1 | the kinds of artifact physically separated | `src/`, `tests/`, `docs/`, `tools/`, `accel/`, `outreach/`, `lab/` |
| §3 | traceability | four ADRs, and a changelog whose entries carry the measurement behind them |
| §3-bis | the force of the artifact declared | ADR status says which act it is: two `Accepted`, two `Proposed` |
| §5 | single source by altitude | `THEORY.md` §9 is declared the arbiter, in the document itself and in `AGENTS.md` |
| §6 | source discipline | every number has a command that reproduces it, written as invariant I1 |
| §8 | immutable history, signal vs noise | git, `.gitignore`, Keep a Changelog, SemVer with a written rule and an errata for the two releases that predate it |
| §10 | durability by redundancy | three independent carriers: git history, GitHub, PyPI |
| §11 | classification before organising | the `docs/` scheme, one kind per file |
| L2 §2 | do not couple the method to an editor | one `.vscode/settings.json` of a single setting |
| L2 §3 | version control forms | all present |
| L2 §4 | ephemeral out of the working tree | verified below |

## The one real gap: no DOI

The L1 pattern *for publishing / making citable* lists three formalizations.
`CITATION.cff` landed on 2026-09-03. Dublin Core / DataCite and JOSS are venue
choices rather than requirements. **A DOI is the one item in that table that is
genuinely missing**, and the project says so in its own manual.

It is worth naming because the cost is near zero and it is the kind of thing
that cannot be applied retroactively to releases that are already out: Zenodo's
GitHub integration mints a DOI per release from the moment it is connected, and
a concept DOI that always resolves to the latest. Every release published
before it is connected has none, permanently.

This is a decision for the owner, not work to be scheduled. It costs one
authorisation and changes nothing in the repository except a badge and two
lines in `CITATION.cff`.

## The one piece of debris, now gone

A `.pytest_cache` directory dated 00:30 was in the working tree, from before the
cache redirect was configured. It had been deleted once already, and the
preserved timestamp suggests OneDrive restored it. Removed, and verified not
recreated: a plain `pytest` run now writes to `M:\caches\pytest` and leaves the
tree clean.

## The environment, against L2 §4

L2 §4 says to redirect the ephemeral out of the working tree and names the
mechanism as per-tool environment variables, giving `PIP_CACHE_DIR` and
`CARGO_TARGET_DIR` as examples and `Z:\caches\` as a plausible destination.
This machine does exactly that:

```
CARGO_TARGET_DIR       M:\caches\cargo
UV_CACHE_DIR           M:\caches\uv
PIP_CACHE_DIR          M:\caches\pip
RUFF_CACHE_DIR         M:\caches\ruff
MYPY_CACHE_DIR         M:\caches\mypy
PYTHONPYCACHEPREFIX    M:\caches\pycache
PYTEST_ADDOPTS         -o cache_dir=M:\caches\pytest
```

The virtual environment lives at `Z:\venvs\patchcraft` and is reached through a
directory junction, so the tree carries a `.venv` name without the 20,000 files
behind it. `build/` and `src/patchcraft.egg-info/` still appear during a build
and are the documented exception: redirecting them would require a machine path
in a tracked file, which §5 and this project's own history both argue against.

## What was deliberately not called a defect

Applying §9's genre rule rather than a checklist:

- **No Diataxis split** of `docs/` into tutorials, how-to, reference and
  explanation. The method lists that as an example instantiation and says the
  names are local while the principle, separating kinds, is L0. This project
  separates by kind already, on axes that suit a library.
- **No `tickets/`, no `experiments/{dirty,clean}/`.** Those are research-project
  shapes. `docs/design/` and `docs/STUDIES/` carry the same function at the
  scale a library needs.
- **No JOSS submission.** A venue, not a requirement, and the project's own
  gate for calling itself settled is an external consumer, which it does not
  have yet.

## Conclusion

Conformant for its genre. One gap named, the DOI, which is the owner's
decision; one piece of debris removed. No work is proposed beyond that, which
is the point of §9 and the answer this review is allowed to give.
