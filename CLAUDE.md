# PatchCraft: Claude Code

> **The canonical guide is [`AGENTS.md`](AGENTS.md)**, brand-free and the single
> source. **Read it first**: the invariants, the gates and the scope boundary
> live there, not here. This file exists only because Claude Code loads
> `CLAUDE.md` automatically, and it holds what is specific to this tool.
> On any disagreement, **`AGENTS.md` wins.**

## Entry route

1. [`AGENTS.md`](AGENTS.md), the rules. **Start at §0**, the eleven invariants:
   I1 measure before writing it down · I2 THEORY §9 is the arbiter · I3 a fix
   ships with a failing test · I4 the exactness suite falsifies · I5 the surface
   is 20 names · I6 both code paths stay green · I7 the version answers one
   question · I8 nothing ephemeral in the tree · I9 a tracked file ships ·
   I10 an ADR carries the choice, not the research · I11 do not invent what
   exists
2. [`MAP.md`](MAP.md), where things are
3. [`STATUS.md`](STATUS.md), where things stand and who each open item waits on
4. [`docs/FOCO-1.0.md`](docs/FOCO-1.0.md), what is left before 1.0
5. [`README.md`](README.md), the overview a human reads

## Specific to this tool

**Memory tiers.** User-scope memory (`~/.claude/projects/<slug>/memory/`) holds
preferences and process feedback that travel across projects, and is not in
this repository. Project-scope knowledge is [`AGENTS.md`](AGENTS.md) and
`docs/ADR/`, versioned in git. The test: if it would be true of any project,
it is user-scope; if it is true of PatchCraft, it belongs here.

**The environment is not the default one.** The interpreter lives at
`Z:\venvs\patchcraft`, reached through the `.venv` junction in this tree. Do
not create a `.venv` here. Caches go to `M:\caches` by user-level environment
variables, and `CARGO_TARGET_DIR` is one of them: building the Rust kernel
without it puts hundreds of megabytes back into the project folder.

**A path with a non-ASCII character breaks one thing.** This checkout sits
under `Acadêmicos`, and `setuptools-rust` mis-decodes the artifact path cargo
reports, so a native build fails on the copy step. `CARGO_TARGET_DIR` pointing
at an ASCII directory is the fix, and it is already set.

**Verify chained shell commands.** Two lint errors reached `main` in one day
because a `grep` or a `;` swallowed a non-zero exit and the chain continued.
Read the gate output rather than the last line of it.
