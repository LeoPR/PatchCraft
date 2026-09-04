<!-- l10n: doc_id=patchcraft-outreach-readme · lang=en · canonical -->
**English** · [Português](README.pt-BR.md)

# `outreach/`: material for presenting the project

Assets for showing PatchCraft publicly. This is support material, not part of the library
and not part of its documentation, which lives in [`docs/`](../docs/). It publishes no new
measurement: every number comes from a dated document in the repository, and each one names
the command that reproduces it.

## How it is organized

The **root** holds the dated **news source**: one file per announcement, carrying the state,
the headlines and the honest limits. The **subfolders** are the **channels**, each shaping
that source into a publication that respects the limits of its medium.

The rule that keeps the two in step: never edit a channel text without updating the dated
source first.

| Path | What it is |
|---|---|
| [`2026-09-04-lancamento.md`](2026-09-04-lancamento.md) / [`2026-09-04-release.en.md`](2026-09-04-release.en.md) | the current news source (PT / EN) |
| [`linkedin/`](linkedin/) | LinkedIn: `post.*` (short), `artigo.*` (long technical) |

Portuguese is the canonical language here, unlike the rest of the project, because the
audience these texts are aimed at reads Portuguese first. English is the translation.

## Channel limits

- **LinkedIn post** (`linkedin/post.*`): about 3,000 characters, and only the first two or
  three lines show before "see more", so the hook goes first. Hashtags at the end, without
  accents, because accented hashtags break LinkedIn search.
- **LinkedIn article** (`linkedin/artigo.*`): long form, headings and tables render, good
  for the version that carries the numbers. End with the repository link.

## Before you publish

**Every number was verified**, not estimated, and re-checked on 2026-09-03: the 3840 pixels
left at zero, the reshape that keeps the shape and loses the pixels, the 126,736 enumerated
geometries, the 3936 of 14969 mispredictions of the old rule against 8 of the new one, and
the whole benchmark table. The measurements live in `docs/PERFORMANCE.md` and the
`0.5.0`/`0.5.1` entries of the changelog.

**What these texts avoid on purpose:**

- comparing against other libraries. The baseline in every table is PatchCraft's own
  pure-torch path, and saying so keeps the discussion on the measurement rather than on
  whether the comparison was fair;
- saying "faster" without a geometry, a machine and a torch version attached;
- superlatives. The hook is the silent defect, not the advantage and not the retraction.

**Where the retraction goes, and why it moved.** Until 2026-09-04 these texts opened with
the fact that the library had published a false claim about its own numerics. That was the
wrong place for it, for a reason that has nothing to do with courage: the reader has never
seen the old claim. They have no before. So the retraction describes a state they did not
witness, and the only thing it actually transmits on first contact is that the library got
something wrong, delivered before the reader knows what the library is for.

It is now inside the section on the falsification suite, where it earns its place as the
answer to "why is that test shaped like that". The material did not get softer and no
number left; it stopped being the headline. What leads instead is the silent defect, which
is the one item in these texts that the reader may have in their own code at the moment
they read it.

**Do not soften the limits section.** It is short, it is true, and it is the part that makes
the rest credible.
