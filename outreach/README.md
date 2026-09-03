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
| [`2026-09-03-lancamento.md`](2026-09-03-lancamento.md) / [`2026-09-03-release.en.md`](2026-09-03-release.en.md) | the current news source (PT / EN) |
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
- superlatives. The hook is the retraction, not the advantage.

**The risk worth knowing.** These texts lead with the fact that the library published a
false claim about its own numerics. That is the most interesting thing in them and also the
easiest to read as a weakness. My view is that it works in the project's favour, because a
reader who works with numerics recognises what it means for an author to measure a claim,
find it wrong and publish the correction with the sweep behind it. But the choice is yours,
and both texts stand without it: cut the retraction section and the piece becomes an
ordinary "here is a small library" post, which is a weaker text and a safer one.

**Do not soften the limits section.** It is short, it is true, and it is the part that makes
the rest credible.
