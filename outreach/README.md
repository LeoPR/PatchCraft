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
| [`linkedin/`](linkedin/) | LinkedIn: `post.*` (short), `artigo.*` (long technical), and the figures |

The figures live in `linkedin/figuras/<language>/`, one subfolder per language so the
channel directory does not mix prose with binaries. `tools/make_outreach_figures.py` builds
them all, PNG for uploading and SVG beside it for editing, and running the script
regenerates everything. They obey the same rule the numbers in the text do: there is a
command that reproduces them.

None of them is an illustration. The image panels are the tensors `extract` and
`reconstruct` actually return, the error in the corner of the approximate panel is the real
difference amplified until it is visible, the refusal text is what `reconstruct` raises, and
the coverage maps come from torch's own fold and unfold over a tensor of ones.

It is one page per language, `pagina.png` and `page.png`, in three blocks: the cut and the
reshape that scrambles, the per-stride comparison between hand-written `fold`/`unfold` and
PatchCraft, and the typical case on an MNIST digit. Where both paths return the same tensor
the page says so instead of repeating the image, because the difference there is not the
arithmetic, it is the contract.

MNIST is downloaded on first use, and the page degrades without it: the third block is
dropped and the script says so rather than failing.

Portuguese is the canonical language here, unlike the rest of the project, because the
audience these texts are aimed at reads Portuguese first. English is the translation.

## Channel limits

- **LinkedIn post** (`linkedin/post.*`): about 3,000 characters, and only the first two or
  three lines show before "see more". Those lines cannot carry jargon: the LinkedIn
  audience is wide, and a first sentence that only speaks to people who already know the
  subject filters instead of inviting. Context before jargon, density without a lecturing
  tone, and a close that closes rather than stops. Hashtags at the end, without accents,
  because accented hashtags break LinkedIn search.
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

**The surface carries no development history.** These texts say what the library does
today. The road here, including the numerical claim that was published, measured and
retracted, lives in the CHANGELOG, in ADR 0003 and in the dated studies, which is where
someone goes on purpose to look for it.

The reason is about the reader, not about courage: whoever arrives now never saw the old
version. Telling them about the correction describes nothing they witnessed, and the only
thing it conveys on first contact is that the library got something wrong, before they know
what the library is for. Worse when the defect being reported was in the tests, because
then the reader finishes knowing nothing about the product and wondering whether the suite
works.

The test of this, for the next text: every paragraph has to answer "what does this tell me
about using the library?". An old measurement may stay, as long as it enters as an argument
for the current rule rather than as an account of what happened. The comparison between the
maximum rule and the power-of-two rule is the example: it explains why the contract is what
it is.

**Do not soften the limits section.** It is short, it is true, and it is the part that makes
the rest credible.
