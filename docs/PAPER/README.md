# `docs/PAPER/`: a mean that never leaves its domain

A scientific article in preparation, about computing a mean so that every
intermediate stays in the input's own numeric domain: divide first, then sum.
It draws on two projects. PatchCraft supplies the patch-grid mean and its
exactness rule; a private optical mark reader for lottery slips supplies the
frame averaging case study.

This folder is not part of the library's documentation and does not ship:
`MANIFEST.in` prunes it and `tools/check_dist.py` fails a release that carries
it. The library does not change because of it.

**Status, 2026-10-10.** A scaffold: the outline, the formal statements with
their proofs, the measurements and the references are in place; the prose of
the article is not written yet. [OUTLINE.md](OUTLINE.md) lists what is open,
and the first item, a search for prior art on the integer carry, comes before
any claim of novelty.

## What is here

| File | What it holds |
|---|---|
| [OUTLINE.md](OUTLINE.md) | working title and abstract, contributions, sections, the case study, open questions |
| [FORMAL.md](FORMAL.md) | definitions and propositions, each with a proof or the measurement behind it |
| [MEASUREMENTS.md](MEASUREMENTS.md) | every number the article states, with the command that reproduces it |
| [REFERENCES.md](REFERENCES.md) | 63 sources, what each is used for, and where it was opened |
| [sources.tsv](sources.tsv) | each reference's locator and a text the source must contain |
| [experiments/](experiments/) | `integer_mean.py`, `float_mean.py`, `fold_mean.py`, `check_sources.py` |

## Reproducing

From the repository root, with the development environment of the project:

```
python docs/PAPER/experiments/integer_mean.py
python docs/PAPER/experiments/float_mean.py
python docs/PAPER/experiments/fold_mean.py
python docs/PAPER/experiments/check_sources.py
```

The first three take seconds each and need only numpy, torch and the
repository. The last one reads the 85 source rows over the network and needs
`pdftotext` for the PDF rows; its downloads go to the system temp folder.

## Rules for this folder

- **Synthetic data only.** Frames are constants, noisy ramps or seeded random
  values, and small formats are enumerated whole.
- **The case study's code stays private.** It is cited by file, line and date,
  and only its pattern is reproduced. No image, document or data of that
  project enters this repository.
- **Every number has a command**, as everywhere else in the repository, and
  every reference has a row in `sources.tsv`, except the two marked as not
  opened.
