# ADR 0005: padding a grid that does not cover the image, the way the ecosystem already does

- **Status:** Proposed
- **Date:** 2026-10-10
- **Deciders:** Leonardo Marques de Souza
- **Relates to:** [ADR 0001](0001-patch-extraction-api.md) (truncation as the border policy), [ADR 0003](0003-reversibility-classes.md) (exactness per regime), [`THEORY.md`](../THEORY.md) §2.6, §9.1, §9.2, [`FOCO-1.0.md`](../FOCO-1.0.md) §3 (padding as the named 1.1 candidate), [`STUDIES/2026-10-10-border-policy.md`](../STUDIES/2026-10-10-border-policy.md) (every measurement and source cited here)

## Context

When a grid does not cover the image, `extract` drops the remainder and
`reconstruct` refuses. It is the problem every convolution has at the border,
and the libraries around PatchCraft solved it long ago. kornia solves it for
exactly this pair of operations. This ADR takes that solution rather than
designing one.

## What is decided here, and what is not

**The design is not a decision.** The pattern, the parameter's name, its
values, where the padding goes and what happens to an image smaller than the
patch are all determined by precedent, read in the pinned sources the study
lists. Section 1 states them.

**What remains is the owner's.** Whether to build it, and whether before 1.0
or as the 1.1 that FOCO-1.0 already names. It adds no public symbol, only a
keyword with a default, so the ROADMAP's consumer gate does not block it by
its letter; whether to build ahead of a consumer is the question that gate
exists to ask.

**Nothing here is implemented.**

## 1. The design, borrowed

| Piece | What it is | Taken from |
|---|---|---|
| The pattern | Pad on the way in by an amount computed from the geometry; remove the same amount on the way back | kornia's `extract_tensor_patches(allow_auto_padding=)` and `combine_tensor_patches(allow_auto_unpadding=)` |
| The name | `padding_mode`, keyword-only | torch's `Conv2d`, torchvision's `Pad`, MONAI's `sliding_window_inference` |
| The values | `"constant"`, `"reflect"`, `"replicate"`, `"circular"`, plus `None` | `torch.nn.functional.pad`, as MONAI uses them |
| The default | `None`, meaning today's behaviour exactly: `extract` truncates, `reconstruct` refuses | no call changes |
| The amount, per axis | `p - n` when the side `n` is shorter than the patch, otherwise `(-(n - p)) mod s` | kornia for the general case, MONAI for the short side, which kornia's formula misses |
| The placement | `amount // 2` before, the rest after | kornia, MONAI's `"symmetric"`, Albumentations' `"center"` |
| The fill for `"constant"` | zero | `F.pad`'s default, kornia, timm |

**Where it applies.** `extract` and `Patchify` pad and then cut. `num_patches`
counts on the padded canvas, so planning matches. `reconstruct` and `stitch`
take the same keyword, compute the same amounts from `image_shape`, the patch
and the stride, assemble the padded canvas and crop it back to `image_shape`.
There the mode cannot change the result, since the crop discards every padded
pixel; only `None` against a mode matters. It is one keyword on both sides,
rather than kornia's two flags, so that the second call repeats the first, as
it already does with `stride`.

**`pair`** pads the LR image as above and derives every HR amount as the scale
times the LR amount. Centring the HR image on its own misaligns the pair by
half an LR pixel whenever the LR amount is odd and the scale is even.

**Exactness.** THEORY §9.2's rule is evaluated on the padded geometry. On the
frame the study measures, 1080×1920 at patch 256 and stride 128, the padded
map holds only 1, 2 and 4 and the round trip is exact.

## 2. What is deferred, and until when

Each of these exists in the ecosystem, and each is additive later. None is
built until a caller needs it.

- **A fill value other than zero**, which `F.pad` calls `value`, MONAI `cval`
  and torchvision `fill`.
- **Padding at the end only.** Only timm does it. If wanted, it arrives as
  MONAI's `method="end"` or Albumentations' `position="top_left"`, whichever
  the precedent check then favours. With one fixed placement there is nothing
  for `extract` and `reconstruct` to disagree about.
- **Moving the last window back to the border**, MONAI's sliding window. It is
  the only strategy that neither loses nor invents a pixel, and it gives up the
  regular grid and the exactness rule to do it.
- **`"symmetric"`**, numpy's and torchvision's mode that repeats the edge
  pixel. `F.pad` does not have it.
- **Centring without padding.** A grid centred inside the image still drops
  the remainder, half on each side; torchvision's `CenterCrop` before
  `extract` does it today.

## 3. Consequences

- **Additive.** Every existing call returns what it returns today, so it is a
  minor release.
- **Surface.** One keyword on `extract`, `Patchify`, `num_patches`,
  `reconstruct`, `stitch` and `pair`; `tests/test_public_api.py` pins the new
  signatures. `tilings` does not change: it enumerates the geometries that
  cover exactly, which padding does not need.
- **Contract.** THEORY §9 gains the padded regime in each function's section.
  ADR 0001's truncation stays the default, and stays true.
- **Tests.** The round trip on both code paths, in every mode, for remainders
  odd and even, for an image shorter than the patch, and for an LR/HR pair at
  an odd LR amount and an even scale.
- **GUIDE §6.** The padding recipe becomes one keyword. The library centres
  where the recipe pads the end, and the section has to say so.
