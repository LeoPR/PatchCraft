# Border policy: what exists, and what was measured

Date: 2026-10-10. CPU, torch 2.14.0+cpu, numpy 2.5.2.

These are the measurements and the survey behind
[ADR 0005](../ADR/0005-border-policy.md), kept here so that the ADR reads as a
decision. The numbers come from one command:

```
python tools/compare_border_strategies.py
```

## The problem

A regular grid of patch `p` and stride `s` covers an axis of length `n` exactly
when `(n - p)` is a multiple of `s`. When it is not, `extract` drops the
remainder, which is the border policy of ADR 0001, and `reconstruct` refuses,
by THEORY §9.2. The answer to "my image does not tile" is a `ValueError` or the
padding recipe in GUIDE §6.

This is the problem every convolution has at the image border, and the
ecosystem has answered it many times. Nothing below is new; the point of this
study is to find the answer closest to PatchCraft's shape and take it.

## Five strategies on one frame

A 1080×1920 frame, patch 256, stride 128. The regular grid ends at row 1024,
so 56 rows are left over.

| Strategy | Rows lost | Rows outside the frame | Most in one patch | Counts | Round trip exact |
|---|---|---|---|---|---|
| truncate, remainder at the end (`extract` today) | 56 | 0 | 0 | 1, 2, 4 | no round trip |
| truncate, remainder split | 56 | 0 | 0 | 1, 2, 4 | no round trip |
| pad, all at the end (GUIDE §6) | 0 | 72 | 72 | 1, 2, 4 | yes |
| pad, split between both sides | 0 | 72 | 36 | 1, 2, 4 | yes |
| move the last window back to the border | 0 | 0 | 0 | 1, 2, 3, 4, 6 | no |

Of three properties, losing no pixel, putting no pixel from outside the frame
into a patch, and an exact round trip, every strategy keeps two. Truncation
gives up the border. Padding puts pixels the frame never had into the edge
patches, and a crop after `reconstruct` keeps them out of the output.
Moving the last window keeps every pixel real and complete, but it overlaps the
last strip more than the rest, puts a 3 in the count map and so loses the
exactness rule, and its grid no longer has a single stride, so `F.fold` cannot
assemble it in one call.

Splitting the remainder does not change what a strategy loses or invents. It
changes where: with padding, the worst patch holds 36 rows from outside the
frame instead of 72, 14 percent of its height instead of 28.

## An LR/HR pair cannot be centred one image at a time

At scale 2, with the stride equal to the patch, an LR side of 33 and patch 8
leaves a remainder of 1, so centring it starts the grid at 0. The HR side of 66
and patch 16 leaves 2 and starts at 1, while the LR start scaled is 0. Padding
fails the same way: the LR side needs 7 more pixels, 3 of them before, and the
HR side needs 14, 7 before, while 3 scaled is 6. Either way the pair comes out
one HR pixel apart, half an LR pixel, whenever the LR amount is odd and the
scale is even. The HR amounts have to be the LR amounts times the scale, never
computed from the HR image.

## What exists

Every row was read in the pinned source, at the lines given.

| Library | Where | What it does |
|---|---|---|
| kornia 0.8.0 | `kornia/contrib/extract_patches.py`: `extract_tensor_patches` and `combine_tensor_patches` | The same pair of operations as `extract` and `reconstruct`. Without `allow_auto_padding`, a grid that does not fit warns and drops the incomplete patches (476-490). With it, `compute_padding` (53-115) pads with zeros, split between both sides, the odd pixel at the bottom and right. `combine_tensor_patches` with `allow_auto_unpadding` computes the same padding from the geometry and removes it (362-380). |
| MONAI 1.4.0 | `monai/inferers/utils.py`: `sliding_window_inference` | `padding_mode`, typed as `PytorchPadMode`, whose values are `F.pad`'s, default `constant`, with `cval` for the fill (51). Pads only when the image is smaller than the window, split `(half, diff - half)` (160-171). Otherwise the last window is moved back to end at the border (`monai/data/utils.py`, `dense_patch_slices`, 196-203). |
| MONAI 1.4.0 | `monai/transforms/croppad/array.py`: `SpatialPad`, `DivisiblePad` | `method="symmetric"` (default) or `"end"` (214, 232-238, 314; the enum is in `monai/utils/enums.py`, 292-298). `DivisiblePad` pads to a multiple of `k`. |
| Albumentations 1.4.18 | `augmentations/geometric/transforms.py`: `PadIfNeeded` | `pad_height_divisor` and `pad_width_divisor` pad to a multiple; `position` defaults to `"center"`, with `"top_left"` and four others (1171-1284; the values in `core/types.py`, 105); `border_mode` defaults to OpenCV's `BORDER_REFLECT_101`. |
| timm 1.0.9 | `timm/layers/patch_embed.py`: `PatchEmbed` | `dynamic_img_pad=True` pads with zeros at the right and bottom only, up to a multiple of the patch (127-130). |
| torchvision 0.20.0 | `torchvision/transforms/v2/_geometry.py` | `Pad(padding_mode=...)` with `"constant"`, `"edge"`, `"reflect"`, `"symmetric"` (419, 471); `RandomCrop(pad_if_needed=...)` (779-854); `CenterCrop` (170). |
| torch | `torch.nn.functional.pad`, `torch.nn.Conv2d` | `F.pad(mode=...)` with `"constant"`, `"reflect"`, `"replicate"`, `"circular"` and `value=`; `Conv2d(padding_mode=...)` with `"zeros"`, `"reflect"`, `"replicate"`, `"circular"`. |

**kornia's formula misses one case.** `compute_padding` takes the remainder of
`(n - p)` modulo the stride, and when the image is smaller than the patch by a
multiple of the stride that remainder is 0: at `n = 2`, `p = 8`, `s = 2` it pads
nothing, and `_check_patch_fit` (301-311) reports a fit, so no patch is
produced. MONAI's rule for a small image, pad up to the window, covers it.

## The names do not agree

The same five modes, applied to `1 2 3 4 5` with two pixels on each side:

| Result | torch `F.pad` | numpy `np.pad` |
|---|---|---|
| `0 0 1 2 3 4 5 0 0` | `constant` | `constant` |
| `1 1 1 2 3 4 5 5 5` | `replicate` | `edge` |
| `3 2 1 2 3 4 5 4 3` | `reflect` | `reflect` |
| `2 1 1 2 3 4 5 5 4` | not available | `symmetric` |
| `4 5 1 2 3 4 5 1 2` | `circular` | `wrap` |

From their documentation, not measured here: scipy's `ndimage` calls the
fourth row `reflect` and the third `mirror`, and OpenCV calls them
`BORDER_REFLECT` and `BORDER_REFLECT_101`. The word "reflect" therefore names
two different operations depending on the library.

## What the survey settles

- **The pattern** is kornia's: pad on the way in by an amount computed from the
  geometry, remove the same amount on the way back. It is the exact analogue
  of `extract` and `reconstruct`.
- **The name and its values**: `padding_mode` is the name torch's `Conv2d`,
  torchvision's `Pad` and MONAI's sliding window all use, and MONAI pairs it
  with `F.pad`'s values, which are what PatchCraft would call.
- **The placement** is split between both sides with the odd pixel at the end
  in kornia, MONAI and Albumentations by default; only timm pads the end
  alone.
- **A small image** is padded up to the window, as MONAI does.
- **Moving the last window** is what MONAI's sliding window does when the
  image is larger than the window. It keeps every pixel real, at the price of
  exactness and of a regular grid.
