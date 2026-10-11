"""Draw the figures of docs/GUIDE.md from computation rather than by hand.

Nine figures into docs/figures/, each named after the GUIDE section it sits
in:

  1-silent              the reshape that scrambles and the fold that blacks out
  2-patch-order         what extract returns, and in which order
  2-count-map           where patches overlap, and what reconstruct does there
  2-votes               a per-patch decision, averaged where patches overlap
  4-where-it-misses     the exactness rule on an image: misses only off 1, 2, 4
  5-windows             what uniform, hann and gaussian do to a seam
  5-halo                a per-tile blur with and without a halo
  7-pair                one LR/HR pair, and the stride applied twice
  7-antialias           a zone plate reduced with and without the filter

Every panel is a tensor the library returned. Every result the GUIDE states
next to a figure is a claim here: it is asserted against the measurement
before anything is drawn, printed, and checked to appear in the paragraph
that follows the figure in the GUIDE.

    python tools/make_guide_figures.py           # write docs/figures/*.png
    python tools/make_guide_figures.py --check   # write nothing; exit 1 if a
                                                 # figure drifted (see below)

--check renders every figure in memory and fails if a committed PNG differs
from the fresh render outside the text labels on more than 0.1 percent of its
pixels, if a claim is missing from its caption, if a figure is not linked from
the GUIDE, or if a PNG in docs/figures is linked from nowhere. Labels use
Pillow's bundled font (Pillow 10.1 or newer), so Windows and Linux draw the
same glyphs, but text rasterization may still differ between FreeType builds,
which is why the comparison leaves the label boxes out. The 0.1 percent is
for the last bit of a float32 `cos` or sum, which may differ between CPUs and
flip a pixel that sits on an 8-bit rounding boundary; a changed figure moves
far more pixels than that. tests/test_docs_figures.py runs --check, so the
suite fails when a figure drifts.
"""

from __future__ import annotations

import argparse
import io
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)
from PIL import Image, ImageDraw, ImageFont

from patchcraft import (
    extract,
    num_patches,
    pair,
    paired_tilings,
    patch_metrics,
    reconstruct,
    resize,
    scale_factor,
    stitch,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_outreach_figures as outreach  # the shared test image, defects and colours

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"
GUIDE = ROOT / "docs" / "GUIDE.md"

WHITE = (255, 255, 255)
INK, MUTED, RULE, LOSS = outreach.INK, outreach.MUTED, outreach.RULE, outreach.LOSS
M, GAP, TOP = 16, 24, 30          # outer margin, gap between panels, band for one label line
PANEL = 256                       # the side of a full panel in pixels
PALETTE = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
           (148, 103, 189), (140, 86, 75), (227, 119, 194)]


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size)


def to_image(t: torch.Tensor, scale: int) -> Image.Image:
    """A (C, H, W) tensor in [0, 1] as an RGB image, enlarged by nearest neighbour."""
    t = t.repeat(3, 1, 1) if t.shape[0] == 1 else t
    arr = (t.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    im = Image.fromarray(arr)
    return im.resize((im.width * scale, im.height * scale), Image.NEAREST)


def rgb(colour: tuple[int, int, int]) -> torch.Tensor:
    return torch.tensor(colour, dtype=torch.float32).view(3, 1) / 255


def fade(t: torch.Tensor, keep: float = 0.25) -> torch.Tensor:
    """A pale grey version of an image, to draw marks on."""
    grey = t.mean(0, keepdim=True).repeat(3, 1, 1)
    return 1 - keep * (1 - grey)


def count_map(shape: tuple[int, int, int], patch: int, stride: int) -> torch.Tensor:
    """How many patches cover each pixel, zero where the grid never reaches."""
    _, h, w = shape
    nh, nw = (h - patch) // stride + 1, (w - patch) // stride + 1
    eh, ew = (nh - 1) * stride + patch, (nw - 1) * stride + patch
    folded = F.fold(torch.ones(1, patch * patch, nh * nw), (eh, ew), patch, stride=stride)
    full = torch.zeros(h, w)
    full[:eh, :ew] = folded[0, 0]
    return full


def colour_counts(counts: torch.Tensor) -> torch.Tensor:
    """Blue for counts of 1, 2 and 4, amber for every other, darker as the count grows."""
    vmax = max(int(counts.max()), 2)
    t = torch.log2(counts.clamp(min=1)) / math.log2(vmax)
    blue = torch.stack([223 + (30 - 223) * t, 236 + (80 - 236) * t, 248 + (130 - 248) * t]) / 255
    amber = torch.stack([250 + (170 - 250) * t, 226 + (105 - 226) * t, 180 + (15 - 180) * t]) / 255
    ok = (counts == 1) | (counts == 2) | (counts == 4)
    return torch.where(ok.unsqueeze(0), blue, amber)


def u8(t: torch.Tensor) -> torch.Tensor:
    return (t.clamp(0, 1) * 255).round().to(torch.uint8)


class Figure:
    """One PNG, the boxes its text occupies, and the claims its caption must state."""

    def __init__(self, name: str, width: int, height: int) -> None:
        self.name = name
        self.im = Image.new("RGB", (width, height), WHITE)
        self.d = ImageDraw.Draw(self.im)
        self.text_boxes: list[tuple[int, int, int, int]] = []
        self.claims: list[str] = []
        self.grey = False

    def text(self, xy: tuple[float, float], string: str, size: int = 15,
             colour: tuple[int, int, int] = INK) -> None:
        assert string.isascii(), string           # the bundled font has no other glyphs
        f = font(size)
        self.d.text(xy, string, font=f, fill=colour)
        x0, y0, x1, y1 = self.d.textbbox(xy, string, font=f)
        self.text_boxes.append((int(x0), int(y0), math.ceil(x1), math.ceil(y1)))

    def panel(self, t: torch.Tensor, xy: tuple[int, int], scale: int, label: str = "") -> None:
        im = to_image(t, scale)
        self.im.paste(im, xy)
        # A hairline outside the panel, so that white pixels do not merge into the page.
        self.d.rectangle([xy[0] - 1, xy[1] - 1, xy[0] + im.width, xy[1] + im.height], outline=RULE)
        if label:
            self.text((xy[0], xy[1] - TOP + 6), label)

    def claim(self, string: str) -> None:
        self.claims.append(string)
        print(f"    claim: {string}")

    def png(self) -> bytes:
        buf = io.BytesIO()
        (self.im.convert("L") if self.grey else self.im).save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    def text_mask(self, pad: int = 2) -> np.ndarray:
        mask = np.zeros((self.im.height, self.im.width), dtype=bool)
        for x0, y0, x1, y1 in self.text_boxes:
            mask[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad] = True
        return mask


# --------------------------------------------------------------------------
# The figures, in GUIDE order
# --------------------------------------------------------------------------


def fig_silent(img: torch.Tensor) -> Figure:
    shape = tuple(img.shape)
    cols = F.unfold(img.unsqueeze(0), kernel_size=32, stride=32)
    scrambled = reconstruct(cols[0].view(-1, 3, 32, 32), shape, stride=32)  # the intuitive reshape
    assert torch.equal(scrambled, outreach.naive_reshape(img))
    assert sorted(scrambled.flatten().tolist()) == sorted(img.flatten().tolist())  # a permutation
    moved = float((scrambled != img).float().mean())
    hand = outreach.hand_fold(img, 20)
    black = int((hand == 0).all(0).sum())
    assert black == 3840 and (hand[:, :112, :112] - img[:, :112, :112]).abs().max() < 1e-6

    fig = Figure("1-silent", M + 3 * PANEL + 2 * GAP + M, TOP + PANEL + M)
    fig.panel(img, (M, TOP), 2, "the image")
    fig.panel(scrambled, (M + PANEL + GAP, TOP), 2, "unfold, then the intuitive reshape")
    fig.panel(hand, (M + 2 * (PANEL + GAP), TOP), 2, "fold by hand, patch 32, stride 20")
    fig.claim(f"{moved * 100:.1f} percent")
    fig.claim(f"{black} pixels")
    fig.claim(f"{black / 128 / 128 * 100:.1f} percent")
    return fig


def fig_patch_order(img: torch.Tensor) -> Figure:
    patches = extract(img, patch_size=32, stride=32)
    nh, nw = num_patches(tuple(img.shape), patch_size=32, stride=32)
    assert tuple(patches.shape) == (16, 3, 32, 32) and (nh, nw) == (4, 4)
    for k in range(len(patches)):
        top, left = (k // nw) * 32, (k % nw) * 32
        assert torch.equal(patches[k], img[:, top:top + 32, left:left + 32])

    tile, tgap = 64, 6
    fig = Figure("2-patch-order", M + PANEL + 32 + 8 * tile + 7 * tgap + M, TOP + PANEL + M)
    fig.panel(img, (M, TOP), 2, "image, (3, 128, 128)")
    for k in range(1, 4):
        fig.d.line([(M + k * 64, TOP), (M + k * 64, TOP + PANEL - 1)], fill=WHITE)
        fig.d.line([(M, TOP + k * 64), (M + PANEL - 1, TOP + k * 64)], fill=WHITE)
    for k in range(16):
        r, c = divmod(k, 4)
        x, y = M + c * 64 + 3, TOP + r * 64 + 3
        fig.d.rectangle([x, y, x + 19, y + 15], fill=WHITE)
        fig.text((x + 3, y + 1), str(k), 12)
    x0 = M + PANEL + 32
    fig.text((x0, 6), "extract(image, 32, stride=32), (16, 3, 32, 32)")
    for k in range(16):
        r, c = divmod(k, 8)
        x, y = x0 + c * (tile + tgap), TOP + r * (tile + 22)
        fig.panel(patches[k], (x, y), 2)
        fig.text((x + 28, y + tile + 3), str(k), 12, MUTED)
    fig.claim("(16, 3, 32, 32)")
    return fig


def fig_count_map(img: torch.Tensor) -> Figure:
    shape = tuple(img.shape)
    _, nw = num_patches(shape, patch_size=32, stride=16)
    counts = count_map(shape, 32, 16)
    assert sorted({int(v) for v in counts.unique()}) == [1, 2, 4]
    stack = extract(img, patch_size=32, stride=16)
    stack[0] = 1.0                                     # patch 0 says white
    back = reconstruct(stack, shape, stride=16)
    foot = torch.zeros(shape[1:], dtype=torch.bool)
    foot[:32, :32] = True
    assert torch.equal(back[:, ~foot], img[:, ~foot])  # nothing outside patch 0 moves
    # Inside, each pixel moves 1/count of the way to white: the mean of its patches.
    assert ((back - (img + (1 - img) / counts)).abs()[:, foot].max()) < 1e-6
    assert torch.equal(stitch(stack, shape, stride=16, weight="uniform"), back)

    fig = Figure("2-count-map", M + 3 * PANEL + 2 * GAP + M, TOP + PANEL + M)
    fig.panel(colour_counts(counts), (M, TOP), 2, "count map, patch 32, stride 16")
    outlined = [(0, PALETTE[3]), (1, PALETTE[1]), (nw, PALETTE[2]), (nw + 1, PALETTE[6])]
    for inset, (k, colour) in enumerate(outlined):
        y0, x0 = (k // nw) * 16 * 2 + TOP, (k % nw) * 16 * 2 + M
        fig.d.rectangle([x0 + inset, y0 + inset, x0 + 63 - inset, y0 + 63 - inset],
                        outline=colour, width=2)
    for (y, x), v in (((120, 8), 1), ((120, 60), 2), ((60, 60), 4)):
        fig.text((M + x * 2 - 4, TOP + y * 2 - 9), str(v), 16, WHITE if v == 4 else INK)
    corner = (slice(None), slice(0, 64), slice(0, 64))
    fig.panel(img[corner], (M + PANEL + GAP, TOP), 4, "the image, top-left corner")
    fig.panel(back[corner], (M + 2 * (PANEL + GAP), TOP), 4, "patches[0] set to white, reconstruct")
    fig.claim("half")
    fig.claim("a quarter")
    return fig


def fig_votes(img: torch.Tensor) -> Figure:
    grey = img.mean(0, keepdim=True)
    per_patch = {}
    for s in (32, 16):
        p = extract(grey, patch_size=32, stride=s)
        decided = (p > p.mean(dim=(1, 2, 3), keepdim=True)).float()
        per_patch[s] = reconstruct(decided, grey.shape, stride=s)
    once = reconstruct(extract((grey > grey.mean()).float(), 32, 16), grey.shape, stride=16)
    assert sorted(per_patch[32].unique().tolist()) == [0.0, 1.0]
    assert sorted(per_patch[16].unique().tolist()) == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert sorted(once.unique().tolist()) == [0.0, 1.0]
    between = float(((per_patch[16] > 0) & (per_patch[16] < 1)).float().mean())

    fig = Figure("2-votes", M + 3 * PANEL + 2 * GAP + M, TOP + PANEL + M)
    fig.grey = True
    fig.panel(per_patch[32], (M, TOP), 2, "a threshold per patch, stride 32")
    fig.panel(per_patch[16], (M + PANEL + GAP, TOP), 2, "a threshold per patch, stride 16")
    fig.panel(once, (M + 2 * (PANEL + GAP), TOP), 2, "one threshold, before cutting")
    fig.claim(f"{between * 100:.1f} percent")
    return fig


def fig_where_it_misses(img: torch.Tensor) -> Figure:
    shape = tuple(img.shape)
    back = reconstruct(extract(img, patch_size=32, stride=12), shape, stride=12)
    counts = count_map(shape, 32, 12)
    assert sorted({int(v) for v in counts.unique()}) == [1, 2, 3, 4, 6, 9]
    missed = (back != img).any(0)
    in_rule = (counts == 1) | (counts == 2) | (counts == 4)
    assert not (missed & in_rule).any()                # every miss sits off 1, 2 and 4
    metrics = patch_metrics(back, img)
    assert torch.equal(u8(back), u8(img))              # identical once saved as 8 bits
    off_rule = float(missed.sum()) / float((~in_rule).sum())

    marked = fade(colour_counts(counts), 0.6)
    marked[:, missed] = rgb(LOSS)
    fig = Figure("4-where-it-misses", M + 2 * PANEL + GAP + M, TOP + PANEL + M)
    fig.panel(colour_counts(counts), (M, TOP), 2, "count map, patch 32, stride 12")
    fig.panel(marked, (M + PANEL + GAP, TOP), 2, "pixels that came back different")
    fig.claim(f"{metrics['max_abs']:.4e}")
    fig.claim(f"{metrics['psnr_db']:.1f} dB")
    fig.claim(f"{(1 / 255) / metrics['max_abs']:,.0f} times")
    fig.claim(f"{off_rule * 100:.0f} percent")
    return fig


def fig_windows(img: torch.Tensor) -> Figure:
    img = img.double()                                 # so no pixel sits a last bit from rounding
    kinds = ("uniform", "hann", "gaussian")
    n = 7                                              # patches of 32 at stride 16 on 128
    share = {}
    for kind in kinds:
        rows = []
        for k in range(n):
            one = torch.zeros(n, 1, 1, 32, dtype=torch.float64)
            one[k] = 1.0                               # patch k says 1, the others 0
            rows.append(stitch(one, (1, 1, 128), stride=(1, 16), weight=kind)[0, 0])
        share[kind] = torch.stack(rows)                # share of patch k in each pixel
        assert (share[kind].sum(0) - 1).abs().max() <= (0 if kind == "uniform" else 1e-9)
    assert torch.equal(share["uniform"][3, 48:80], torch.full((32,), 0.5, dtype=torch.float64))
    g = torch.Generator().manual_seed(7)
    offset = (torch.rand(49, 1, 1, 1, generator=g) * 0.2 - 0.1).double()
    stitched = {k: stitch(extract(img, 32, 16) + offset, img.shape, stride=16, weight=k)
                for k in kinds}

    height = 110
    fig = Figure("5-windows", M + 3 * PANEL + 2 * GAP + M, TOP + height + 22 + PANEL + M)
    for i, kind in enumerate(kinds):
        x = M + i * (PANEL + GAP)
        fig.text((x, 6), kind)
        for level in (0.0, 0.5, 1.0):
            y = TOP + height - level * height
            fig.d.line([(x, y), (x + PANEL - 1, y)], fill=RULE)
        for k in range(n):
            # Rounded, because Pillow truncates: a share of 0.99999 would land a row below 1.
            points = [(x + c * 2, round(TOP + height - float(share[kind][k, c]) * height))
                      for c in range(128)]
            fig.d.line(points, fill=PALETTE[k], width=2)
        fig.panel(stitched[kind], (x, TOP + height + 22), 2)
    fig.text((M - 12, TOP + height - 7), "0", 11, MUTED)
    fig.text((M - 12, TOP - 7), "1", 11, MUTED)
    fig.claim("0.5")
    fig.claim("0.1")
    return fig


def fig_halo(img: torch.Tensor) -> Figure:
    # float64, because convolution picks its summation order by input size and by machine:
    # in float32 a last-bit difference could flip an 8-bit pixel and the comparison with it.
    img = img.double()
    shape = tuple(img.shape)
    r, tile = 2, 32
    kernel = torch.ones(3, 1, 2 * r + 1, 2 * r + 1, dtype=torch.float64) / (2 * r + 1) ** 2
    padded = F.pad(img.unsqueeze(0), (r, r, r, r), mode="reflect")[0]
    whole = F.conv2d(padded.unsqueeze(0), kernel, groups=3)[0]
    tiles = extract(img, patch_size=tile, stride=tile)
    alone = F.conv2d(F.pad(tiles, (r, r, r, r), mode="reflect"), kernel, groups=3)
    no_halo = reconstruct(alone, shape, stride=tile)
    with_halo = reconstruct(F.conv2d(extract(padded, tile + 2 * r, tile), kernel, groups=3),
                            shape, stride=tile)
    wrong = (u8(no_halo) != u8(whole)).any(0)
    y = torch.arange(128)
    near = (((y % tile) < r) | ((y % tile) >= tile - r)) & (y >= r) & (y < 128 - r)
    assert not (wrong & ~(near.unsqueeze(1) | near.unsqueeze(0))).any()  # only along tile edges
    assert torch.equal(u8(with_halo), u8(whole))
    worst = round(float((no_halo - whole).abs().max()) * 255)

    marked = fade(img).float()
    marked[:, wrong] = rgb(LOSS)
    fig = Figure("5-halo", M + 3 * PANEL + 2 * GAP + M, TOP + PANEL + M)
    fig.panel(img, (M, TOP), 2, "one tile and its halo")
    t0 = tile                                          # the tile whose top-left is (32, 32)
    fig.d.rectangle([M + (t0 - r) * 2, TOP + (t0 - r) * 2, M + (t0 + tile + r) * 2 - 1,
                     TOP + (t0 + tile + r) * 2 - 1], outline=WHITE, width=2)
    fig.d.rectangle([M + t0 * 2, TOP + t0 * 2, M + (t0 + tile) * 2 - 1,
                     TOP + (t0 + tile) * 2 - 1], outline=INK, width=2)
    fig.panel(marked, (M + PANEL + GAP, TOP), 2, "without a halo: pixels that differ")
    fig.panel(fade(img), (M + 2 * (PANEL + GAP), TOP), 2, "with a halo: pixels that differ")
    fig.claim(f"{float(wrong.float().mean()) * 100:.1f} percent")
    fig.claim(f"{worst}/255")
    return fig


def fig_pair(img: torch.Tensor) -> Figure:
    hr = img
    lr = resize(hr, (64, 64), backend="torch", antialias=True)
    assert scale_factor(tuple(lr.shape), tuple(hr.shape)) == 2
    lp, st, k = 16, 4, 28
    assert any(s.lr.patch_size == (lp, lp) and s.lr.stride == (st, st)
               for s in paired_tilings(tuple(lr.shape), tuple(hr.shape), allow_overlap=True))
    result = pair(lr, hr, lr_patch_size=lp, scale_factor=2, stride=st)
    meta = result.metas[k]
    nw = (64 - lp) // st + 1
    assert (meta.row, meta.col) == (8, 8)
    assert torch.equal(result.lr_patches[k], lr[:, 8:8 + lp, 8:8 + lp])
    assert torch.equal(result.hr_patches[k], hr[:, 16:16 + 2 * lp, 16:16 + 2 * lp])
    wr, wc = meta.row * st, meta.col * st              # the stride applied a second time
    k_wrong = (wr // st) * nw + wc // st
    assert torch.equal(result.lr_patches[k_wrong], lr[:, wr:wr + lp, wc:wc + lp])

    fig = Figure("7-pair", M + PANEL + GAP + PANEL + GAP + 128 + M, TOP + PANEL + M)
    fig.panel(lr, (M, TOP), 4, "LR, (3, 64, 64)")
    fig.panel(hr, (M + PANEL + GAP, TOP), 2, "HR, (3, 128, 128)")
    for x0, s in ((M, 4), (M + PANEL + GAP, 4)):       # 4 screen pixels per LR pixel on both
        fig.d.rectangle([x0 + meta.col * s, TOP + meta.row * s, x0 + (meta.col + lp) * s - 1,
                         TOP + (meta.row + lp) * s - 1], outline=WHITE, width=3)
    fig.d.rectangle([M + wc * 4, TOP + wr * 4, M + (wc + lp) * 4 - 1, TOP + (wr + lp) * 4 - 1],
                    outline=LOSS, width=3)
    x_right = M + 2 * (PANEL + GAP)
    fig.panel(result.lr_patches[k], (x_right, TOP), 6, f"patch {k}, LR")
    fig.panel(result.hr_patches[k], (x_right, TOP + 96 + 8 + TOP), 3, f"patch {k}, HR")
    fig.claim(f"patch {k}")
    fig.claim(f"row {meta.row} and column {meta.col}")
    fig.claim(f"patch {k_wrong}")
    return fig


def fig_antialias() -> Figure:
    n, fmax = 256, 0.35
    yy, xx = torch.meshgrid(torch.arange(n, dtype=torch.float64),
                            torch.arange(n, dtype=torch.float64), indexing="ij")
    c = (n - 1) / 2
    phase = math.pi * fmax * ((xx - c) ** 2 + (yy - c) ** 2) / (math.sqrt(2) * c)
    plate = ((0.5 + 0.5 * torch.cos(phase)) * 255).round() / 255   # exact in 8 bits
    plate = plate.float().unsqueeze(0)
    plain = resize(plate, (64, 64), backend="torch")
    filtered = resize(plate, (64, 64), backend="torch", antialias=True)
    pil = resize(plate, (64, 64), backend="pil", resample="bilinear")
    near_pil = float((filtered - pil).abs().max()) * 255
    far = float((plain - pil).abs().max()) * 255
    assert near_pil <= 1.0 and far > 90
    print(f"    measured: antialias {near_pil:.2f}/255 from PIL, plain {far:.1f}/255")

    fig = Figure("7-antialias", M + 3 * PANEL + 2 * GAP + M, TOP + PANEL + M)
    fig.grey = True
    fig.panel(plate, (M, TOP), 1, "zone plate, (1, 256, 256)")
    fig.panel(plain, (M + PANEL + GAP, TOP), 4, "resize to 64, torch")
    fig.panel(filtered, (M + 2 * (PANEL + GAP), TOP), 4, "the same, antialias=True")
    fig.claim("1/255")
    fig.claim(f"{int(far // 10 * 10)}/255")
    return fig


def build_all() -> list[Figure]:
    torch.set_num_threads(1)                           # reductions independent of the core count
    img = outreach.test_image()
    builders = [fig_silent, fig_patch_order, fig_count_map, fig_votes, fig_where_it_misses,
                fig_windows, fig_halo, fig_pair]
    figures = []
    for build in builders:
        print(f"  {build.__name__}")
        figures.append(build(img))
    print("  fig_antialias")
    figures.append(fig_antialias())
    return figures


# --------------------------------------------------------------------------
# Writing and checking
# --------------------------------------------------------------------------


def caption(guide: str, file: str) -> str | None:
    """The paragraph that follows the line linking ``figures/<file>``, or None."""
    lines = guide.splitlines()
    for i, line in enumerate(lines):
        if f"](figures/{file})" in line:
            body = []
            for nxt in lines[i + 1:]:
                if body and not nxt.strip():
                    break
                if nxt.strip():
                    body.append(nxt)
            return " ".join(body)
    return None


def drawn_differs(fig: Figure, committed: Path) -> str | None:
    fresh = np.asarray(fig.im.convert("RGB"))
    with Image.open(committed) as im:
        old = np.asarray(im.convert("RGB"))
    if old.shape != fresh.shape:
        return f"size {old.shape[1]}x{old.shape[0]}, fresh render {fresh.shape[1]}x{fresh.shape[0]}"
    if fig.grey:
        fresh = np.asarray(fig.im.convert("L").convert("RGB"))
    outside = ~fig.text_mask()
    differs = int(((old != fresh).any(2) & outside).sum())
    if differs * 1000 > int(outside.sum()):            # more than 0.1 percent, see the docstring
        return f"{differs} pixels outside the labels differ from a fresh render"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="write nothing; exit 1 if the committed figures or captions drifted")
    args = parser.parse_args(argv)

    figures = build_all()
    guide = GUIDE.read_text(encoding="utf-8")
    linked = set(re.findall(r"\]\(figures/([\w.-]+\.png)\)", guide))
    problems = []
    OUT.mkdir(parents=True, exist_ok=True)
    for fig in figures:
        file = f"{fig.name}.png"
        path = OUT / file
        if args.check:
            if not path.exists():
                problems.append(f"{file}: not in docs/figures")
            else:
                drift = drawn_differs(fig, path)
                if drift:
                    problems.append(f"{file}: {drift}")
        else:
            data = fig.png()
            path.write_bytes(data)
            print(f"  wrote {path.relative_to(ROOT)}  {fig.im.width}x{fig.im.height}  "
                  f"{len(data) / 1024:.1f} KB")
        if file not in linked:
            problems.append(f"{file}: not linked from docs/GUIDE.md")
            continue
        text = caption(guide, file) or ""
        for claim in fig.claims:
            if claim not in text:
                problems.append(f"{file}: the caption does not state {claim!r}")
    produced = {f"{f.name}.png" for f in figures}
    for orphan in sorted({p.name for p in OUT.glob("*.png")} - produced):
        problems.append(f"{orphan}: in docs/figures but drawn by nothing")
    for missing in sorted(linked - produced):
        problems.append(f"{missing}: linked from docs/GUIDE.md but drawn by nothing")
    for problem in problems:
        print("PROBLEM", problem)
    if problems and args.check:
        return 1
    print("OK" if not problems else f"{len(problems)} problem(s); the figures were written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
