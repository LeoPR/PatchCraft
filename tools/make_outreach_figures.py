"""Draw the outreach figures, from measurement rather than by hand.

Three figures, in two languages, into outreach/linkedin/figuras/<lang>/:

  cobertura    the mechanism. How many patches cover each pixel at three
               steps, which is why the other two figures look the way they do.
  fold-unfold  what torch's fold and unfold give you on their own, including
               the two results that are wrong without saying so.
  patchcraft   the same operation through the library, across the three
               regimes it actually has: identical, approximate, refused.

Every panel is computed here. The images are the real tensors, the errors are
measured with torch, the error map is the real difference amplified, and the
refusal carries the message the library actually raises. Re-running the script
regenerates all of it:

    python tools/make_outreach_figures.py
"""

from __future__ import annotations

import base64
import io
import math
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)
from PIL import Image, ImageDraw, ImageFont

from patchcraft import extract, reconstruct

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outreach" / "linkedin" / "figuras"

W = 1200
BG = (255, 255, 255)
INK = (26, 26, 26)
MUTED = (110, 110, 110)
RULE = (206, 206, 206)
LOSS = (176, 42, 42)
GOOD = (28, 108, 76)
WARN = (170, 110, 20)
BAND = (246, 247, 249)

COUNT_COLOUR = {
    0: (176, 42, 42),
    1: (219, 233, 246),
    2: (141, 184, 219),
    4: (52, 105, 156),
}

SIZE = 128
PATCH = 32

FONT_DIRS = (r"C:\Windows\Fonts", "/usr/share/fonts/truetype/dejavu")
FAMILY = 'font-family="Arial,Helvetica,sans-serif"'
MONO_FAMILY = 'font-family="Consolas,Menlo,monospace"'


def font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    if mono:
        names = ("consola.ttf", "DejaVuSansMono.ttf")
    else:
        names = ("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("arial.ttf", "DejaVuSans.ttf")
    for directory in FONT_DIRS:
        for name in names:
            try:
                return ImageFont.truetype(str(Path(directory) / name), size)
            except OSError:
                continue
    return ImageFont.load_default(size)


# --------------------------------------------------------------------------
# The subject
# --------------------------------------------------------------------------


def test_image() -> torch.Tensor:
    """A deterministic image with structure at several scales.

    Smooth gradients show a wrong reshape as a break in the ramp, the rings
    and bars show it as a discontinuity, and the fine diagonal texture is what
    makes a scrambled patch obvious rather than merely noisy.
    """
    y, x = torch.meshgrid(torch.arange(SIZE), torch.arange(SIZE), indexing="ij")
    fx, fy = x / (SIZE - 1), y / (SIZE - 1)

    r = 0.18 + 0.72 * fx
    g = 0.30 + 0.55 * (1 - fy)
    b = 0.45 + 0.45 * ((fx + fy) / 2)

    cx, cy = 0.38 * SIZE, 0.42 * SIZE
    d = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    rings = 0.5 + 0.5 * torch.cos(d / 5.5)
    mask = (d < 46).float()
    r = r * (1 - 0.55 * mask) + rings * 0.55 * mask
    g = g * (1 - 0.35 * mask) + (1 - rings) * 0.35 * mask

    stripes = ((x + y) % 12 < 6).float()
    corner = ((x > 84) & (y > 84)).float()
    b = b * (1 - corner) + stripes * corner

    bar = ((y > 20) & (y < 30) & (x > 14) & (x < 114)).float()
    r = r * (1 - bar) + 0.97 * bar
    g = g * (1 - bar) + 0.92 * bar
    b = b * (1 - bar) + 0.15 * bar

    return torch.stack([r, g, b]).clamp(0, 1).float()


def to_pil(t: torch.Tensor, box: int) -> Image.Image:
    arr = (t.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(arr).resize((box, box), Image.NEAREST)


# --------------------------------------------------------------------------
# The measurements
# --------------------------------------------------------------------------


def coverage_map(stride: int, patch: int = PATCH, size: int = SIZE) -> torch.Tensor:
    n = (size - patch) // stride + 1
    end = (n - 1) * stride + patch
    ones = torch.ones(1, 1, end, end)
    folded = F.fold(F.unfold(ones, patch, stride=stride), (end, end), patch, stride=stride)
    full = torch.zeros(size, size)
    full[:end, :end] = folded[0, 0]
    return full



def count_colour(v: int, vmax: int) -> tuple[int, int, int]:
    """Blue for a power-of-two coverage count, amber for anything else.

    The split is the contract drawn in colour: the round trip is exact
    exactly when every count is a power of two, so a panel with amber in it
    is a panel that cannot come back bit for bit.
    """
    if v == 0:
        return LOSS
    t = math.log2(max(v, 1)) / max(math.log2(max(vmax, 2)), 1.0)
    if v & (v - 1) == 0:
        a, b = (223, 236, 248), (30, 80, 130)
    else:
        a, b = (250, 226, 180), (170, 105, 15)
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))  # type: ignore[return-value]


def runs(values: list) -> list[tuple[int, int]]:
    """Boundaries of each maximal constant run, so the map draws as few rects."""
    out: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            out.append((start, i))
            start = i
    return out


def naive_reshape(img: torch.Tensor) -> torch.Tensor:
    """The reshape that keeps the shape and loses the pixels."""
    cols = F.unfold(img[None], PATCH, stride=PATCH)
    wrong = cols.reshape(16, 3, PATCH, PATCH)
    return wrong.reshape(4, 4, 3, PATCH, PATCH).permute(2, 0, 3, 1, 4).reshape(3, SIZE, SIZE)


def hand_fold(img: torch.Tensor, stride: int) -> torch.Tensor:
    """A fold written by hand, which returns the uncovered part as zero."""
    n = (SIZE - PATCH) // stride + 1
    end = (n - 1) * stride + PATCH
    cols = F.unfold(img[None, :, :end, :end], PATCH, stride=stride)
    num = F.fold(cols, (end, end), PATCH, stride=stride)
    ones = F.unfold(torch.ones(1, 1, end, end), PATCH, stride=stride)
    den = F.fold(ones, (end, end), PATCH, stride=stride)
    out = torch.zeros(3, SIZE, SIZE)
    out[:, :end, :end] = (num / den)[0]
    return out


def error_map(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Where the two differ, scaled to its own maximum so it is visible."""
    diff = (a - b).abs().amax(0)
    peak = float(diff.max())
    if peak == 0:
        return torch.zeros(3, SIZE, SIZE), 0.0
    norm = (diff / peak).clamp(0, 1)
    heat = torch.stack([norm * 0.85 + 0.08, norm * 0.25 + 0.08, 1 - norm * 0.75])
    return heat.clamp(0, 1), peak


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------


class Canvas:
    """Writes each element into the raster and the vector output at once."""

    def __init__(self, height: int, width: int = W) -> None:
        self.h = height
        self.w = width
        self.img = Image.new("RGB", (width, height), BG)
        self.d = ImageDraw.Draw(self.img)
        self.svg = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            f'<rect width="{width}" height="{height}" fill="rgb{BG}"/>',
        ]

    def text(self, xy, string, size, colour, bold=False, mono=False):
        x, y = xy
        self.d.text((x, y), string, font=font(size, bold, mono), fill=colour)
        weight = ' font-weight="bold"' if bold else ""
        fam = MONO_FAMILY if mono else FAMILY
        safe = string.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.svg.append(
            f'<text x="{x}" y="{y + size}" {fam} font-size="{size}"{weight} '
            f'fill="rgb{colour}">{safe}</text>'
        )

    def box(self, xy0, xy1, fill=None, outline=None):
        x0, y0 = xy0
        x1, y1 = xy1
        self.d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=fill, outline=outline)
        attrs = f'fill="rgb{fill}"' if fill else 'fill="none"'
        if outline:
            attrs += f' stroke="rgb{outline}"'
        self.svg.append(
            f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{x1 - x0:.1f}" '
            f'height="{y1 - y0:.1f}" {attrs}/>'
        )

    def paste(self, tensor: torch.Tensor, xy, box_size: int):
        pil = to_pil(tensor, box_size)
        self.img.paste(pil, xy)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        data = base64.b64encode(buf.getvalue()).decode("ascii")
        self.svg.append(
            f'<image x="{xy[0]}" y="{xy[1]}" width="{box_size}" height="{box_size}" '
            f'href="data:image/png;base64,{data}"/>'
        )

    def title(self, string: str, left: int) -> None:
        size = 42
        while size > 22 and font(size, bold=True).getlength(string) > self.w - 2 * left:
            size -= 1
        self.text((left, 46), string, size, INK, bold=True)

    def save(self, lang: str, stem: str) -> None:
        self.svg.append("</svg>")
        folder = OUT / lang
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{stem}.png").parent.mkdir(parents=True, exist_ok=True)
        self.img.save(folder / f"{stem}.png")
        (folder / f"{stem}.svg").write_text("\n".join(self.svg), encoding="utf-8")
        png_kb = (folder / f"{stem}.png").stat().st_size // 1024
        svg_kb = (folder / f"{stem}.svg").stat().st_size // 1024
        rel = (folder / stem).relative_to(ROOT)
        print(f"  {rel}.png  {png_kb} KB      {rel}.svg  {svg_kb} KB")


def panel_grid(count: int, panel: int, gap: int) -> tuple[int, list[int]]:
    total = count * panel + (count - 1) * gap
    left = (W - total) // 2
    return left, [left + i * (panel + gap) for i in range(count)]


def draw_panels(c: Canvas, xs, top, panel, labels, details, verdicts, colours):
    for i, x in enumerate(xs):
        c.text((x, top - 34), labels[i], 22, INK, bold=True)
        c.text((x, top + panel + 14), details[i], 16, MUTED)
        c.text((x, top + panel + 38), verdicts[i], 18, colours[i], bold=True)


# --------------------------------------------------------------------------
# Wording
# --------------------------------------------------------------------------

STRINGS = {
    "pt-BR": {
        "foot": "Tudo medido. Reproduz com: python tools/make_outreach_figures.py",
        "naive": {
            "stem": "fold-unfold",
            "title": "O que o fold e o unfold devolvem sozinhos",
            "sub1": (
                "A mesma imagem 128x128 com patch 32, recortada e remontada "
                "de quatro maneiras."
            ),
            "sub2": "Duas delas devolvem a imagem errada sem levantar erro nenhum.",
            "labels": ("original", "reshape intuitivo", "passo 20", "passo 32"),
            "details": (
                "a imagem de entrada",
                "unfold, reshape para (L,C,ph,pw)",
                "fold escrito à mão",
                "unfold e fold, feito certo",
            ),
            "verdicts": (
                "o que deveria voltar",
                "erro máximo 0,996",
                "3840 pixels em zero",
                "idêntica, erro 0",
            ),
        },
        "craft": {
            "stem": "patchcraft",
            "title": "O PatchCraft tem três respostas, e diz qual delas deu",
            "sub1": (
                "A mesma imagem por extract e reconstruct, variando só o "
                "passo. O regime muda com a"
            ),
            "sub2": (
                "geometria: exato quando toda contagem de cobertura é potência "
                "de dois, e não por sorte."
            ),
            "labels": ("passo 32", "passo 16", "passo 12", "passo 20"),
            "details": (
                "contagens: 1",
                "contagens: 1, 2, 4",
                "contagens: 1, 2, 3, 4, 6, 9",
                "cobre 112 de 128",
            ),
            "note": "canto: o erro real, ampliado até ficar visível",
            "refusal": (
                "ValueError",
                "patch grid leaves",
                "pixels uncovered",
                "(partial coverage",
                "forbidden)",
            ),
        },
        "cover": {
            "stem": "cobertura",
            "title": "O passo decide se a imagem volta inteira",
            "sub1": (
                "Imagem 128x128, patch 32. A cor é o número de patches que "
                "cobrem aquele pixel,"
            ),
            "sub2": "medido com o fold e o unfold do torch sobre um tensor de uns.",
            "labels": ("passo 32", "passo 16", "passo 20"),
            "details": (
                "4x4 patches, sem sobreposição",
                "7x7 patches, com sobreposição",
                "5x5 patches, grade termina em 112",
            ),
            "verdicts": (
                "todo pixel coberto 1 vez",
                "contagens de 1, 2 e 4",
                "3840 pixels sem cobertura",
            ),
            "legend": "patches cobrindo o pixel:",
            "zero": "0, perdido",
        },
    },
    "en": {
        "foot": "All measured. Reproduce with: python tools/make_outreach_figures.py",
        "naive": {
            "stem": "fold-unfold",
            "title": "What fold and unfold return on their own",
            "sub1": "The same 128x128 image with patch 32, cut and reassembled four ways.",
            "sub2": "Two of them return the wrong image without raising anything.",
            "labels": ("original", "intuitive reshape", "stride 20", "stride 32"),
            "details": (
                "the input image",
                "unfold, reshape to (L,C,ph,pw)",
                "fold written by hand",
                "unfold and fold, done right",
            ),
            "verdicts": (
                "what should come back",
                "max error 0.996",
                "3840 pixels at zero",
                "identical, error 0",
            ),
        },
        "craft": {
            "stem": "patchcraft",
            "title": "PatchCraft has three answers, and says which one you got",
            "sub1": (
                "The same image through extract and reconstruct, varying only "
                "the stride. The regime"
            ),
            "sub2": (
                "follows the geometry: exact when every coverage count is a "
                "power of two, not by luck."
            ),
            "labels": ("stride 32", "stride 16", "stride 12", "stride 20"),
            "details": (
                "counts: 1",
                "counts: 1, 2, 4",
                "counts: 1, 2, 3, 4, 6, 9",
                "covers 112 of 128",
            ),
            "note": "corner: the real error, amplified until visible",
            "refusal": (
                "ValueError",
                "patch grid leaves",
                "pixels uncovered",
                "(partial coverage",
                "forbidden)",
            ),
        },
        "cover": {
            "stem": "coverage",
            "title": "The step decides whether the image comes back whole",
            "sub1": (
                "128x128 image, patch 32. The colour is the number of "
                "patches covering that pixel,"
            ),
            "sub2": "measured with torch's own fold and unfold over a tensor of ones.",
            "labels": ("stride 32", "stride 16", "stride 20"),
            "details": (
                "4x4 patches, no overlap",
                "7x7 patches, overlapping",
                "5x5 patches, grid ends at 112",
            ),
            "verdicts": (
                "every pixel covered once",
                "counts of 1, 2 and 4",
                "3840 pixels uncovered",
            ),
            "legend": "patches covering the pixel:",
            "zero": "0, lost",
        },
    },
}


# --------------------------------------------------------------------------
# The figures
# --------------------------------------------------------------------------

PANEL4, GAP4, TOP4, H4 = 250, 30, 218, 588
PANEL3, GAP3, TOP3, H3 = 300, 45, 218, 664


def build_naive(lang: str) -> None:
    s = STRINGS[lang]["naive"]
    img = test_image()
    panels = [img, naive_reshape(img), hand_fold(img, 20), hand_fold(img, 32)]

    left, xs = panel_grid(4, PANEL4, GAP4)
    c = Canvas(H4)
    c.title(s["title"], left)
    c.text((left, 118), s["sub1"], 20, MUTED)
    c.text((left, 146), s["sub2"], 20, MUTED)

    for x, panel in zip(xs, panels, strict=True):
        c.paste(panel, (x, TOP4), PANEL4)
        c.box((x, TOP4), (x + PANEL4, TOP4 + PANEL4), outline=RULE)

    draw_panels(
        c, xs, TOP4, PANEL4, s["labels"], s["details"], s["verdicts"],
        (MUTED, LOSS, LOSS, GOOD),
    )
    c.text((left, H4 - 40), STRINGS[lang]["foot"], 16, MUTED)
    c.save(lang, s["stem"])


def build_craft(lang: str) -> None:
    s = STRINGS[lang]["craft"]
    img = test_image()
    left, xs = panel_grid(4, PANEL4, GAP4)
    c = Canvas(H4)
    c.title(s["title"], left)
    c.text((left, 118), s["sub1"], 20, MUTED)
    c.text((left, 146), s["sub2"], 20, MUTED)

    verdicts, colours = [], []
    for i, stride in enumerate((32, 16, 12, 20)):
        x = xs[i]
        try:
            back = reconstruct(
                extract(img, patch_size=PATCH, stride=stride),
                image_shape=(3, SIZE, SIZE),
                stride=stride,
            )
        except ValueError:
            c.box((x, TOP4), (x + PANEL4, TOP4 + PANEL4), fill=BAND, outline=RULE)
            c.box((x, TOP4), (x + 5, TOP4 + PANEL4), fill=LOSS)
            ty = TOP4 + 78
            for j, line in enumerate(s["refusal"]):
                c.text((x + 22, ty), line, 17, LOSS if j == 0 else INK, mono=True)
                ty += 23
            verdicts.append("recusa" if lang == "pt-BR" else "refused")
            colours.append(LOSS)
            continue

        c.paste(back, (x, TOP4), PANEL4)
        c.box((x, TOP4), (x + PANEL4, TOP4 + PANEL4), outline=RULE)
        heat, peak = error_map(back, img)
        if peak == 0:
            verdicts.append("idêntica, erro 0" if lang == "pt-BR" else "identical, error 0")
            colours.append(GOOD)
        else:
            inset = 88
            ix, iy = x + PANEL4 - inset - 8, TOP4 + PANEL4 - inset - 8
            c.paste(heat, (ix, iy), inset)
            c.box((ix, iy), (ix + inset, iy + inset), outline=(255, 255, 255))
            exponent = math.floor(math.log10(peak))
            mantissa = peak / (10**exponent)
            word = "aproximada" if lang == "pt-BR" else "approximate"
            verdicts.append(f"{word}, {mantissa:.1f}e{exponent}")
            colours.append(WARN)

    draw_panels(c, xs, TOP4, PANEL4, s["labels"], s["details"], verdicts, colours)
    c.text((left, H4 - 40), s["note"] + ".  " + STRINGS[lang]["foot"], 16, MUTED)
    c.save(lang, s["stem"])


def build_cover(lang: str) -> None:
    s = STRINGS[lang]["cover"]
    left, xs = panel_grid(3, PANEL3, GAP3)
    c = Canvas(H3)
    c.title(s["title"], left)
    c.text((left, 118), s["sub1"], 20, MUTED)
    c.text((left, 146), s["sub2"], 20, MUTED)

    scale = PANEL3 / SIZE
    for x, stride in zip(xs, (32, 16, 20), strict=True):
        rows = coverage_map(stride).tolist()
        for ya, yb in runs([tuple(r) for r in rows]):
            for xa, xb in runs(rows[ya]):
                c.box(
                    (x + xa * scale, TOP3 + ya * scale),
                    (x + xb * scale, TOP3 + yb * scale),
                    fill=COUNT_COLOUR[int(rows[ya][xa])],
                )
        c.box((x, TOP3), (x + PANEL3, TOP3 + PANEL3), outline=RULE)

    draw_panels(
        c, xs, TOP3, PANEL3, s["labels"], s["details"], s["verdicts"], (GOOD, GOOD, LOSS)
    )

    ly = TOP3 + PANEL3 + 78
    c.text((left, ly), s["legend"], 18, MUTED)
    lx = left + int(font(18).getlength(s["legend"])) + 24
    for count in (1, 2, 4, 0):
        caption = s["zero"] if count == 0 else str(count)
        c.box((lx, ly), (lx + 20, ly + 20), fill=COUNT_COLOUR[count])
        c.text((lx + 28, ly + 1), caption, 18, INK)
        lx += 28 + int(font(18).getlength(caption)) + 30

    c.text((left, H3 - 40), STRINGS[lang]["foot"], 16, MUTED)
    c.save(lang, s["stem"])


# --------------------------------------------------------------------------
# Three figures and the page that links them
#
# The prose lives in the markdown file, not baked into the pixels, so it can
# be edited and the images can be placed wherever the channel wants them.
# --------------------------------------------------------------------------

STRIDES = (32, 16, 12, 20)


def mnist_digit() -> torch.Tensor | None:
    """One real MNIST digit, downloaded on first use. None if unreachable."""
    try:
        import os

        import numpy as np
        from torchvision import datasets

        root = Path(os.environ.get("PATCHCRAFT_DATASETS", r"Z:\caches\datasets")) / "mnist"
        ds = datasets.MNIST(root=str(root), train=False, download=True)
        arr = np.array(ds[0][0], dtype="float32") / 255.0
        return torch.from_numpy(arr)[None].repeat(3, 1, 1)
    except Exception as exc:
        print(f"  (MNIST unavailable, third figure skipped: {exc})")
        return None


def patch_grid_overlay(gray: torch.Tensor, patch: int) -> torch.Tensor:
    """The digit with the patch boundaries drawn on it."""
    out = gray.clone()
    for k in range(patch, gray.shape[-1], patch):
        for axis in (1, 2):
            sl = [slice(None)] * 3
            sl[axis] = slice(k - 1, k + 1)
            out[(0, *sl[1:])] = 1.0
            out[(1, *sl[1:])] = 0.45
            out[(2, *sl[1:])] = 0.10
    return out


class Fig(Canvas):
    def block(self, xy, lines, size=15, colour=MUTED, lead=20):
        x, y = xy
        for line in lines:
            bold = line.startswith("*")
            self.text((x, y), line.lstrip("*"), size, INK if bold else colour, bold=bold)
            y += lead
        return y

    def framed(self, tensor, xy, side):
        self.paste(tensor, xy, side)
        self.box(xy, (xy[0] + side, xy[1] + side), outline=RULE)


def fig_cut(lang: str) -> None:
    """The unfold, and the reshape that keeps the shape and moves the pixels."""
    s = FIG[lang]
    img = test_image()
    bad = naive_reshape(img)
    # Measured here rather than written down. The reshape is a permutation, so
    # the number that means something is how much of the data lands elsewhere,
    # not how far any one value moves. Counted per channel value.
    moved = float((bad != img).float().mean()) * 100
    side, gap, m = 390, 30, 48
    c = Fig(494, 2 * side + gap + 2 * m)
    caps = (s["cut_cap"][0], s["cut_cap"][1].format(f"{moved:.1f}".replace(".", s["decimal"])))
    for i, (t, cap) in enumerate([(img, caps[0]), (bad, caps[1])]):
        x = m + i * (side + gap)
        c.text((x, 12), s["cut_lab"][i], 21, INK, bold=True)
        c.framed(t, (x, 48), side)
        c.text((x, 48 + side + 14), cap, 17, LOSS if i else MUTED, bold=bool(i))
    c.save(lang, s["cut_stem"])


def fig_stride(lang: str) -> None:
    """Coverage, the hand-written fold, and PatchCraft, for four strides.

    The row descriptions live in the prose, not here: keeping them out buys
    about a fifth of the width back for the panels, which is what decides
    whether the labels survive being scaled down in a feed.
    """
    s = FIG[lang]
    img = test_image()
    m, panel, gap = 22, 225, 18
    xs = [m + i * (panel + gap) for i in range(4)]
    c = Fig(788, 1000)

    for x, st in zip(xs, STRIDES, strict=True):
        c.text((x, 10), s["stride_rest"].format(st), 21, INK, bold=True)

    def row_label(y: int, key: str) -> None:
        head, rest = s[key]
        c.text((m, y), head, 18, INK, bold=True)
        c.text((m + int(font(18, bold=True).getlength(head)) + 8, y), rest, 18, MUTED)

    cov, cov_y = 120, 76
    row_label(46, "row_cov")
    scale = cov / SIZE
    off = (panel - cov) // 2
    for x, st in zip(xs, STRIDES, strict=True):
        rows = coverage_map(st).tolist()
        vmax = int(max(max(r) for r in rows))
        for ya, yb in runs([tuple(r) for r in rows]):
            for xa, xb in runs(rows[ya]):
                c.box(
                    (x + off + xa * scale, cov_y + ya * scale),
                    (x + off + xb * scale, cov_y + yb * scale),
                    fill=count_colour(int(rows[ya][xa]), vmax),
                )
        c.box((x + off, cov_y), (x + off + cov, cov_y + cov), outline=RULE)

    hand_y = 244
    row_label(214, "row_hand")
    for x, st in zip(xs, STRIDES, strict=True):
        c.framed(hand_fold(img, st), (x, hand_y), panel)

    craft_y = 519
    row_label(489, "row_craft")

    # Strides 32 and 16 give the same tensor by both paths, so one box covers
    # both columns instead of the same sentence printed twice.
    span = 2 * panel + gap
    c.box((xs[0], craft_y), (xs[0] + span, craft_y + panel), fill=BAND, outline=RULE)
    c.block((xs[0] + 24, craft_y + 62), s["same"], size=17, colour=MUTED, lead=25)
    c.text((xs[0], craft_y + panel + 12), s["caps"][0], 18, GOOD, bold=True)

    st = STRIDES[2]
    back = reconstruct(
        extract(img, patch_size=PATCH, stride=st), image_shape=(3, SIZE, SIZE), stride=st
    )
    heat, peak = error_map(back, img)
    gain = f"{1.0 / peak / 1e6:.1f}".replace(".", s["decimal"])
    assert torch.equal(
        (img * 255).round().to(torch.uint8), (back * 255).round().to(torch.uint8)
    )
    x = xs[2]
    c.box((x, craft_y), (x + panel, craft_y + panel), fill=BAND, outline=RULE)
    c.text((x + 18, craft_y + 14), s["diff_title"], 16, INK, bold=True)
    inner = 104
    c.framed(heat, (x + (panel - inner) // 2, craft_y + 40), inner)
    c.block(
        (x + 18, craft_y + 156),
        [line.format(gain) for line in s["diff_cap"]],
        size=14,
        lead=18,
    )
    c.text((x, craft_y + panel + 12), s["caps"][1], 18, WARN, bold=True)

    x = xs[3]
    c.box((x, craft_y), (x + panel, craft_y + panel), fill=BAND, outline=RULE)
    c.box((x, craft_y), (x + 5, craft_y + panel), fill=LOSS)
    c.block((x + 22, craft_y + 40), s["refusal"], size=16, colour=INK, lead=24)
    c.text((x, craft_y + panel + 12), s["caps"][2], 18, LOSS, bold=True)

    c.save(lang, s["stride_stem"])


def fig_mnist(lang: str) -> bool:
    """A real digit, its patch grid, and one patch."""
    s = FIG[lang]
    digit = mnist_digit()
    if digit is None:
        return False
    patches = extract(digit, patch_size=7, stride=7)
    side, gap, m = 300, 22, 40
    c = Fig(400, 3 * side + 2 * gap + 2 * m)
    for i, (t, cap) in enumerate(
        zip([digit, patch_grid_overlay(digit, 7), patches[5]], s["mnist_cap"], strict=True)
    ):
        x = m + i * (side + gap)
        c.framed(t, (x, 26), side)
        c.text((x, 26 + side + 14), cap, 17, MUTED)
    c.save(lang, s["mnist_stem"])
    return True


def write_page(lang: str, with_mnist: bool) -> None:
    s = FIG[lang]
    folder = OUT / lang
    body = s["md"].replace("<<MNIST>>", s["md_mnist"] if with_mnist else "")
    path = folder.parent.parent / f"{s['page_stem']}.md"
    path.write_text(body.replace("FIG/", f"figuras/{lang}/"), encoding="utf-8")
    print(f"  {path.relative_to(ROOT)}")


FIG = {
    "pt-BR": {
        "cut_stem": "1-recorte",
        "cover_kicker": "PATCHCRAFT",
        "cover_title": ["Recortar uma imagem", "em patches, e remontar"],
        "cover_sub": "Dois defeitos que não levantam erro, e um contrato que os fecha.",
        "cover_cap": "à direita, o mesmo tensor depois do reshape intuitivo",
        "table_title": "Kernel Rust contra o caminho em torch puro, CPU",
        "table_note": [
            "Medido em docs/PERFORMANCE.md, com máquina, versões e data.",
            "O benchmark compara os dois caminhos com torch.equal antes de cronometrar.",
        ],
        "title_label": "=== TÍTULO (cole no campo Título) ===",
        "body_label": "=== CORPO (cole abaixo) ===",
        "marker": "[IMAGEM: {}.png — legenda: {}]",
        "alt_table": "a tabela de desempenho",
        "placements": [
            (
                "essa imagem parcialmente preta sem reclamar.",
                "1-recorte",
                "a mesma imagem antes e depois do reshape intuitivo",
            ),
            (
                "O contrato acima tem, na suíte, um teste cuja função explícita é falsificá-lo.",
                "2-stride",
                "cobertura, fold à mão e PatchCraft, em quatro strides",
            ),
            (
                "razoável supor que exista um que eu ainda não medi.",
                "3-mnist",
                "um dígito do MNIST, a grade de patches, e um patch",
            ),
        ],
        "howto": """<!-- Versão do artigo preparada para o editor de artigos do LinkedIn.
     Gerada por tools/make_outreach_figures.py a partir de artigo.pt-BR.md.
     Não edite este arquivo: edite o artigo e rode o script.

     O que mudou em relação ao original:
       - as crases saíram, porque o editor não tem código embutido na frase;
       - a tabela virou imagem, porque o editor não faz tabela.

     Como colar:
       1. Capa: figuras/pt-BR/0-capa.png, no quadro do topo (é 1.91:1).
       2. Título: a primeira linha abaixo, no campo Título.
       3. Corpo: cole como texto simples e aplique o formato pelos botões.
          Cada linha "## " vira Estilo -> título. Teste um parágrafo antes de
          colar tudo, para ver se o negrito sobrevive à colagem.
       4. Onde aparecer [IMAGEM: ...], use o botão de imagem, suba o arquivo
          de figuras/pt-BR/ e escreva a legenda indicada. Depois apague a
          linha do marcador.
-->

""",
        "tail": """

<!-- Fim do corpo. Quatro imagens em seis seções, que é o espaçamento que
     deixa o texto respirar. Todas em figuras/pt-BR/, mais a capa 0-capa.png
     para o quadro do topo. -->
""",
        "stride_stem": "2-stride",
        "mnist_stem": "3-mnist",
        "page_stem": "pagina",
        "cut_lab": ("original", "reshape intuitivo"),
        "cut_cap": ("a imagem de entrada", "mesma forma, {}% dos valores fora do lugar"),
        "stride_first": "stride 32",
        "stride_rest": "stride {}",
        "row_cov": (
            "Cobertura",
            "quantos patches cobrem cada pixel. Azul, potência de dois. "
            "Âmbar, não é. Vermelho, nenhum.",
        ),
        "row_hand": (
            "fold e unfold à mão",
            "somar e dividir pela cobertura, sem validar a geometria "
            "e sem dizer o que saiu.",
        ),
        "row_craft": (
            "PatchCraft",
            "a mesma conta, com a geometria conferida antes e o regime "
            "declarado no contrato.",
        ),
        "same": [
            "*Nos dois, o mesmo tensor que a linha de cima.",
            "",
            "A conta é idêntica. O que muda é que aqui o contrato",
            "diz, antes da chamada, que estas duas geometrias voltam",
            "exatas: toda contagem de cobertura é potência de dois.",
        ],
        "diff_title": "o erro, ampliado",
        "decimal": ",",
        "diff_cap": [
            "a diferença real, ampliada",
            "{} milhões de vezes.",
            "Em 8 bits, idênticas.",
        ],
        "refusal": [
            "*ValueError",
            "partial coverage",
            "forbidden.",
            "",
            "Não devolve imagem.",
            "O fold à mão devolve,",
            "e sem avisar.",
        ],
        "caps": ("exatas, erro 0", "aproximada, ≈ 0", "recusada"),
        "mnist_cap": ("o dígito, 28x28", "patch 7, stride 7: 4x4 = 16", "o patch de índice 5"),
        "md_mnist": """
## 3. Numa imagem típica

![Um dígito do MNIST, a grade de patches sobre ele, e um patch isolado](FIG/3-mnist.png)

Um dígito do MNIST tem 28 por 28. Com patch 7 e stride 7, 28 dividido por 7 dá 4 exato, e a
grade cobre o dígito sem sobra e sem sobreposição: toda contagem de cobertura vale 1, e a
volta é bit a bit idêntica.

É o caso mais comum e o único em que não há nada a decidir. Os três problemas acima aparecem
quando o stride deixa de dividir o lado da imagem.
""",
        "md": """<!-- l10n: doc_id=patchcraft-outreach-pagina · lang=pt-BR · canonical -->
**Português** · [English](page.md)

# Recortar uma imagem em patches, processar, e remontar

Página ilustrada para montar em qualquer canal. As imagens ficam em
[`figuras/pt-BR/`](figuras/pt-BR/) e podem ser reordenadas ou usadas soltas. Nada aqui é
desenhado: cada painel é o tensor que aquele caminho devolve, e todas as figuras saem de
`python tools/make_outreach_figures.py`.

Para o computador, uma imagem é uma matriz de números. Uma imagem grande raramente entra
inteira numa rede neural: ela é cortada em pedaços, cada pedaço é processado, e no fim tudo é
colado de volta. Os pedaços se chamam patches, e a distância que a janela anda de um patch
para o próximo é o stride.

## 1. O recorte (unfold) e a remontagem (fold)

![A imagem de entrada e o reshape intuitivo, com os pixels embaralhados](FIG/1-recorte.png)

O `unfold` do PyTorch percorre a imagem com uma janela e devolve todas as janelas empilhadas,
mas não na ordem que parece. Ele achata canal, linha e coluna do patch numa dimensão só, na
forma `(1, C·ph·pw, L)`, e deixa o número de patches no fim.

Ler isso direto como `(L, C, ph, pw)` pede a forma certa, e o tensor tem mesmo essa forma,
então nada reclama. O que muda é a ordem em que os números são lidos do buffer, e o
resultado é o painel da direita.

Vale ser exato sobre o que aconteceu ali, porque a aparência engana nos dois sentidos. **Não
é perda: é uma permutação.** O conjunto de valores é idêntico ao da imagem original, pixel
por pixel, e nada foi destruído nem arredondado. O que se perdeu foi só a correspondência
entre cada valor e a posição dele, e 99,6% dos valores acabam numa posição que não é a sua.
Toda posição de pixel da imagem recebe um valor que não era o dela.

É por isso que o defeito é silencioso e não um acidente barulhento. O tensor continua tendo
a forma certa, o dtype certo e a mesma distribuição de valores, então ele passa em qualquer
verificação de sanidade, o treino roda, e a perda desce um pouco menos.

A volta, o `fold`, soma cada janela no lugar de origem. Onde os patches se sobrepõem ela soma
mais de uma vez, então remontar exige dividir cada pixel pelo número de vezes que ele foi
coberto. Essa contagem é o mapa de cobertura, a primeira linha da figura seguinte.

## 2. O stride decide o resultado

![Cobertura, fold escrito à mão e PatchCraft, para quatro strides](FIG/2-stride.png)

A figura tem três linhas e quatro strides. A de cima é o mapa de cobertura, quantos patches
cobrem cada pixel, com azul onde a contagem é potência de dois e âmbar onde não é. A do meio
é o `fold` e o `unfold` escritos à mão, somando e dividindo pela cobertura, sem validar a
geometria e sem dizer o que saiu. A de baixo é o PatchCraft, com a mesma conta e a geometria
conferida antes.

Nos strides 32 e 16 os dois caminhos devolvem o mesmo tensor, e ele é exato. Vale dizer com
todas as letras: a conta do PatchCraft é a mesma. O que ele acrescenta é conferir a geometria
antes e declarar o regime, não somar diferente.

No stride 12 a volta é aproximada. As contagens de cobertura dele incluem 3, 6 e 9, e dividir
um float por um número que não é potência de dois arredonda. O mapa do erro desenha exatamente
a grade dessas regiões, que são as âmbar da primeira linha.

A figura marca esse caso como `≈ 0` porque é o que ele significa na prática, e aqui vai o
número com a medida ao lado. O erro máximo é `1,1921e-07`, o que dá 151,5 dB de PSNR quando
40 dB já costuma ser tratado como visualmente sem perda. Ele cabe 32.897 vezes dentro de um
degrau de 8 bits, então convertendo as duas imagens para `uint8` elas saem bit a bit
idênticas; só em 16 bits a diferença aparece.

Nada disso torna o erro irrelevante, e é por isso que o contrato o declara: quem soma
milhares de patches, ou encadeia a operação, acumula. O ponto é que "aproximada" aqui
significa abaixo do que qualquer olho ou arquivo de 8 bits registra, e não uma imagem
degradada.

No stride 20 os dois caminhos divergem de verdade. A grade termina no pixel 112 de 128, e o
`fold` escrito à mão devolve uma imagem com 3840 pixels em zero, sem levantar nada. O
`reconstruct` recusa, com uma mensagem que nomeia os números e aponta para `patchcraft.tilings`,
que lista as geometrias que fecham.
<<MNIST>>
## Reprodução

```
pip install patchcraft
python tools/make_outreach_figures.py
```

Repositório: https://github.com/LeoPR/PatchCraft
""",
    },
    "en": {
        "cut_stem": "1-cut",
        "cover_kicker": "PATCHCRAFT",
        "cover_title": ["Cutting an image into", "patches, and putting it back"],
        "cover_sub": "Two defects that raise nothing, and a contract that closes them.",
        "cover_cap": "on the right, the same tensor after the intuitive reshape",
        "table_title": "Rust kernel against the pure torch path, CPU",
        "table_note": [
            "Measured in docs/PERFORMANCE.md, with machine, versions and date.",
            "The benchmark compares both paths with torch.equal before timing anything.",
        ],
        "title_label": "=== TITLE (paste into the Title field) ===",
        "body_label": "=== BODY (paste below) ===",
        "marker": "[IMAGE: {}.png - caption: {}]",
        "alt_table": "the performance table",
        "placements": [
            (
                "that partly black image without complaining.",
                "1-cut",
                "the same image before and after the intuitive reshape",
            ),
            (
                "The contract above has, in the suite, a test whose explicit job is to falsify it.",
                "2-stride",
                "coverage, the hand-written fold and PatchCraft, across four strides",
            ),
            (
                "is reasonable to assume there is one I have not measured yet.",
                "3-mnist",
                "an MNIST digit, the patch grid, and one patch",
            ),
        ],
        "howto": """<!-- Version of the article prepared for the LinkedIn article editor.
     Generated by tools/make_outreach_figures.py from artigo.en.md.
     Do not edit this file: edit the article and run the script.

     What changed against the original:
       - the backticks are gone, because the editor has no inline code;
       - the table became an image, because the editor cannot make tables.

     How to paste it:
       1. Cover: figuras/en/0-capa.png, into the box at the top (it is 1.91:1).
       2. Title: the first line below, into the Title field.
       3. Body: paste as plain text and apply the formatting with the buttons.
          Every "## " line becomes Style -> heading. Test one paragraph before
          pasting the lot, to see whether bold survives the paste.
       4. Wherever [IMAGE: ...] appears, use the image button, upload the file
          from figuras/en/ and write the caption given. Then delete the marker
          line.
-->

""",
        "tail": """

<!-- End of the body. Four images across six sections, which is the spacing
     that lets the text breathe. All in figuras/en/, plus the cover
     0-capa.png for the box at the top. -->
""",
        "stride_stem": "2-stride",
        "mnist_stem": "3-mnist",
        "page_stem": "page",
        "cut_lab": ("original", "intuitive reshape"),
        "cut_cap": ("the input image", "same shape, {}% of the values moved"),
        "stride_first": "stride 32",
        "stride_rest": "stride {}",
        "row_cov": (
            "Coverage",
            "how many patches cover each pixel. Blue, a power of two. "
            "Amber, not one. Red, none at all.",
        ),
        "row_hand": (
            "fold and unfold by hand",
            "sum and divide by the coverage, validating no geometry "
            "and saying nothing about the result.",
        ),
        "row_craft": (
            "PatchCraft",
            "the same arithmetic, with the geometry checked first and the "
            "regime declared in the contract.",
        ),
        "same": [
            "*Both give the same tensor as the row above.",
            "",
            "The arithmetic is identical. What changes is that here",
            "the contract says, before the call, that these two",
            "geometries come back exact: every count is a power of two.",
        ],
        "diff_title": "the error, amplified",
        "decimal": ".",
        "diff_cap": [
            "the real difference,",
            "amplified {} million times.",
            "At 8 bits, identical.",
        ],
        "refusal": [
            "*ValueError",
            "partial coverage",
            "forbidden.",
            "",
            "It returns no image.",
            "The hand-written fold",
            "returns one, silently.",
        ],
        "caps": ("exact, error 0", "approximate, ≈ 0", "refused"),
        "mnist_cap": ("the digit, 28x28", "patch 7, stride 7: 4x4 = 16", "the patch at index 5"),
        "md_mnist": """
## 3. On a typical image

![An MNIST digit, the patch grid over it, and one isolated patch](FIG/3-mnist.png)

An MNIST digit is 28 by 28. At patch 7 and stride 7, 28 divided by 7 is exactly 4, and the
grid covers the digit with nothing left over and no overlap: every coverage count is 1, and
the round trip is bit for bit identical.

It is the commonest case and the only one with nothing to decide. The three problems above
appear when the stride stops dividing the side of the image.
""",
        "md": """<!-- l10n: doc_id=patchcraft-outreach-pagina · lang=en · translation_of=pagina.md · source_lang=pt-BR -->
**English** · [Português](pagina.md)

# Cutting an image into patches, processing, and putting it back

An illustrated page to assemble into any channel. The images live in
[`figuras/en/`](figuras/en/) and can be reordered or used on their own. Nothing here is
drawn: each panel is the tensor that path returns, and every figure comes out of
`python tools/make_outreach_figures.py`.

To a computer an image is a matrix of numbers. A large image rarely goes into a neural
network whole: it is cut into pieces, each piece is processed, and at the end everything is
glued back. The pieces are called patches, and the distance the window travels from one patch
to the next is the stride.

## 1. The cut (unfold) and the reassembly (fold)

![The input image and the intuitive reshape, with the pixels scrambled](FIG/1-cut.png)

PyTorch's `unfold` slides a window across the image and returns every window stacked, but not
in the order it looks. It flattens channel, patch row and patch column into a single
dimension, shaped `(1, C·ph·pw, L)`, and leaves the number of patches at the end.

Reading that directly as `(L, C, ph, pw)` asks for the right shape, and the tensor does have
that shape, so nothing complains. What changes is the order the numbers are read out of the
buffer, and the result is the panel on the right.

It is worth being exact about what happened there, because the look of it misleads in both
directions. **It is not loss: it is a permutation.** The set of values is identical to the
original image, pixel for pixel, and nothing was destroyed or rounded. What was lost is only
the correspondence between each value and its position, and 99.6% of the values end up
somewhere that is not theirs. Every pixel position in the image receives a value that was
not its own.

That is why the defect is silent rather than a loud accident. The tensor still has the right
shape, the right dtype and the same distribution of values, so it passes any sanity check,
training runs, and the loss falls a little less.

The way back, `fold`, adds each window into the place it came from. Where patches overlap it
adds more than once, so putting the image back means dividing each pixel by the number of
times it was covered. That count is the coverage map, the first row of the next figure.

## 2. The stride decides the result

![Coverage, the hand-written fold and PatchCraft, across four strides](FIG/2-stride.png)

The figure has three rows across four strides. The top one is the coverage map, how many
patches cover each pixel, blue where the count is a power of two and amber where it is not.
The middle one is `fold` and `unfold` written by hand, summing and dividing by the coverage,
validating no geometry and saying nothing about the result. The bottom one is PatchCraft,
with the same arithmetic and the geometry checked first.

At strides 32 and 16 both paths return the same tensor, and it is exact. Worth saying plainly:
PatchCraft's arithmetic is the same. What it adds is checking the geometry first and declaring
the regime, not summing differently.

At stride 12 the round trip is approximate. Its coverage counts include 3, 6 and 9, and
dividing a float by anything that is not a power of two rounds. The error map draws exactly
the grid of those regions, which are the amber ones in the first row.

The figure marks that case `≈ 0` because that is what it means in practice, and here is the
number with a measure beside it. The maximum error is `1.1921e-07`, which is 151.5 dB of
PSNR, where 40 dB is already treated as visually lossless. It fits 32,897 times inside one
8-bit step, so converting both images to `uint8` makes them bit for bit identical; the
difference only appears at 16 bits.

None of that makes the error irrelevant, which is why the contract declares it: anyone
summing thousands of patches, or chaining the operation, accumulates. The point is that
"approximate" here means below anything an eye or an 8-bit file records, not a degraded
image.

At stride 20 the two paths genuinely diverge. The grid ends at pixel 112 of 128, and the
hand-written `fold` returns an image with 3840 pixels at zero, raising nothing. `reconstruct`
refuses, with a message that names the numbers and points at `patchcraft.tilings`, which lists
the geometries that close.
<<MNIST>>
## Reproduction

```
pip install patchcraft
python tools/make_outreach_figures.py
```

Repository: https://github.com/LeoPR/PatchCraft
""",
    },
}


# --------------------------------------------------------------------------
# For the LinkedIn article editor, which has no table and no inline code
# --------------------------------------------------------------------------

ARTICLE = {"pt-BR": "artigo.pt-BR.md", "en": "artigo.en.md"}
NL = chr(10)


def article_path(lang: str) -> Path:
    return OUT.parent / ARTICLE[lang]


def read_table(lang: str) -> list[list[str]]:
    """The benchmark table, taken from the article so the two cannot drift."""
    rows = []
    for line in article_path(lang).read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip().replace("`", "") for c in line.strip("|").split("|")]
        if all(set(c) <= {"-", ":"} for c in cells):  # the header separator
            continue
        rows.append(cells)
    return rows


def fig_cover(lang: str) -> None:
    """The 1.91:1 cover the article editor asks for."""
    s = FIG[lang]
    img = test_image()
    c = Fig(628, 1200)
    c.text((64, 150), s["cover_kicker"], 20, WARN, bold=True)
    y = 190
    for line in s["cover_title"]:
        c.text((64, y), line, 44, INK, bold=True)
        y += 54
    c.text((64, y + 16), s["cover_sub"], 21, MUTED)
    side = 200
    c.framed(img, (700, 214), side)
    c.framed(naive_reshape(img), (700 + side + 24, 214), side)
    c.text((700, 214 + side + 14), s["cover_cap"], 15, MUTED)
    c.save(lang, "0-capa")


def fig_table(lang: str) -> None:
    """The benchmark table drawn, since the editor cannot make one."""
    s = FIG[lang]
    rows = read_table(lang)
    cols = [40, 340, 500, 650, 790]
    c = Fig(104 + 46 * len(rows), 900)
    c.text((40, 16), s["table_title"], 22, INK, bold=True)
    y = 62
    for i, row in enumerate(rows):
        if i == 0:
            c.box((40, y - 8), (860, y - 7), fill=RULE)
        for x, cell in zip(cols, row, strict=True):
            bold = i == 0 or x == cols[-1]
            colour = INK if i == 0 else (GOOD if x == cols[-1] else MUTED)
            c.text((x, y), cell, 17, colour, bold=bold)
        y += 46
        c.box((40, y - 14), (860, y - 13), fill=RULE)
    c.block((40, y + 4), s["table_note"], size=14, lead=19)
    c.save(lang, "4-tabela")


def write_article_for_linkedin(lang: str) -> None:
    """The article with the two things the editor cannot render taken out."""
    s = FIG[lang]
    raw = article_path(lang).read_text(encoding="utf-8")
    title = next(ln[2:].strip() for ln in raw.splitlines() if ln.startswith("# "))
    text = raw.split("---" + NL, 1)[1].lstrip()

    out, in_table = [], False
    for line in text.splitlines():
        if line.startswith("|"):
            if not in_table:
                out.append(s["marker"].format("4-tabela", s["alt_table"]))
                in_table = True
            continue
        in_table = False
        out.append(line.replace("`", ""))
    body = NL.join(out)

    for anchor_text, name, alt in s["placements"]:
        assert anchor_text in body, anchor_text[:40]
        body = body.replace(
            anchor_text, anchor_text + NL + NL + s["marker"].format(name, alt), 1
        )

    header = s["title_label"] + NL + NL + title + NL + NL + s["body_label"] + NL + NL
    path = OUT.parent / f"artigo-linkedin.{lang}.md"
    path.write_text(s["howto"] + header + body + s["tail"], encoding="utf-8")
    print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    for language in FIG:
        print(language)
        fig_cover(language)
        fig_cut(language)
        fig_stride(language)
        ok = fig_mnist(language)
        fig_table(language)
        write_page(language, ok)
        write_article_for_linkedin(language)
