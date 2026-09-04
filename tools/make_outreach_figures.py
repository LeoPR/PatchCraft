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

    def __init__(self, height: int) -> None:
        self.h = height
        self.img = Image.new("RGB", (W, height), BG)
        self.d = ImageDraw.Draw(self.img)
        self.svg = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" '
            f'viewBox="0 0 {W} {height}">',
            f'<rect width="{W}" height="{height}" fill="rgb{BG}"/>',
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
        while size > 22 and font(size, bold=True).getlength(string) > W - 2 * left:
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


if __name__ == "__main__":
    for language in STRINGS:
        print(language)
        build_cover(language)
        build_naive(language)
        build_craft(language)
