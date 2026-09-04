"""Draw the two coverage figures used by the outreach texts.

The first figure shows what the geometry does: how many patches cover each
pixel at three different steps, and that one of the three leaves part of the
image uncovered. The second shows what PatchCraft returns for those same three
geometries, including the error it raises rather than handing back an image
with a black band in it.

Nothing here is drawn by hand. Every panel is the coverage map that torch's
own fold and unfold produce over a tensor of ones, the error text is the
message the library actually raises, and the list of usable steps is the one
``patchcraft.tilings`` returns. Re-running the script regenerates all of it:

    python tools/make_outreach_figure.py

Writes a PNG for uploading and an SVG of the same drawing for later editing,
one pair per figure per language, into outreach/linkedin/.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "outreach" / "linkedin"

W, H = 1200, 848
BG = (255, 255, 255)
INK = (26, 26, 26)
MUTED = (110, 110, 110)
RULE = (214, 214, 214)
LOSS = (176, 42, 42)
GOOD = (28, 108, 76)
BAND = (246, 247, 249)

# One colour per coverage count. The powers of two share a blue ramp; zero is
# red because it is the only count that loses data.
COUNT_COLOUR = {
    0: (176, 42, 42),
    1: (219, 233, 246),
    2: (141, 184, 219),
    4: (52, 105, 156),
}

PANEL = 300
GAP = 45
LEFT = (W - 3 * PANEL - 2 * GAP) // 2
TOP = 236

IMAGE_SIZE = 128
PATCH = 32
STRIDES = (32, 16, 20)

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


def coverage_map(size: int, patch: int, stride: int) -> torch.Tensor:
    """How many patches cover each pixel, zero where the grid does not reach."""
    n = (size - patch) // stride + 1
    end = (n - 1) * stride + patch
    ones = torch.ones(1, 1, size, size)
    cols = F.unfold(ones[:, :, :end, :end], kernel_size=patch, stride=stride)
    folded = F.fold(cols, output_size=(end, end), kernel_size=patch, stride=stride)
    full = torch.zeros(size, size)
    full[:end, :end] = folded[0, 0]
    return full


def runs(values: list) -> list[tuple[int, int]]:
    """Boundaries of each maximal constant run in a sequence."""
    out: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            out.append((start, i))
            start = i
    return out


class Canvas:
    """Writes each element into the raster and the vector output at once."""

    def __init__(self) -> None:
        self.img = Image.new("RGB", (W, H), BG)
        self.d = ImageDraw.Draw(self.img)
        self.svg = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
            f'viewBox="0 0 {W} {H}">',
            f'<rect width="{W}" height="{H}" fill="rgb{BG}"/>',
        ]

    def text(self, xy, string, size, colour, bold=False, mono=False):
        x, y = xy
        self.d.text((x, y), string, font=font(size, bold, mono), fill=colour)
        weight = ' font-weight="bold"' if bold else ""
        fam = MONO_FAMILY if mono else FAMILY
        self.svg.append(
            f'<text x="{x}" y="{y + size}" {fam} font-size="{size}"{weight} '
            f"fill=\"rgb{colour}\">{string.replace('&', '&amp;').replace('<', '&lt;')}</text>"
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

    def fitted_title(self, string: str) -> None:
        """The English titles are longer, so the size is fitted to the column."""
        size = 44
        while size > 24 and font(size, bold=True).getlength(string) > W - 2 * LEFT:
            size -= 1
        self.text((LEFT, 54), string, size, INK, bold=True)

    def panels(self, details, verdicts, oks, labels):
        scale = PANEL / IMAGE_SIZE
        for idx, stride in enumerate(STRIDES):
            rows = coverage_map(IMAGE_SIZE, PATCH, stride).tolist()
            x0 = LEFT + idx * (PANEL + GAP)
            for ya, yb in runs([tuple(r) for r in rows]):
                for xa, xb in runs(rows[ya]):
                    colour = COUNT_COLOUR[int(rows[ya][xa])]
                    self.box(
                        (x0 + xa * scale, TOP + ya * scale),
                        (x0 + xb * scale, TOP + yb * scale),
                        fill=colour,
                    )
            self.box((x0, TOP), (x0 + PANEL, TOP + PANEL), outline=RULE)
            self.text((x0, TOP - 44), labels[idx], 27, INK, bold=True)
            self.text((x0, TOP + PANEL + 18), details[idx], 18, MUTED)
            self.text(
                (x0, TOP + PANEL + 46), verdicts[idx], 20, GOOD if oks[idx] else LOSS, bold=True
            )

    def save(self, stem: str, lang: str) -> None:
        self.svg.append("</svg>")
        png = OUT / f"{stem}.{lang}.png"
        svg = OUT / f"{stem}.{lang}.svg"
        self.img.save(png)
        svg.write_text("\n".join(self.svg), encoding="utf-8")
        root = OUT.parent.parent
        print(f"  {png.relative_to(root)}  {png.stat().st_size // 1024} KB")
        print(f"  {svg.relative_to(root)}  {svg.stat().st_size // 1024} KB")


STRINGS = {
    "pt-BR": {
        "labels": ("passo 32", "passo 16", "passo 20"),
        "foot": "Reproduz com: python tools/make_outreach_figure.py",
        "problem": {
            "stem": "cobertura",
            "title": "O passo decide se a imagem volta inteira",
            "sub1": (
                f"Imagem {IMAGE_SIZE}x{IMAGE_SIZE}, patch {PATCH}. A cor é o número de "
                "patches que cobrem aquele pixel,"
            ),
            "sub2": "medido com o fold e o unfold do torch sobre um tensor de uns.",
            "details": (
                "4x4 patches, sem sobreposição",
                "7x7 patches, com sobreposição",
                "5x5 patches, a grade termina no pixel 112",
            ),
            "verdicts": (
                "todo pixel coberto 1 vez",
                "contagens de 1, 2 e 4",
                "3840 pixels voltam pretos",
            ),
            "legend": "patches cobrindo o pixel:",
            "zero": "0, perdido",
        },
        "solution": {
            "stem": "patchcraft",
            "title": "O PatchCraft recusa em vez de devolver a imagem errada",
            "sub1": (
                "As mesmas três geometrias, agora com o que a biblioteca devolve em cada "
                "uma. O reconstruct confere"
            ),
            "sub2": (
                "a cobertura antes de somar, então a falha chega como erro "
                "e não como pixel preto."
            ),
            "details": (
                "passo 32, sem sobreposição",
                "passo 16, contagens de 1, 2 e 4",
                "passo 20, cobre 112 de 128",
            ),
            "verdicts": (
                "volta bit a bit idêntica",
                "volta bit a bit idêntica",
                "ValueError, não devolve imagem",
            ),
            "band": (
                "ValueError: patch grid leaves pixels uncovered (partial coverage forbidden):",
                "stride=(20, 20) covers (112, 112) of (128, 128).",
                "Choose a geometry with exact coverage (see patchcraft.tilings).",
            ),
            "answer_label": "E a resposta vem junto.",
            "answer": (
                "tilings((3, 128, 128)) lista 517 geometrias legais. Com patch 32, "
                "os passos que cobrem exato:"
            ),
            "answer2": "1, 2, 3, 4, 6, 8, 12, 16, 24 e 32.",
        },
    },
    "en": {
        "labels": ("stride 32", "stride 16", "stride 20"),
        "foot": "Reproduce with: python tools/make_outreach_figure.py",
        "problem": {
            "stem": "coverage",
            "title": "The step decides whether the image comes back whole",
            "sub1": (
                f"{IMAGE_SIZE}x{IMAGE_SIZE} image, patch {PATCH}. The colour is the number "
                "of patches covering that pixel,"
            ),
            "sub2": "measured with torch's own fold and unfold over a tensor of ones.",
            "details": (
                "4x4 patches, no overlap",
                "7x7 patches, overlapping",
                "5x5 patches, grid stops at pixel 112",
            ),
            "verdicts": (
                "every pixel covered once",
                "counts of 1, 2 and 4",
                "3840 pixels come back black",
            ),
            "legend": "patches covering the pixel:",
            "zero": "0, lost",
        },
        "solution": {
            "stem": "patchcraft",
            "title": "PatchCraft refuses instead of returning the wrong image",
            "sub1": (
                "The same three geometries, now with what the library returns for each. "
                "reconstruct checks the"
            ),
            "sub2": (
                "coverage before summing, so the failure arrives as an error "
                "rather than as black pixels."
            ),
            "details": (
                "stride 32, no overlap",
                "stride 16, counts of 1, 2 and 4",
                "stride 20, covers 112 of 128",
            ),
            "verdicts": (
                "comes back bit for bit",
                "comes back bit for bit",
                "ValueError, no image returned",
            ),
            "band": (
                "ValueError: patch grid leaves pixels uncovered (partial coverage forbidden):",
                "stride=(20, 20) covers (112, 112) of (128, 128).",
                "Choose a geometry with exact coverage (see patchcraft.tilings).",
            ),
            "answer_label": "And the answer comes with it.",
            "answer": (
                "tilings((3, 128, 128)) lists 517 legal geometries. With patch 32, "
                "the strides that cover exactly:"
            ),
            "answer2": "1, 2, 3, 4, 6, 8, 12, 16, 24 and 32.",
        },
    },
}


def build_problem(lang: str) -> None:
    top = STRINGS[lang]
    s = top["problem"]
    c = Canvas()
    c.fitted_title(s["title"])
    c.text((LEFT, 126), s["sub1"], 21, MUTED)
    c.text((LEFT, 156), s["sub2"], 21, MUTED)
    c.panels(s["details"], s["verdicts"], (True, True, False), top["labels"])

    ly = TOP + PANEL + 120
    c.text((LEFT, ly), s["legend"], 19, MUTED)
    lx = LEFT + int(font(19).getlength(s["legend"])) + 26
    for count in (1, 2, 4, 0):
        caption = s["zero"] if count == 0 else str(count)
        c.box((lx, ly), (lx + 21, ly + 21), fill=COUNT_COLOUR[count])
        c.text((lx + 29, ly + 1), caption, 19, INK)
        lx += 29 + int(font(19).getlength(caption)) + 32

    c.text((LEFT, H - 50), top["foot"], 17, MUTED)
    c.save(s["stem"], lang)


def build_solution(lang: str) -> None:
    top = STRINGS[lang]
    s = top["solution"]
    c = Canvas()
    c.fitted_title(s["title"])
    c.text((LEFT, 126), s["sub1"], 21, MUTED)
    c.text((LEFT, 156), s["sub2"], 21, MUTED)
    c.panels(s["details"], s["verdicts"], (True, True, False), top["labels"])

    by0 = TOP + PANEL + 104
    by1 = by0 + 158
    c.box((LEFT, by0), (LEFT + 3 * PANEL + 2 * GAP, by1), fill=BAND)
    c.box((LEFT, by0), (LEFT + 5, by1), fill=LOSS)

    y = by0 + 18
    for i, line in enumerate(s["band"]):
        c.text((LEFT + 24, y), line, 17, LOSS if i == 0 else INK, mono=True)
        y += 24

    y += 8
    c.text((LEFT + 24, y), s["answer_label"], 17, MUTED, bold=True)
    c.text((LEFT + 24 + int(font(17, bold=True).getlength(s["answer_label"])) + 10, y),
           s["answer"], 17, MUTED)
    c.text((LEFT + 24, y + 24), s["answer2"], 17, INK, bold=True, mono=True)

    c.text((LEFT, H - 34), top["foot"], 17, MUTED)
    c.save(s["stem"], lang)


if __name__ == "__main__":
    for language in STRINGS:
        build_problem(language)
        build_solution(language)
