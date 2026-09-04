"""Draw the coverage-map figure used by the outreach texts.

Every panel is computed here from torch's own fold/unfold over a tensor of
ones, so the picture reports a measurement rather than illustrating one. Run
it and the figure regenerates identically:

    python tools/make_outreach_figure.py

Writes a PNG for uploading and an SVG of the same drawing for later editing,
one pair per language, into outreach/linkedin/.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "outreach" / "linkedin"

W, H = 1200, 752
BG = (255, 255, 255)
INK = (26, 26, 26)
MUTED = (110, 110, 110)
RULE = (214, 214, 214)
LOSS = (176, 42, 42)

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

FONT_DIRS = (
    r"C:\Windows\Fonts",
    "/usr/share/fonts/truetype/dejavu",
)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = ("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("arial.ttf", "DejaVuSans.ttf")
    for directory in FONT_DIRS:
        for name in names:
            try:
                return ImageFont.truetype(str(Path(directory) / name), size)
            except OSError:
                continue
    return ImageFont.load_default(size)


def coverage_map(size: int, patch: int, stride: int) -> tuple[torch.Tensor, int, int]:
    """How many patches cover each pixel, and where the grid stops."""
    n = (size - patch) // stride + 1
    end = (n - 1) * stride + patch
    ones = torch.ones(1, 1, size, size)
    cols = F.unfold(ones[:, :, :end, :end], kernel_size=patch, stride=stride)
    folded = F.fold(cols, output_size=(end, end), kernel_size=patch, stride=stride)
    full = torch.zeros(size, size)
    full[:end, :end] = folded[0, 0]
    return full, n, end


def runs(values: list) -> list[tuple[int, int]]:
    """Boundaries of each maximal constant run in a sequence."""
    out: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            out.append((start, i))
            start = i
    return out


STRINGS = {
    "pt-BR": {
        "stem": "cobertura",
        "title": "O passo decide se a imagem volta inteira",
        "sub1": (
            f"Imagem {IMAGE_SIZE}x{IMAGE_SIZE}, patch {PATCH}. A cor é o número de patches "
            "que cobrem aquele pixel,"
        ),
        "sub2": "medido com o fold e o unfold do torch sobre um tensor de uns.",
        "panels": [
            ("passo 32", "4x4 patches, sem sobreposição", "todo pixel coberto 1 vez", True),
            ("passo 16", "7x7 patches, com sobreposição", "contagens de 1, 2 e 4", True),
            (
                "passo 20",
                "5x5 patches, a grade termina no pixel 112",
                "3840 pixels voltam pretos",
                False,
            ),
        ],
        "legend": "patches cobrindo o pixel:",
        "zero": "0, perdido",
        "foot": "Reproduz com: python tools/make_outreach_figure.py",
    },
    "en": {
        "stem": "coverage",
        "title": "The step decides whether the image comes back whole",
        "sub1": (
            f"{IMAGE_SIZE}x{IMAGE_SIZE} image, patch {PATCH}. The colour is the number of "
            "patches covering that pixel,"
        ),
        "sub2": "measured with torch's own fold and unfold over a tensor of ones.",
        "panels": [
            ("stride 32", "4x4 patches, no overlap", "every pixel covered once", True),
            ("stride 16", "7x7 patches, overlapping", "counts of 1, 2 and 4", True),
            (
                "stride 20",
                "5x5 patches, grid stops at pixel 112",
                "3840 pixels come back black",
                False,
            ),
        ],
        "legend": "patches covering the pixel:",
        "zero": "0, lost",
        "foot": "Reproduce with: python tools/make_outreach_figure.py",
    },
}

FAMILY = 'font-family="Arial,Helvetica,sans-serif"'


def text_pair(draw, svg, xy, string, size, colour, bold=False):
    """Write the same string into the raster and the vector output."""
    x, y = xy
    draw.text((x, y), string, font=font(size, bold), fill=colour)
    weight = ' font-weight="bold"' if bold else ""
    svg.append(
        f'<text x="{x}" y="{y + size}" {FAMILY} font-size="{size}"{weight} '
        f'fill="rgb{colour}">{string}</text>'
    )


def build(lang: str) -> None:
    s = STRINGS[lang]
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    svg = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="rgb{BG}"/>',
    ]

    # The English title is longer than the Portuguese one, so the size is fitted
    # to the column rather than fixed, and the two figures stay the same shape.
    title_size = 44
    while title_size > 24 and font(title_size, bold=True).getlength(s["title"]) > W - 2 * LEFT:
        title_size -= 1

    text_pair(d, svg, (LEFT, 54), s["title"], title_size, INK, bold=True)
    text_pair(d, svg, (LEFT, 126), s["sub1"], 21, MUTED)
    text_pair(d, svg, (LEFT, 156), s["sub2"], 21, MUTED)

    scale = PANEL / IMAGE_SIZE
    for idx, stride in enumerate(STRIDES):
        cmap, _n, _end = coverage_map(IMAGE_SIZE, PATCH, stride)
        rows = cmap.tolist()
        x0 = LEFT + idx * (PANEL + GAP)

        for ya, yb in runs([tuple(r) for r in rows]):
            for xa, xb in runs(rows[ya]):
                colour = COUNT_COLOUR[int(rows[ya][xa])]
                px0, py0 = x0 + xa * scale, TOP + ya * scale
                px1, py1 = x0 + xb * scale, TOP + yb * scale
                d.rectangle([px0, py0, px1 - 1, py1 - 1], fill=colour)
                svg.append(
                    f'<rect x="{px0:.1f}" y="{py0:.1f}" width="{px1 - px0:.1f}" '
                    f'height="{py1 - py0:.1f}" fill="rgb{colour}"/>'
                )

        d.rectangle([x0, TOP, x0 + PANEL - 1, TOP + PANEL - 1], outline=RULE)
        svg.append(
            f'<rect x="{x0}" y="{TOP}" width="{PANEL}" height="{PANEL}" '
            f'fill="none" stroke="rgb{RULE}"/>'
        )

        label, detail, verdict, ok = s["panels"][idx]
        text_pair(d, svg, (x0, TOP - 44), label, 27, INK, bold=True)
        text_pair(d, svg, (x0, TOP + PANEL + 18), detail, 18, MUTED)
        text_pair(d, svg, (x0, TOP + PANEL + 46), verdict, 20, INK if ok else LOSS, bold=True)

    ly = TOP + PANEL + 112
    text_pair(d, svg, (LEFT, ly), s["legend"], 19, MUTED)
    lx = LEFT + int(font(19).getlength(s["legend"])) + 26
    for count in (1, 2, 4, 0):
        colour = COUNT_COLOUR[count]
        caption = s["zero"] if count == 0 else str(count)
        d.rectangle([lx, ly, lx + 21, ly + 21], fill=colour)
        svg.append(f'<rect x="{lx}" y="{ly}" width="21" height="21" fill="rgb{colour}"/>')
        text_pair(d, svg, (lx + 29, ly + 1), caption, 19, INK)
        lx += 29 + int(font(19).getlength(caption)) + 32

    text_pair(d, svg, (LEFT, H - 50), s["foot"], 17, MUTED)
    svg.append("</svg>")

    png_path = OUT / f"{s['stem']}.{lang}.png"
    svg_path = OUT / f"{s['stem']}.{lang}.svg"
    img.save(png_path)
    svg_path.write_text("\n".join(svg), encoding="utf-8")
    root = OUT.parent.parent
    print(f"  {png_path.relative_to(root)}  {png_path.stat().st_size // 1024} KB")
    print(f"  {svg_path.relative_to(root)}  {svg_path.stat().st_size // 1024} KB")


if __name__ == "__main__":
    for language in STRINGS:
        build(language)
