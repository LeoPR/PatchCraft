"""Compare the ways a patch grid can meet an image it does not cover exactly.

When (H - patch) is not a multiple of the stride, a regular grid either drops
the remainder, pads the image until the grid closes, or moves its last window
back so that it ends at the border. This script measures each on one frame,
with the remainder sent to the end or split between both sides, and reports
how many pixels each loses, how many pixels outside the frame its patches
contain, which coverage counts it produces, and whether the round trip comes
back exact. The round trip is computed independently of the library, by
gathering every window and scatter-adding it back, so the strategies the
library does not implement can be measured too.

It also prints how numpy and torch name the same padding modes, and the
arithmetic that misaligns an LR/HR pair when each image is centred on its own.

    python tools/compare_border_strategies.py

Run it from the repository root. It takes a few seconds on a laptop CPU.
Measurements behind docs/STUDIES/2026-10-10-border-policy.md.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)

H, W, P, S = 1080, 1920, 256, 128
STRATEGIES = (
    "truncate, end",
    "truncate, centre",
    "pad, end",
    "pad, centre",
    "move the last window",
)


def starts(size: int, p: int, s: int, how: str) -> list[int]:
    """Per-axis window starts in frame coordinates; negative or past the end under padding."""
    n = (size - p) // s + 1
    rest = size - ((n - 1) * s + p)
    pad = max(p - size, 0) if size < p else -(size - p) % s
    if how == "truncate, end":
        return [i * s for i in range(n)]
    if how == "truncate, centre":
        return [rest // 2 + i * s for i in range(n)]
    if how == "pad, end":
        return [i * s for i in range((size + pad - p) // s + 1)]
    if how == "pad, centre":
        return [i * s - pad // 2 for i in range((size + pad - p) // s + 1)]
    if how == "move the last window":
        return [min(i * s, size - p) for i in range(-(-(size - p) // s) + 1)]
    raise ValueError(how)


def axis_counts(size: int, p: int, first: list[int]) -> torch.Tensor:
    c = torch.zeros(size, dtype=torch.int64)
    for a in first:
        c[max(a, 0) : min(a + p, size)] += 1
    return c


def round_trip_exact(rows: list[int], cols: list[int]) -> bool:
    """Gather every window (reflect-padding where it leaves the frame), scatter-add, divide."""
    img = torch.rand(H, W, generator=torch.Generator().manual_seed(0))
    top, left = max(0, -min(rows)), max(0, -min(cols))
    bottom, right = max(0, max(rows) + P - H), max(0, max(cols) + P - W)
    canvas = F.pad(img[None, None], (left, right, top, bottom), mode="reflect")[0, 0]
    total = torch.zeros_like(canvas)
    count = torch.zeros_like(canvas)
    for a in rows:
        for b in cols:
            window = (slice(a + top, a + top + P), slice(b + left, b + left + P))
            total[window] += canvas[window]
            count[window] += 1
    return torch.equal((total / count)[top : top + H, left : left + W], img)


def main() -> None:
    print(f"frame {H}x{W}, patch {P}, stride {S}; the regular grid ends at row "
          f"{((H - P) // S) * S + P}")
    for how in STRATEGIES:
        rows, cols = starts(H, P, S, how), starts(W, P, S, how)
        cr, cc = axis_counts(H, P, rows), axis_counts(W, P, cols)
        lost = int((cr == 0).sum())
        outside = max(max(0, -a) for a in rows) + max(max(0, a + P - H) for a in rows)
        worst = max(max(0, -a) + max(0, a + P - H) for a in rows)
        counts = sorted({int(a * b) for a in cr.unique() for b in cc.unique() if a * b})
        exact = round_trip_exact(rows, cols) if lost == 0 else None
        print(f"  {how:21s} rows lost {lost:2d}, rows outside the frame {outside:2d} "
              f"(at most {worst} in one patch), counts {counts}, round trip exact {exact}")

    print("padding modes on 1 2 3 4 5, two on each side")
    row = np.arange(1, 6, dtype=np.float32)
    t = torch.from_numpy(row).view(1, 1, -1)
    for np_mode, torch_mode in (("constant", "constant"), ("edge", "replicate"),
                                ("reflect", "reflect"), ("symmetric", None), ("wrap", "circular")):
        a = np.pad(row, 2, mode=np_mode).astype(int).tolist()
        b = F.pad(t, (2, 2), mode=torch_mode).view(-1).int().tolist() if torch_mode else None
        name = torch_mode or "(none)"
        print(f"  numpy {np_mode:9s} {a}   torch {name:9s} {b if b is not None else ''}")

    print("an LR/HR pair at scale 2, patch = stride, each side centred on its own")
    lr, scale, p_lr = 33, 2, 8
    for label, amount in (
        ("remainder dropped", lambda n, p: n - ((n - p) // p * p + p)),
        ("padding added", lambda n, p: -(n - p) % p),
    ):
        a_lr, a_hr = amount(lr, p_lr), amount(lr * scale, p_lr * scale)
        print(f"  {label}: LR {lr} {a_lr}, first offset {a_lr // 2}; "
              f"HR {lr * scale} {a_hr}, first offset {a_hr // 2}; "
              f"LR offset x {scale} = {a_lr // 2 * scale}")


if __name__ == "__main__":
    main()
