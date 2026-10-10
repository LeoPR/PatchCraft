"""The patch-grid mean: torch's fold in float16, dividing first, and PatchCraft.

Measures the PatchCraft half of the paper (MEASUREMENTS.md, P1 to P3) on
synthetic images only.

    python docs/PAPER/experiments/fold_mean.py

Run it from the repository root: the count-map facts import the geometry
space from tests/test_exactness.py. Under a minute on a laptop CPU.
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from patchcraft import extract, reconstruct
from tests._rng import coverage_counts
from tests.test_exactness import _SPACE

SHAPE, P, S = (1, 16, 16), 3, 1


def fold(patches: torch.Tensor) -> torch.Tensor:
    """torch's fold of (L, C, P, P) patches, accumulating in the patches' dtype."""
    n, c = patches.shape[:2]
    cols = patches.permute(1, 2, 3, 0).reshape(1, c * P * P, n)
    return F.fold(cols, SHAPE[1:], kernel_size=P, stride=S)[0]


def report(name: str, out: torch.Tensor, img: torch.Tensor) -> None:
    inf = int(torch.isinf(out).sum())
    err = (out.float() - img.float()).abs()
    finite = err[torch.isfinite(err)]
    print(
        f"    {name:<30} inf {inf:>3}/{img.numel()}"
        f"   max |err| on finite pixels {float(finite.max()):g}"
        f"   equal to the input: {torch.equal(out, img)}"
    )


def main() -> None:
    start = time.perf_counter()
    counts = fold(extract(torch.ones(SHAPE), patch_size=P, stride=S))
    print(
        f"P1  float16 mean over a {SHAPE[1]}x{SHAPE[2]} grid, patch {P}, stride {S};"
        f" coverage counts {sorted({int(v) for v in counts.unique()})}"
    )
    gen = torch.Generator().manual_seed(20261010)
    for label, img in (
        ("constant 10000.0", torch.full(SHAPE, 10000.0, dtype=torch.float16)),
        ("uniform [0, 60000)", (torch.rand(SHAPE, generator=gen) * 60000).to(torch.float16)),
    ):
        print(f"  {label}")
        patches = extract(img, patch_size=P, stride=S)
        report("sum, then divide (float16)", fold(patches) / counts.to(torch.float16), img)
        per_patch = extract(counts.to(torch.float16), patch_size=P, stride=S)
        report("divide, then sum (float16)", fold(patches / per_patch), img)
        report("patchcraft.reconstruct", reconstruct(patches, image_shape=img.shape, stride=S), img)

    print("P2  count maps over the legal geometry space of tests/test_exactness.py")
    pow2 = 0
    occurring: Counter[int] = Counter()
    largest = 0
    for g in _SPACE:
        c = coverage_counts(*g)
        values = {int(v) for v in c.flat}
        largest = max(largest, max(values))
        if all(v & (v - 1) == 0 for v in values):
            pow2 += 1
            occurring.update(values)
    print(
        f"    {len(_SPACE):,} geometries; {pow2:,} with every count a power of two;"
        f" the counts in those maps: {sorted(occurring)}; largest count anywhere: {largest}"
    )

    print(
        "P3  F.fold on random float32 data, interior pixels exact by count (1-D, patch k, stride 1)"
    )
    gen = torch.Generator().manual_seed(20261010)
    row = []
    for k in (2, 3, 4, 5, 8, 16):
        img = torch.rand(1, 1, 4096, generator=gen)
        cols = F.unfold(img.unsqueeze(0), kernel_size=(1, k))
        back = F.fold(cols, (1, 4096), kernel_size=(1, k)) / F.fold(
            F.unfold(torch.ones(1, 1, 1, 4096), kernel_size=(1, k)), (1, 4096), kernel_size=(1, k)
        )
        interior = slice(k - 1, 4096 - k + 1)
        bad = int((back[0, 0, 0, interior] != img[0, 0, interior]).sum())
        row.append(f"k={k}: {bad}/{4096 - 2 * (k - 1)}")
    print("    not exact: " + ", ".join(row))
    print(f"{time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
