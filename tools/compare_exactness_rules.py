"""Compare two candidate exactness rules on the space the test suite defines.

The contract says the extract/reconstruct round trip is bit-exact if and only
if every value of the coverage count map is a power of two. The obvious looser
alternative is to keep the largest count small, k_max <= 4. This script runs
both against the ground truth, which is the round trip itself on seeded random
full-mantissa float32 data, over every geometry in the space that
tests/test_exactness.py enumerates, and reports how often each rule promises
exactness it does not deliver, and how often it withholds a promise it could
have made. It then reports the worst miss outside the rule in ULP of the
pixel's own value, over the geometries where coverage reaches 36 or more,
which is where the bound (k+1)*eps*|v| is largest.

    python tools/compare_exactness_rules.py

Run it from the repository root. It takes a few minutes on a laptop CPU.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from patchcraft import extract, reconstruct
from tests._rng import bit_equal, coverage_counts, rand_image
from tests.test_exactness import _SPACE

SEEDS = range(5)


def exact_on_every_seed(g: tuple[int, int, int, int, int, int]) -> bool:
    h, w, ph, pw, sh, sw = g
    for seed in SEEDS:
        img = rand_image(1, h, w, torch.float32, seed)
        out = reconstruct(
            extract(img, patch_size=(ph, pw), stride=(sh, sw)),
            image_shape=img.shape,
            stride=(sh, sw),
        )
        if not bit_equal(out, img):
            return False
    return True


def worst_ulp(min_count: int = 36) -> tuple[float, tuple[int, ...], int]:
    """Largest |out - img| / ulp(img) over the high-coverage geometries."""
    worst, where, at_k = 0.0, (), 0
    for g in _SPACE:
        counts = coverage_counts(*g)
        if counts.max() < min_count:
            continue
        h, w, ph, pw, sh, sw = g
        for seed in SEEDS:
            img = rand_image(1, h, w, torch.float32, seed)
            out = reconstruct(
                extract(img, patch_size=(ph, pw), stride=(sh, sw)),
                image_shape=img.shape,
                stride=(sh, sw),
            )
            ulp = torch.nextafter(img.abs(), torch.tensor(float("inf"))) - img.abs()
            ratio = ((out - img).abs() / ulp)[0]
            val = float(ratio.max())
            if val > worst:
                idx = int(ratio.argmax())
                worst, where = val, g
                at_k = int(counts.flat[idx])
    return worst, where, at_k


def main() -> None:
    start = time.perf_counter()
    over = {"power of two": 0, "k_max <= 4": 0}
    under = {"power of two": 0, "k_max <= 4": 0}
    disagree = 0
    for g in _SPACE:
        counts = {int(v) for v in coverage_counts(*g).flat}
        promise = {
            "power of two": all(v & (v - 1) == 0 for v in counts),
            "k_max <= 4": max(counts) <= 4,
        }
        disagree += promise["power of two"] != promise["k_max <= 4"]
        truth = exact_on_every_seed(g)
        for rule, says_exact in promise.items():
            over[rule] += says_exact and not truth
            under[rule] += truth and not says_exact

    print(f"space: {len(_SPACE):,} legal geometries (tests/test_exactness.py)")
    print(f"ground truth: the round trip, float32, seeds {list(SEEDS)}")
    print(f"the two rules disagree on {disagree:,} geometries")
    for rule in over:
        print(
            f"  {rule:>12}: promises exact and is not {over[rule]:>6,}"
            f"   exact without being promised {under[rule]:>6,}"
        )
    ulps, where, k = worst_ulp()
    print(f"worst miss outside the rule: {ulps:.1f} ULP, at (h, w, ph, pw, sh, sw) = {where}, "
          f"on a pixel covered {k} times")
    print(f"{time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
