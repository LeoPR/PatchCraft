"""Floating-point means: identical copies, overflow, and leaving [min, max].

Measures the floating-point half of the paper (MEASUREMENTS.md, F1 to F6) on
synthetic values only.

    python docs/PAPER/experiments/float_mean.py

numpy only; under a minute on a laptop CPU.
"""

from __future__ import annotations

import math
import time

import numpy as np


def seq_sum(x: np.ndarray, k: int) -> np.ndarray:
    """k copies of x added one at a time, in x's dtype."""
    s = x.copy()
    for _ in range(k - 1):
        s = s + x
    return s


def tree_sum(x: np.ndarray, k: int) -> np.ndarray:
    """k = 2**m copies of x added pairwise: each level doubles the previous one."""
    s = x.copy()
    while k > 1:
        s = s + s
        k //= 2
    return s


def all_float16_in(lo: float, hi: float) -> np.ndarray:
    """Every float16 value in [lo, hi), from its bit patterns."""
    bits = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16).view(np.float16)
    return bits[(bits >= lo) & (bits < hi)]


def main() -> None:
    start = time.perf_counter()
    with np.errstate(over="ignore"):
        print("F1  k identical copies, summed one at a time, then divided by k")
        x16 = all_float16_in(2.0**-8, 2.0**9)
        fails16 = {
            k: int(np.count_nonzero(seq_sum(x16, k) / np.float16(k) != x16)) for k in range(1, 17)
        }
        print(f"    float16, all {len(x16):,} values in [2^-8, 2^9): failures by k {fails16}")
        x32 = (np.arange(1 << 23, dtype=np.uint32) | np.uint32(0x3F800000)).view(np.float32)
        fails32 = {
            k: int(np.count_nonzero(seq_sum(x32, k) / np.float32(k) != x32))
            for k in (2, 3, 4, 8, 16, 32)
        }
        print(f"    float32, all {len(x32):,} values in [1, 2): failures by k {fails32}")
        print("F2  the same copies added pairwise, k a power of two")
        tree16 = {
            k: int(np.count_nonzero(tree_sum(x16, k) / np.float16(k) != x16))
            for k in (2, 4, 8, 16, 32, 64)
        }
        tree32 = {
            k: int(np.count_nonzero(tree_sum(x32, k) / np.float32(k) != x32))
            for k in (2, 4, 8, 16, 32, 64)
        }
        print(f"    float16 failures {tree16}; float32 failures {tree32}")

        print("F3  divide first is not unconditionally overflow-free")
        top = np.float16(65504.0)
        third = top / np.float16(3)
        print(f"    float16: 65504/3 = {third}, then summed three times: {third + third + third}")
        print(
            "    float16: 65504 summed three times, then divided:",
            (top + top + top) / np.float16(3),
        )

        print("F4  a computed mean can leave [min, max]")
        tiny = np.float16(2.0**-24)
        print(f"    float16: x = y = 2^-24, x/2 + y/2 = {tiny / 2 + tiny / 2}, outside [x, x]")
        print(
            "    and every failure counted in F1 is a mean of identical values"
            " that is not that value"
        )

        print(
            "F5  error over the bound gamma_n * sum|x| / n, both orders,"
            " 2,000 rows of uniform [-1, 1)"
        )
        rng = np.random.default_rng(20261010)
        for dt in (np.float16, np.float32):
            u = np.finfo(dt).eps / 2
            ratios = {}
            for n in (3, 5, 9, 81, 100, 1000):
                x = rng.uniform(-1, 1, (2000, n)).astype(dt)
                exact = np.array([math.fsum(row) for row in x.astype(np.float64)]) / n
                bound = n * u / (1 - n * u) * np.abs(x.astype(np.float64)).sum(axis=1) / n
                s = np.zeros(2000, dt)
                t = np.zeros(2000, dt)
                for j in range(n):
                    s = s + x[:, j]
                    t = t + x[:, j] / dt(n)
                s = s / dt(n)
                ratios[n] = (
                    round(float((np.abs(s - exact) / bound).max()), 3),
                    round(float((np.abs(t - exact) / bound).max()), 3),
                )
            print(
                f"    {np.dtype(dt).name}: largest error/bound by n,"
                f" (sum first, divide first): {ratios}"
            )

        print("F6  wide accumulators in the libraries")
        frames = np.full((300, 1), 1000.0, np.float16)
        print(
            f"    numpy float16: sum of 300 x 1000.0 = {frames.sum(axis=0)[0]},"
            f" mean = {frames.mean(axis=0)[0]},"
            f" mean with dtype=float16 = {frames.mean(axis=0, dtype=np.float16)[0]}"
        )
    print(f"{time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
