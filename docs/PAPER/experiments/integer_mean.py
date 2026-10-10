"""Means of n uint8 frames, and which of them keep every register in [0, 255].

Measures the integer half of the paper (MEASUREMENTS.md, I1 to I6) on
synthetic frames only: constant frames for the exhaustive checks, and a
smooth ramp with Gaussian noise for the bias. No captured image is read.

    python docs/PAPER/experiments/integer_mean.py

numpy only; about a minute on a laptop CPU.
"""

from __future__ import annotations

import time

import numpy as np

U8 = 255


def mean_widened_u16(frames: np.ndarray) -> np.ndarray:
    """The common idiom: widen the accumulator to uint16, divide, narrow back."""
    return np.mean(frames, axis=0, dtype=np.uint16).astype(np.uint8)


def mean_floor_each(frames: np.ndarray) -> np.ndarray:
    """Divide first, floor each term, then sum. Stays in uint8; biased low."""
    n = len(frames)
    acc = frames[0] // n
    for f in frames[1:]:
        acc = acc + f // n
    return acc


def mean_two_sums(frames: np.ndarray) -> np.ndarray:
    """Quotients and remainders summed separately, both held in uint8."""
    n = len(frames)
    q = np.zeros(frames.shape[1:], np.uint8)
    r = np.zeros(frames.shape[1:], np.uint8)
    for f in frames:
        q = q + f // n
        r = r + f % n
    return q + r // n


def mean_carry(frames: np.ndarray, *, rounded: bool = False, check: bool = False) -> np.ndarray:
    """Online quotient and remainder, every register in [0, 255] for n <= 255.

    The state is (q, r) with 0 <= r < n and n*q + r == S + r0, where S is the
    sum so far and r0 is 0, or n // 2 for the rounded mean. A frame adds its
    own quotient to q and its remainder to r, and a carry moves n from r to q
    when r reaches n. The carry is computed without forming r + r_i, which can
    reach 2n - 2: with d = n - r_i, carry when r >= d, and then r - d < n.

    With check=True the same steps run in int64 and every intermediate is
    asserted to lie in [0, 255], which is what running them in uint8 relies on.
    """
    n = len(frames)
    if not 1 <= n <= U8:
        raise ValueError("the divisor has to be a uint8 value")
    dt = np.int64 if check else np.uint8
    shape = frames.shape[1:]
    q = np.zeros(shape, dt)
    r = np.full(shape, n // 2 if rounded else 0, dt)
    for f in frames.astype(dt):
        qi, ri = f // n, f % n
        d = n - ri
        c = (r >= d).astype(dt)
        r = (r - c * d) + (1 - c) * ri
        q = q + qi + c
        if check:
            for name, v in (("qi", qi), ("ri", ri), ("d", d), ("c*d", c * d), ("r", r), ("q", q)):
                assert v.min() >= 0 and v.max() <= U8, (name, n, int(v.min()), int(v.max()))
    return q.astype(np.uint8)


def exact(frames: np.ndarray, *, rounded: bool = False) -> np.ndarray:
    n = len(frames)
    s = frames.astype(np.int64).sum(axis=0)
    return ((s + (n // 2 if rounded else 0)) // n).astype(np.uint8)


def noisy_ramp(n: int, sigma: float, seed: int) -> np.ndarray:
    """n frames of a horizontal 0..255 ramp plus Gaussian noise, rounded and clipped."""
    rng = np.random.default_rng(seed)
    base = np.tile(np.linspace(0, U8, 256), (64, 1))
    return np.clip(np.rint(base + rng.normal(0, sigma, (n, *base.shape))), 0, U8).astype(np.uint8)


def main() -> None:
    start = time.perf_counter()
    values = np.arange(256, dtype=np.uint8)

    print("I1  widened uint16 accumulator, constant frames, n = 1..1024, v = 0..255")
    first_wrong = {}
    formula_holds = True
    for n in range(1, 1025):
        out = mean_widened_u16(np.broadcast_to(values, (n, 256)))
        formula_holds &= bool(
            np.array_equal(out, ((n * values.astype(np.int64)) % 65536 // n).astype(np.uint8))
        )
        for v in np.flatnonzero(out != values):
            first_wrong.setdefault(int(v), n)
    print(f"    output == floor(((n*v) mod 65536) / n) for every pair: {formula_holds}")
    print(
        "    first wrong n for v = 255, 128, 64:",
        f"{first_wrong[255]}, {first_wrong[128]}, {first_wrong[64]}",
    )
    for n in (257, 258, 300):
        print(f"    {n} frames of 255 -> {int(mean_widened_u16(np.full((n, 1), U8, np.uint8))[0])}")

    print("I2  divide first and floor each term (stays in uint8)")
    for n in (3, 5):
        print(
            f"    {n} frames of 254 ->",
            f"{int(mean_floor_each(np.full((n, 1), 254, np.uint8))[0])}, exact 254",
        )
        frames = noisy_ramp(n, 8.0, seed=n)
        bias = (mean_floor_each(frames).astype(np.float64) - frames.mean(axis=0)).mean()
        print(
            f"    noisy ramp, sigma 8: mean error {bias:+.3f} against the exact mean;"
            f" the dropped remainders predict {-(n - 1) / 2:+.1f}"
        )

    print("I3  quotient sum and remainder sum, both in uint8")
    wrong = 0
    first = None
    for n in range(1, 256):
        out = mean_two_sums(np.broadcast_to(values, (n, 256)))
        bad = np.flatnonzero(out != values)
        wrong += len(bad)
        if first is None and len(bad):
            first = (n, int(bad[0]), int(out[bad[0]]))
    print(
        f"    wrong (n, v) pairs over n = 1..255: {wrong:,} of {255 * 256:,};"
        f" first (n, v, output) = {first}"
    )
    print(
        "    17 frames of 254 ->",
        f"{int(mean_two_sums(np.full((17, 1), 254, np.uint8))[0])}, exact 254",
    )

    print("I4  online carry, every register checked in [0, 255]")
    rng = np.random.default_rng(20261010)
    mism = {"floor": 0, "rounded": 0}
    for n in range(1, 256):
        frames = np.concatenate(
            [
                np.broadcast_to(values, (n, 256)),
                rng.integers(0, 256, (n, 256), dtype=np.uint8),
                np.where(rng.random((n, 64)) < 0.5, 0, U8).astype(np.uint8),
            ],
            axis=1,
        )
        for rounded, key in ((False, "floor"), (True, "rounded")):
            fast = mean_carry(frames, rounded=rounded)
            wide = mean_carry(frames, rounded=rounded, check=True)
            mism[key] += int(np.count_nonzero(fast != exact(frames, rounded=rounded)))
            mism[key] += int(np.count_nonzero(fast != wide))
    print(
        f"    n = 1..255, 576 columns each: mismatches against floor(S/n) {mism['floor']},"
        f" against floor((S + n//2)/n) {mism['rounded']}; no register left [0, 255]"
    )
    print(f"    17 frames of 254 -> {int(mean_carry(np.full((17, 1), 254, np.uint8))[0])}")

    print("I5  two-value floor mean, all uint8 pairs")
    a, b = np.meshgrid(values, values, indexing="ij")
    mid = (a & b) + ((a ^ b) >> 1)
    true = ((a.astype(np.int64) + b) // 2).astype(np.uint8)
    print(f"    (a & b) + ((a ^ b) >> 1) == floor((a + b) / 2): {bool(np.array_equal(mid, true))}")
    print(
        "    (a + b) >> 1 computed in uint8 is wrong on",
        f"{int(np.count_nonzero(((a + b) >> 1) != true)):,} of 65,536",
    )

    print(f"{time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
