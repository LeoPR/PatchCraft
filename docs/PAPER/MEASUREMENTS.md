# Measurements

Every number the paper states, with the command that reproduces it. The data
are synthetic: constant frames, noisy ramps, seeded random values and every
value of a format where the format is small enough to enumerate. No captured
image is read.

All of these are counts or values, not timings, so they do not depend on the
machine. They were taken on 2026-10-10 with Python 3.13.13, numpy 2.5.2 and
torch 2.14.0+cpu, and the patch-grid ones with PatchCraft at commit `c5959a9`.
Run every command from the repository root.

## Integer means

`python docs/PAPER/experiments/integer_mean.py`

| ID | What | Measured |
|---|---|---|
| I1 | `uint16` widened accumulator, constant frames, `n = 1..1024`, `v = 0..255` | output `floor(((n v) mod 65536) / n)` for every pair. First wrong `n`: 258 for `v = 255`, 512 for 128, 1024 for 64. 257 frames of 255 give 255, 258 give 0, 300 give 36 |
| I2 | divide first, floor each term | 3 frames of 254 give 252, 5 give 250. Noisy ramp, sigma 8: mean error −0.978 at `n = 3` and −1.947 at `n = 5`, against −1 and −2 predicted |
| I3 | quotient sum and remainder sum, both `uint8` | wrong on 56,695 of 65,280 `(n, v)` pairs over `n = 1..255`; first failure `n = 17`, `v = 16`, output 0; 17 frames of 254 give 238 |
| I4 | online carry, floor and rounded | `n = 1..255`, 576 columns each (the 256 constants, 256 random, 64 of 0 or 255): 0 mismatches against `floor(S/n)` and against `floor((S + floor(n/2))/n)`; every intermediate asserted inside `[0, 255]`; 17 frames of 254 give 254 |
| I5 | two-value floor mean | `(a & b) + ((a ^ b) >> 1)` equals `floor((a + b)/2)` on all 65,536 `uint8` pairs; `(a + b) >> 1` computed in `uint8` is wrong on 32,640 |

## Floating-point means

`python docs/PAPER/experiments/float_mean.py`

| ID | What | Measured |
|---|---|---|
| F1 | `k` identical copies summed one at a time, divided by `k` | `float16`, all 17,408 values in `[2^-8, 2^9)`: 0 failures for `k` = 1, 2, 4; 2,907 at `k = 3`, 8,109 at 8, 12,512 at 16, and every `k` up to 16 other than 1, 2, 4 fails. `float32`, all 8,388,608 values in `[1, 2)`: 0 at `k` = 2, 4; 1,398,101 at 3, 3,914,684 at 8 (46.7%), 6,018,398 at 16, 7,199,620 at 32 |
| F2 | the same, summed pairwise, `k = 2..64` powers of two | 0 failures in both formats |
| F3 | divide first, overflow | `float16`: `65504 / 3 = 21840`, summed three times: `inf` |
| F4 | a mean outside `[min, max]` | `float16`, `x = y = 2^-24`: `x/2 + y/2 = 0` |
| F5 | error over `gamma_n sum|x| / n` | 2,000 rows of uniform `[-1, 1)` per `n` in {3, 5, 9, 81, 100, 1000}: largest ratio 0.661 for sum first (`float32`, `n = 3`) and 0.652 for divide first (`float16`, `n = 3`) |
| F6 | numpy and a `float16` stack | sum of 300 × 1000.0 is `inf`; `mean` returns 1000.0 through a `float32` intermediate; `mean(dtype=float16)` returns `inf` |

## The patch grid

`python docs/PAPER/experiments/fold_mean.py`, and for P4
`python tools/compare_exactness_rules.py`.

| ID | What | Measured |
|---|---|---|
| P1 | `float16`, 1×16×16, patch 3, stride 1, counts {1, 2, 3, 4, 6, 9} | constant 10000.0: sum then divide is `inf` on 144 of 256 pixels, the count-9 interior; divide then sum and `patchcraft.reconstruct` return the input. Uniform `[0, 60000)`: sum then divide is `inf` on 198 of 256; divide then sum has no `inf` and misses by up to 96; `reconstruct` returns the input |
| P2 | count maps of the 126,736 legal geometries in `tests/test_exactness.py` | 34,969 have every count a power of two, and those maps hold only 1, 2 and 4; the largest count anywhere is 81 |
| P3 | `F.fold`, one row of 4,096 random `float32` values, patch `k`, stride 1 | interior pixels not exact: 0 of 4,094 at `k = 2`, 663 of 4,092 at 3, 0 of 4,090 at 4, 524 of 4,088 at 5, 1,648 of 4,082 at 8, 2,715 of 4,066 at 16 |
| P4 | worst miss outside the rule | 19.0 ULP at `(h, w, ph, pw, sh, sw) = (18, 23, 9, 9, 1, 1)`, on a pixel covered 81 times |

## References

`python docs/PAPER/experiments/check_sources.py` re-opens the 85 rows of
[`sources.tsv`](sources.tsv). On 2026-10-10, 84 confirmed; the miss was the
PMC copy of [hernandez-mier-2025], which answered with a browser check, while
the DOI row of the same paper confirmed.
