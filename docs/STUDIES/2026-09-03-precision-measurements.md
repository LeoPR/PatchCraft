# Precision and speed: what was measured

Date: 2026-09-03. CPU, torch 2.14.0+cpu, on the machine named in
[`../PERFORMANCE.md`](../PERFORMANCE.md).

These are the measurements behind
[ADR 0004](../ADR/0004-precision-and-effort-parameters.md). They live here so
that the ADR can be read as a decision rather than as a lab notebook, and so
that a number can be checked without reading the argument that used it.

## The library already has a precision policy. It has no name.


Five places decide precision on the caller's behalf, none of them visible from a call site:

| Site | The choice made for you |
|---|---|
| `reconstruct.py:55`, `stitch.py:189` | `float16` and `bfloat16` accumulate in `float32`, rounding once on return |
| `stitch.py:229` | the denominator is floored by `clamp(min=1e-6)`, an absolute constant |
| `metrics.py:67`, `:123` | accumulation promotes to `float64` whatever the input was |
| `metrics.py:164` | `clamp_min(finfo(float64).tiny)` before the PSNR logarithm |
| `resize.py:47`, `:148` | every cross-backend conversion passes through normalised `float32` in [0, 1] |

The concept is already here, applied by default and unnamed. That is the finding: a knob
naming any of these introduces nothing, it exposes what is already decided for the caller.

## What forced the question


A user with a `uint8` image cannot use the library at all. `extract` raises
`NotImplementedError: "im2col_out_cpu" not implemented for 'Byte'`, and `reconstruct` and
`stitch` reject every non-floating-point dtype outright. The only route is `.float()`
first, which on an 8192x8192 tile at patch 512 stride 256 turns a 721 MiB patch stack into
2883 MiB.

The owner's framing of the problem is the right one and worth quoting, because it rules out
the easy answer: `input(byte) -> operation(???) -> output(byte)` is not the same thing as
`input(byte) -> operation(byte) -> output(byte)`. Returning the input dtype after widening
internally gives back neither the memory nor the time.

## The measurements


All taken on 2026-09-03, CPU, torch 2.14.0+cpu, on the machine named in
[`PERFORMANCE.md`](../PERFORMANCE.md).

**Byte-native averaging is exact, and it is the fastest option.** The identity
`sum(a_i)//k == sum(a_i//k) + sum(a_i mod k)//k` holds for every `k`, verified exhaustively
for k in 2..5 and over 200,000 random samples up to k=16. The quotient accumulator always
fits `uint8`; the remainder accumulator holds `k*(k-1)` and so fits `uint8` only for k <= 16.
At k=4 over 3.1M pixels: 2.51 ms byte-native, 4.24 ms widening to `int16`, 6.19 ms via
`float32`, with accumulators of 3.0, 6.0 and 12.0 MiB.

**A first measurement of this said the opposite, and the error is instructive.** Writing
the byte path as `(x // k).sum(0)` measured 9x slower than float. `torch`'s `uint8.sum()`
promotes to `int64` silently, so that benchmark was the very
`input(byte) -> operation(???)` shape the design exists to avoid. Only accumulators declared
`uint8` measure the thing.

**What decides speed is not integer versus float. It is whether the divisor is a power of
two.** The same arithmetic, the same k, written with `//` and `%` against `>>` and `&`:

| k | via `//` and `%` | via `>>` and `&` | ratio |
|---|---|---|---|
| 2 | 22.51 ms | 1.39 ms | 16.1x |
| 4 | 40.17 ms | 2.53 ms | 15.9x |
| 8 | 76.46 ms | 5.09 ms | 15.0x |
| 16 | 148.71 ms | 9.71 ms | 15.3x |

x86 has no SIMD integer division. A power-of-two divisor turns it into a shift, which
vectorises.

**The same idea generalises to floats, and fixes a defect the library papered over.**
THEORY §9.2 records that a constant `float16` image at 10000.0 with `ph=3, stride=1`
returned `inf` in 144 of 256 pixels, and that `float32` promotion was added to prevent it.
Reproduced exactly. Scaling each patch by `1/k` before the fold keeps the numerator at
1.0e+04 against a finite `float16` maximum of 6.55e+04, produces no `inf`, and returns the
image with zero error, in a 2 byte per pixel accumulator instead of 4. On ordinary [0, 1]
data it costs 5.96e-08. On this machine it is nonetheless 1.35x slower than the promotion,
because torch's CPU `float16` kernels are weak. The advantage is real but it is per-dtype
and must be measured, never assumed.

**`bfloat16` is in the wrong bucket.** THEORY §9.2 justifies promoting both half formats by
`float16` overflowing its finite maximum of 65504, then applies the rule to `bfloat16` too.
`bfloat16` carries `float32`'s exponent: on the same case its numerator reached 9.01e+04
with no `inf` and a finite maximum of 3.39e+38. Promoting it buys precision, since it has
7 mantissa bits against `float16`'s 10, not range. The stated reason covers one of the two.

## What each measurement decided

| Measurement | What it settled in the ADR |
|---|---|
| Byte-native averaging is exact and fastest, 2.51 ms against 6.19 via float32 | that an integer path is worth having at all |
| The first benchmark said the opposite, because `uint8.sum()` promotes to `int64` | that only accumulators declared in the target dtype measure the thing |
| 15 to 16x between `//` and `>>` at every k | that the deciding property is a power-of-two divisor, not integer versus float |
| Scaling by `1/k` before the fold removes the `float16` overflow at half the accumulator memory | that the idea generalises beyond integers |
| `bfloat16`'s numerator peaks at 9.0e+04 against a finite maximum of 3.4e+38 | that THEORY §9.2 justified its promotion with a fact false of it, fixed in 0.5.2 |
| A `float64` accumulator is exact on 76 of 76 geometries where `float32` is exact on 45 | that the exactness predicate is a property of the geometry **and** the accumulator, which amended ADR 0003 |
