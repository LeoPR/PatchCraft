# Formal statements

The definitions and propositions the paper rests on, each with its proof or,
where only a measurement supports it, the measurement's ID in
[MEASUREMENTS.md](MEASUREMENTS.md). A statement marked *measured* is
established for the range the experiment covers and no further.

## 1. Integer means

**Setting.** `U_w = {0, ..., M}` with `M = 2^w - 1` is the set of `w`-bit
unsigned integers, `M = 255` for `uint8`. Frames `x_1, ..., x_n` are in `U_w`,
pixel by pixel, and `S_i = x_1 + ... + x_i`, `S = S_n`. Write `q_i = x_i div n`
and `r_i = x_i mod n`, so `x_i = n q_i + r_i` with `0 <= r_i < n`. The target
is the floor mean `floor(S / n)` or the rounded mean `floor((S + floor(n/2)) / n)`.

**Domain.** Both targets lie in `[min x_i, max x_i]`, hence in `U_w`. The
result always fits; the question is whether the computation does.

### Proposition 1 (widened accumulator)

Summing in `U_b`, `b >= w` bits, wraps modulo `2^b`, so the computed mean is
`floor((S mod 2^b) / n)`. For `n <= 2^b` it equals `floor(S / n)` if and only if
`S < 2^b`. For `n` constant frames of value `v > 0` the first wrong `n` is
`ceil(2^b / v)`.

*Proof.* The sum in `U_b` is `S mod 2^b`, and the division is exact integer
division of that value. If `S < 2^b` nothing wrapped. If `S >= 2^b`, then
`S mod 2^b <= S - 2^b <= S - n`, so the computed floor is at least one below
`floor(S / n)`. For constant frames `S = n v`, which first reaches `2^b` at
`n = ceil(2^b / v)`. ∎

*Measured (I1):* for `uint16` and `v = 255`, the first wrong `n` is 258; 258
frames of 255 average to 0 and 300 frames to 36, which is
`floor((300 · 255 mod 65536) / 300)`.

### Proposition 2 (divide first, floor each term)

`sum q_i = floor(S / n) - floor(R / n)` with `R = sum r_i`, and
`0 <= R <= n(n - 1)`, so the deficit `floor(R / n)` lies in `[0, n - 1]`.
Against the exact mean `S / n` the deficit is `R / n`, whose expectation is
`(n - 1)/2` when the remainders are uniform on `{0, ..., n - 1}`. Every partial
sum `q_1 + ... + q_i <= floor(S_i / n) <= M`, so nothing leaves `U_w`.

*Proof.* `S = n sum q_i + R`, so `floor(S / n) = sum q_i + floor(R / n)`. The
bound on `R` follows from `r_i <= n - 1`. The partial sums are bounded the
same way, applied to the first `i` frames. ∎

*Measured (I2):* a mean error of −0.978 at `n = 3` and −1.947 at `n = 5` on a
noisy ramp, against −1 and −2 predicted; 3 frames of 254 give 252.

### Proposition 3 (two sums)

Accumulating `Q = sum q_i` and `R = sum r_i` separately and returning
`Q + R div n` is exact, and `Q <= M` always. `R` needs `n(n - 1) <= M`; in
`uint8` that holds for `n <= 16` and fails from `n = 17`, where `17 · 16 = 272`.

*Measured (I3):* the first failure is `n = 17`, `v = 16`, output 0; 17 frames
of 254 give 238.

### Proposition 4 (online carry)

Fix `n` with `1 <= n <= M` and `r_0` equal to 0 or `floor(n/2)`. Start from
`Q_0 = 0`, `R_0 = r_0` and, for each frame,

```
d_i = n - r_i
c_i = [R_{i-1} >= d_i]                        (0 or 1)
R_i = (R_{i-1} - c_i d_i) + (1 - c_i) r_i
Q_i = Q_{i-1} + q_i + c_i
```

Then for every `i`:

1. `n Q_i + R_i = S_i + r_0` and `0 <= R_i < n`, so `Q_i = floor((S_i + r_0) / n)`
   and `Q_n` is the floor mean (`r_0 = 0`) or the rounded mean (`r_0 = floor(n/2)`);
2. every value the update forms lies in `U_w`.

*Proof.* (1) By induction; it holds at `i = 0`. If `c_i = 1`, then
`R_{i-1} >= n - r_i`, so `R_i = R_{i-1} + r_i - n >= 0`, and `R_i < n` because
`R_{i-1}, r_i < n`; also `n Q_i + R_i = n Q_{i-1} + R_{i-1} + n q_i + r_i`. If
`c_i = 0`, then `R_i = R_{i-1} + r_i < n`, with the same identity. Either way
`n Q_i + R_i = S_{i-1} + r_0 + x_i = S_i + r_0`.

(2) `q_i <= x_i <= M` and `r_i, R_i <= n - 1 < M`. `d_i` lies in `[1, n]`,
inside `U_w` because `n <= M`. When `c_i = 1`, `c_i d_i <= R_{i-1}`, so
`R_{i-1} - c_i d_i` lies in `[0, R_{i-1}]`; when `c_i = 0` it is `R_{i-1}`.
`(1 - c_i) r_i <= r_i`. Finally `Q_{i-1} + q_i <= Q_i` and
`Q_i = floor((S_i + r_0)/n) <= floor((n M + floor(n/2))/n) = M`, since
`floor(n/2)/n < 1`. ∎

The update never forms `R_{i-1} + r_i`, which can reach `2n - 2`. That is
what keeps `n` up to `M` and not up to `(M + 1)/2`.

**Reading.** This is "divide first, then sum" made exact: `q_i` is the part
of each frame that divides cleanly, and the carry restores what Proposition 2
loses. `Q_i` is the partial sum of the final mean, so it grows from 0 to the
result and never passes it.

*Measured (I4):* `n = 1..255` over 576 columns each, 0 mismatches against both
targets, every intermediate asserted inside `[0, 255]` by a wide-integer twin.

### Proposition 5 (two values)

`(a & b) + ((a ^ b) >> 1) = floor((a + b) / 2)` and
`(a | b) - ((a ^ b) >> 1) = ceil((a + b) / 2)`, with every intermediate at most
`max(a, b)`.

*Proof.* `a + b = 2 (a & b) + (a ^ b)`, since `a ^ b` is the sum without
carries and `a & b` the carries. Halving gives the first; `a | b = (a & b) + (a ^ b)`
gives the second. [warren2013, sec. 2-5; dietz-magic] ∎

*Measured (I5):* all 65,536 `uint8` pairs.

## 2. Floating-point means

**Setting.** Round to nearest, ties to even, precision `p`, unit roundoff
`u = 2^-p`, `gamma_n = n u / (1 - n u)`. The standard model
`fl(a op b) = (a op b)(1 + delta)`, `|delta| <= u`, holds only while no overflow
or underflow occurs [higham1993; croci2022; ogita2005]: the precision
literature takes the range for granted, which is the gap this paper studies.

### Proposition 6 (identical copies, one at a time)

Let `s_1 = x` and `s_j = fl(s_{j-1} + x)`. Absent overflow and underflow,
`fl(s_k / k) = x` for every `x` when `k` is 1, 2 or 4.

*Proof.* `k = 1, 2`: `s_2 = 2x` exactly, and division by 2 is exact. `k = 4`:
write `x = m 2^e` with `2^(p-1) <= m < 2^p`. `s_3 = fl(3x)` differs from `3x` by
at most half an ulp of `3x`, which is at most `2^(e+1)`, while the floats
nearest `4x = m 2^(e+2)` are `4x ± 2^(e+2)` (or `4x - 2^(e+1)` below when `m` is
`2^(p-1)`). So `s_3 + x` lies within half an ulp of `4x` and rounds to it, with
two edge cases. A tie at exactly `2^(e+1)` requires `3m ≡ 2 (mod 4)`, so `m` is
even and ties-to-even picks `4x`. When `m = 2^(p-1)`, `3x` is exact, so there
is nothing to round. Then `s_4 = 4x` and the division by 4 is exact. ∎

*Measured (F1), the converse within range:* every other `k` up to 16 fails for
some `x` in `float16` (all 17,408 values in `[2^-8, 2^9)`), and `k` = 3, 8, 16, 32
fail in `float32` (all 8,388,608 values in `[1, 2)`; 3,914,684 of them at
`k = 8`). A proof for every `k` is open.

### Proposition 7 (identical copies, pairwise)

For `k = 2^m`, summing pairwise doubles the value at each level, so the sum
is `2^m x` exactly and the mean is `x`, absent overflow. *Measured (F2):* 0
failures for `k` up to 64 in both formats.

### Proposition 8 (divide first in floating point)

Let `t_j = fl(t_{j-1} + fl(x_j / n))`.

1. If `n` is a power of two, `t_n` equals the sum-first result bit for bit,
   absent overflow and underflow: `x_j / n` is exact, and scaling by a power
   of two commutes with rounding, so each `t_j` is the sum-first partial sum
   divided by `n`. Dividing first then costs nothing.
2. In exact arithmetic the partial sums are sums of the `x_j` with
   nonnegative weights totalling `j / n <= 1`, so they lie in
   `[min(0, min x), max(0, max x)]`: sub-convex, not convex.
3. It is not unconditionally overflow-free: rounding can carry a partial sum
   past `realmax`. `(1 + gamma_n) max |x| <= realmax` suffices, because by
   induction each computed `|t_j|` is at most `(1 + gamma_j)(j / n) max |x|`.
4. Both orders satisfy `|computed - S/n| <= gamma_n sum |x_i| / n`: sum first
   carries `gamma_(n-1)` from the recursive sum [higham1993, (2.6)] and one
   more rounding from the division, and divide first carries one rounding
   per term into the same recursive sum. Dividing first gives up no
   worst-case accuracy.

*Measured:* (3) in `float16`, `fl(65504 / 3) = 21840`, and three of those sum
to `inf` (F3). (4) The largest ratio of error to bound is 0.661 over `n` from 3
to 1,000 in both formats and both orders (F5).

### Proposition 9 (a computed mean can leave `[min, max]`)

Rounding and underflow can each put the result outside the interval the
exact mean lives in. *Measured:* every failure in F1 is a mean of identical
values that is not that value; and in `float16`, `2^-24 / 2 + 2^-24 / 2 = 0`
(F4) [boldo2015, Thm 2].

## 3. The patch grid

**Setting.** A legal geometry of PatchCraft covers an `h × w` image with
patches `ph × pw` at strides `sh <= ph`, `sw <= pw`, exactly, with no pixel left
uncovered. The count map is `K(y, x) = a(y) b(x)`, the product of the per-axis
coverage counts. Reconstructing untouched patches sums, at each pixel, `K`
identical copies of the pixel's value and divides by `K`.

### Proposition 10 (power-of-two maps hold only 1, 2 and 4)

On each axis the count starts at 1 at the edge and changes by at most 1
between neighbours, so it takes every value from 1 to its maximum. If every
value of `K` is a power of two, each axis has maximum at most 2, and `K` holds
only 1, 2 and 4.

*Proof.* `a(y + 1) - a(y) = [a patch starts at y + 1] - [a patch ends at y]`,
and at most one patch starts or ends at a position because the stride is at
least 1; `a(0) = 1`. So `a` takes every value in `1..max a`. If `max a >= 3`,
some `y` has `a(y) = 3`, and at the column edge `b = 1`, so `K = 3` there. ∎

*Measured (P2):* 34,969 of the 126,736 legal geometries have every count a
power of two, and those maps hold exactly `{1, 2, 4}`; the largest count
anywhere is 81.

**Corollary (PatchCraft's exactness rule).** By Propositions 6 and 10, a
geometry whose count map is all powers of two reconstructs exactly, whatever
order the fold sums in, since the copies are identical. The converse, that a
count of 3 or more breaks exactness for some data, is what the falsification
suite in `tests/test_exactness.py` measures; outside the rule the miss reaches
19 ULP on a pixel covered 81 times (P4).

### Proposition 11 (the fold's accumulator)

torch's `F.fold` accumulates in the tensor's own dtype [pytorch-col2im], so a
`float16` fold of values whose sum exceeds 65504 overflows even when their
mean does not. There are two ways out. Dividing each patch by its count
before folding keeps every partial sum within the input's range in exact
arithmetic (Proposition 8.2), and in `float16` that is measured, not
guaranteed (Proposition 8.3). Promoting to a `float32` accumulator, which is
what PatchCraft does, needs a wider type to exist.

*Measured (P1):* constant 10000.0 at counts up to 9: sum then divide gives
`inf` on the 144 count-9 pixels of 256; divide then sum and
`patchcraft.reconstruct` both return the input. On uniform `[0, 60000)` data,
divide then sum has no `inf` and misses by up to 96, three `float16` steps at
that magnitude, while the promoted accumulator returns the input.
