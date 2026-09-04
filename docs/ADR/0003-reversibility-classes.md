# ADR 0003: declaring exactness per regime

- **Status:** Accepted 2026-09-03
- **Date:** 2026-08-04, rescoped and accepted 2026-09-03
- **Deciders:** Leonardo Marques de Souza
- **Relates to:** [ADR 0001](0001-patch-extraction-api.md), [ADR 0004](0004-precision-and-effort-parameters.md), [`THEORY.md`](../THEORY.md) §0, §9, [`STUDIES/2026-08-04-patch-techniques.md`](../STUDIES/2026-08-04-patch-techniques.md) §4, §5

> **What this rescoping did.** The original draft decided two things at once: a
> rule about *how exactness is declared*, and a vocabulary plus naming policy for
> *six transforms that were being considered*. A month later the rule is
> governing the code and the vocabulary has never been used: `R1`, `R2` and `R3`
> appear in no docstring, in no contract section and on no page a reader opens,
> and none of rotation, flips, padding, DCT, quantization or dithering shipped.
> The rule is what was accepted. The vocabulary is deferred, and §4 says until
> when. Freezing unused vocabulary into 1.0 would be inventing work where
> nothing is broken.

## 1. Context: "reversible" means three different things

Several operations claim a reversal, and the word covers three unrelated cases.

`rot90` and flips reverse **exactly**: the same tensor comes back, any dtype.
Arbitrary-angle rotation reverses only **approximately**, because the forward and
inverse interpolations each degrade the signal, and no parameter choice makes it
exact. Quantization does not reverse at all: it is many-to-one by construction,
and the useful contract is a measured error bound rather than an inverse.

The library already ships operations of the second and third kind, and they are
not accidents. `resize` from 64 to 32 and back differs from the original by
0.619 on data in [0, 1], because two thirds of the pixels are gone; that is what
resizing is. `stitch(weight="hann")` averages the pixels that overlap, so the
sum survives and the parts do not; that is what seam blending is.

**So the problem is not that inexact operations exist.** The problem is that
nothing in the contract distinguishes them, and a caller cannot tell from a
signature whether what comes back is the same tensor, a close one, or a
different thing entirely.

## 2. Decision: exactness is declared per regime, not per function

Each transform declares, in its docstring and in its [`THEORY.md`](../THEORY.md)
§9 contract section, the conditions under which its result is exact, and states
the measured error where it is not.

**The condition must be evaluable by the caller before the call.** A statement
that a result is exact "in the usual case" is not a contract; a statement that
it is exact when every value of the coverage map is a power of two is one,
because the caller can compute that from the geometry alone.

**One label per function is not writable for this library**, which is the whole
reason for the rule. `stitch`'s behaviour is selected by its `weight` argument.
`reconstruct`'s depends on the geometry, on the dtype, and, since ADR 0004
measured it, on the accumulator. Any single label attached to those names would
be false over part of their accepted input.

### 2.1 The regime axes, as measured

**Geometry.** The `extract`/`reconstruct` round trip is bit-exact when every
value of the overlap coverage map is a power of two, and not otherwise. The
reason is arithmetic rather than statistical: the fold sums `k` copies of a value
and divides by `k`, which is exact in binary floating point exactly when `k` is a
power of two. Every exact tiling qualifies, since all its counts are 1.

**Accumulator.** ADR 0004 measured the sentence above to carry an unstated
premise, which is the accumulator the library happens to use. Over 76 legal
geometries on float32 input, a float32 accumulator is exact on 45, and those 45
are precisely the power-of-two ones. A float64 accumulator is exact on all 76.
The predicate is therefore a property of the geometry **and** the accumulator,
and a regime row must say which accumulator it holds for.

**Dtype, and it runs the opposite way to intuition.** In a geometry where
float32 and float64 are inexact, `float16` and `bfloat16` are exact, because half
precision is promoted to a float32 accumulator before the fold and a short
mantissa times a small count still lands inside 24 bits. No per-function label
can express an ordering in which the lower precision is the exact one.

**Input values, and this is why the suite missed the error.** The round-trip
tests built their images from `torch.arange`, and integer-valued data comes back
exact in regimes where random data does not. Any test asserting exactness must
use random full-mantissa data in the declared dtype, or it is a false-negative
generator.

## 3. The refutation this ADR exists to prevent from returning

An earlier revision proposed `k_max <= 4` as the predicate. It is false, and the
measurement is recorded here so that it cannot come back.

It coincides with the correct predicate on square patches, where a count of 3
cannot occur, and fails on rectangular ones, where it can. Swept over 14,969
rectangular full-coverage geometries: `k_max <= 4` mispredicts 3,936 of them,
26 percent, while the power-of-two predicate mispredicts 8. Those 8 fail in the
safe direction: the power-of-two predicate has **zero false positives**, never
promising exactness and then failing, and its 8 misses are geometries that happen
to be exact without having been promised.

**A contract may under-promise. It may not over-promise.** That sentence is the
norm this ADR contributes, and the project has already used it once, to retract
the wrong predicate in public.

## 4. What is deferred, and until when

The original draft also proposed a three-letter vocabulary (`R1` bit-exact, `R2`
measured-approximate, `R3` irreversible), a naming rule reserving `inverse_*`,
`un*` and `de*` for exact inverses, and a policy sending entirely-inexact
features to auxiliary packages.

None of it is wrong. All of it was written for six transforms that do not exist,
and it has been used nowhere in a month. It is **deferred until the first
transform that needs it**, which is the first operation offering an inverse that
is not exact. At that point this ADR is amended or a successor is written, with
the vocabulary applied to a real case rather than to six hypothetical ones.

Until then the naming rule is worth keeping as a one-line habit rather than a
frozen policy: **do not give an approximate inverse a name that promises an exact
one**. `rotate_back`, never `derotate`.

## 5. Consequences

- **Nothing in the code changes.** No public name moves, no call that worked
  stops working, no new exception is raised. What this ratifies is how the
  contract is written.
- **Inexact operations are not restricted.** `resize` and `stitch` with a window
  are exactly the operations users come for. What is forbidden is shipping one
  whose docstring implies an exactness it does not have.
- **The 1.0 freeze gains a rule and not a vocabulary.** [`FOCO-1.0.md`](../FOCO-1.0.md)
  makes accepting this ADR part of the freeze; what gets frozen is the
  requirement that every exactness claim name its regime and be caller-evaluable.
- **Tests mirror the regime, not the function.** An exactness claim is asserted
  by a sweep over the regime's own condition, on random full-mantissa data.
  `tests/test_exactness.py` is that, and it enumerates the space independently of
  the predicate so it can falsify rather than confirm.
- **Integers are outside this predicate.** For unmodified patches the integer sum
  is exactly `k*v` and `k*v // k == v` for every `k`, so an integer regime needs
  its own condition, which is that the accumulator does not overflow. ADR 0004
  carries that.
- **CPU verified only.** Every measurement here was taken on CPU, because no CUDA
  path in this library has ever executed. The power-of-two boundary is argued
  from IEEE arithmetic rather than from an accumulation order, so it should hold
  anywhere, but the caveat stays until a CUDA sweep confirms it.
