# ADR 0003: reversibility classes for image and patch transforms

- **Status:** Proposed
- **Date:** 2026-08-04
- **Deciders:** Leonardo Marques de Souza
- **Amended by:** [ADR 0004](0004-precision-and-effort-parameters.md), which measures the exactness predicate below to be a property of the geometry **and the accumulator**, not of the geometry alone: with a `float64` accumulator, 76 of 76 sampled legal geometries are bit-exact where a `float32` accumulator gives 45, and those 45 are exactly the power-of-two ones.
- **Relates to:** [ADR 0001](0001-patch-extraction-api.md), [`THEORY.md`](../THEORY.md) §0, §9, [`STUDIES/2026-08-04-patch-techniques.md`](../STUDIES/2026-08-04-patch-techniques.md) §4, §5

## Context

The expansion study (T1) surfaces a recurring design question that the current contract (THEORY §9) does not answer: several candidate transforms claim a "reversal" operation, but *reversal* means three different things depending on the transform:

1. `rot90` / flips reverse **bit-exactly** (`rot90(x, k)` then `rot90(_, -k mod 4)` returns the identical tensor, any dtype).
2. Arbitrary-angle rotation reverses only **approximately**: two interpolations (forward and inverse) degrade the signal, and the error depends on angle, resample mode and content. No parameter choice makes it exact.
3. Quantization is **irreversible** by construction (many-to-one map); the useful contract is a measured error bound, not a reversal.

Today, every PatchCraft primitive is either bit-exact (`extract`/`reconstruct` round-trip at `stride == patch_size`) or explicit about its approximation (`stitch` with non-uniform weights blends; `resize` interpolates and documents dtype cast semantics). Adding transforms that silently mix "exactly invertible" with "approximately invertible" with "lossy" would erode the library's central promise: the caller always knows what comes back.

The alternative considered was a per-function documentation note with no shared vocabulary. Rejected: the study lists at least six upcoming features where the distinction matters (rotation, flips, padding round-trip, DCT, quantization, dithering), and an ad-hoc note per function invites drift, the same disease the §9 contract sections exist to prevent.

## Decision

A reversibility class is declared **per regime**, not per function. Each transform declares a small
**regime table** in its docstring and in its THEORY §9 contract section; every row of that table
carries exactly one class, and the function as a whole carries none. One class per function is not
writable for this library: `stitch`'s class is selected by its `weight` kwarg, and `reconstruct`'s
depends on the patch geometry and on the dtype.

A row states the condition under which its class holds, in terms the caller can evaluate before
calling. The three classes are unchanged:

- **R1, bit-exact.** Within the row's stated condition, a paired inverse returns a bit-identical
  result. Enforced by a round-trip test that compares bit patterns, not by bare `torch.equal`:
  `torch.equal` is not reflexive on NaN, so a provably bit-identical round trip fails it (measured:
  `torch.equal` False while `view(torch.int32)` is identical). Compare the integer view, or use
  `allclose(equal_nan=True)` alongside a bitwise check.
- **R2, measured-approximate.** A paired inverse exists but loses information to interpolation or
  requantization. The row must state the measured round-trip error under stated conditions, in
  relative terms, and the function must not be marketed as an inverse.
- **R3, irreversible.** Many-to-one by construction. No inverse is provided or implied; the row
  names the metric the caller uses to evaluate the result.

The classification duty applies to **transforms**: public callables that take image or patch data
and return image or patch data. Today that is `extract`, `Patchify`, `reconstruct`, `stitch`,
`resize` and `pair`. It does not apply to pure geometry (`num_patches`, `tilings`, `scale_factor`,
`paired_tilings`), to metrics (`patch_metrics`, `per_patch_mse`, `per_patch_psnr`), to storage
(`Cache`) or to the data carriers (`TilingSpec`, `PairedTilingSpec`, `PatchPair`, `PatchMeta`,
`WeightKind`).

Naming carries the class, and this rule is unchanged: names that imply exact inversion
(`inverse_*`, `un*`, `de*`, reversal kwargs) are reserved for R1 rows. An R2 inverse gets an
explicitly approximate name (`rotate_back`, never `derotate`). R3 gets no inversion entry point.
A kwarg may select a row, as `weight` does, but must never be a switch that renames the class.

## Consequences

- **Core admits R2 rows; what it does not admit is an undeclared row.** "Core stays R1-only" was
  false about the code as shipped, since `reconstruct` has been core since 0.1 and is R2 over part
  of its accepted input space. The rule that holds is narrower: every core transform declares at
  least one R1 regime and names it as the recommended geometry, and every other row it accepts is
  declared with its class. Features whose *entire* table is R2 or R3 still belong in auxiliary
  packages (`patchcraft-quant` for quantization, arbitrary-angle rotation where it lands),
  consistent with the consumer gate.
- **The `reconstruct` claim in the previous draft was wrong, and the real boundary is a property of
  the whole count map, not of its maximum.** The round trip is bit-exact exactly when **every value
  in the count map is a power of two**. The reason is arithmetic rather than statistical: the fold
  sums `k` copies of the same value and divides by `k`, and that is exact in binary floating point
  when, and only when, `k` is a power of two. So `reconstruct` is R1 wherever the count map holds
  only powers of two, which includes every exact tiling (all counts are 1), and R2 otherwise
  (`H=16, p=4, s=1`, counts up to 16 but including 9, max_abs 2.4e-07 in float32;
  `H=32, p=8, s=1`, 8.9e-07).

  An earlier revision of this ADR proposed `k_max <= 4` instead. That predicate is false and is
  recorded here so it does not return: it coincides with the correct one on square patches, where
  a count of 3 cannot occur, and it fails on rectangular ones, where it can. Swept over 14969
  rectangular full-coverage geometries with `k_max <= 4`, `k_max <= 4` mispredicts 3936 of them
  (26 percent) while the power-of-two predicate mispredicts 8. Those 8 are the safe direction: the
  power-of-two predicate has **zero false positives**, meaning it never promises exactness and then
  fails, and its 8 misses are geometries that happen to be exact without being promised. A contract
  may under-promise. It may not over-promise.

  `stitch(weight="uniform")` inherits this table; `hann` and `gaussian` make no inversion claim and
  are R3 rows carrying a metric.
- **Dtype is a regime axis, and it runs the opposite way to intuition.** In the same geometry where
  float32 and float64 are R2, float16 and bfloat16 are bit-exact, because 0.2.1 promotes half
  precision to a float32 accumulator before the fold and an 11-bit mantissa times a small count
  still lands exactly inside 24 bits. No per-function class can express an ordering where lower
  precision is the exact one.
- **Input values are a third regime axis, and this is why the suite missed it.**
  `tests/test_reconstruct.py` builds its images from `_ramp` (`torch.arange`, integer-valued), and
  integer-valued data round-trips bit-exactly in regimes where `U(0, 1)` data does not (measured on
  the same geometry: ramp `torch.equal` True, random False). Any test enforcing an R1 row must use
  random full-mantissa data in the declared dtype, or it is a false-negative generator. Fixing that
  test data is a 0.3 task, not a 0.2.x one, since it changes what the suite asserts.
- **Lab tasks stay falsification instruments.** Before any R2 feature ships, a lab script publishes
  its measured round-trip error curve in relative terms, and that measurement goes in the row. T7
  exists for this.
- **Tests mirror the row, not the function.** R1 rows assert a bitwise round-trip over a sweep of
  the row's own condition; R2 rows assert the stated relative error; R3 rows assert that the metric
  is computed and reported.
- **No retroactive renames and no 0.2.x behaviour change.** No public name moves, no call that
  worked stops working, no new exception is raised. What changes is documentation and, at 0.3,
  test data. `reconstruct` keeps its name: the naming rule governs the `inverse_*`, `un*` and `de*`
  forms, none of which it uses.
- **CPU verified only.** Every measurement quoted here was taken on CPU, because the development
  machine has no working CUDA build. The power-of-two boundary is argued from IEEE (summing k
  identical addends and dividing by k is exact when k is a power of two) rather than from an
  accumulation order, so it should hold anywhere, but the tag stays until a CUDA sweep confirms it.
