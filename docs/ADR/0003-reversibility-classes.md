# ADR 0003: reversibility classes for image and patch transforms

- **Status:** Proposed
- **Date:** 2026-08-04
- **Deciders:** Leonardo Marques de Souza
- **Relates to:** [ADR 0001](0001-patch-extraction-api.md), [`THEORY.md`](../THEORY.md) §0, §9, [`STUDIES/2026-08-04-patch-techniques.md`](../STUDIES/2026-08-04-patch-techniques.md) §4, §5

## Context

The expansion study (T1) surfaces a recurring design question that the current contract (THEORY §9) does not answer: several candidate transforms claim a "reversal" operation, but *reversal* means three different things depending on the transform:

1. `rot90` / flips reverse **bit-exactly** (`rot90(x, k)` then `rot90(_, -k mod 4)` returns the identical tensor, any dtype).
2. Arbitrary-angle rotation reverses only **approximately**: two interpolations (forward and inverse) degrade the signal, and the error depends on angle, resample mode and content. No parameter choice makes it exact.
3. Quantization is **irreversible** by construction (many-to-one map); the useful contract is a measured error bound, not a reversal.

Today, every PatchCraft primitive is either bit-exact (`extract`/`reconstruct` round-trip at `stride == patch_size`) or explicit about its approximation (`stitch` with non-uniform weights blends; `resize` interpolates and documents dtype cast semantics). Adding transforms that silently mix "exactly invertible" with "approximately invertible" with "lossy" would erode the library's central promise: the caller always knows what comes back.

The alternative considered was a per-function documentation note with no shared vocabulary. Rejected: the study lists at least six upcoming features where the distinction matters (rotation, flips, padding round-trip, DCT, quantization, dithering), and an ad-hoc note per function invites drift, the same disease the §9 contract sections exist to prevent.

## Decision

Every transform added to PatchCraft (core or auxiliary packages) declares exactly one of three **reversibility classes** in its docstring and in its THEORY §9 contract section:

- **R1, bit-exact.** There exists a paired inverse such that `inverse(forward(x)) is bit-identical to x` for every supported dtype and geometry, enforced by a `torch.equal` round-trip test. Examples: `extract`/`reconstruct` at exact tilings, `rot90`, flips, pad/crop round-trip.
- **R2, measured-approximate.** A paired inverse exists but loses information to interpolation or requantization; the contract requires the round-trip error to be *measured and reported* (PSNR/max-abs against the input under stated conditions), and the function must not be marketed as an inverse. Example: arbitrary-angle `rotate`.
- **R3, irreversible.** Many-to-one by construction. No inverse is provided or implied; the contract instead documents the error metric the caller should use to evaluate the result (e.g. `patch_metrics` after quantization).

Naming carries the class: functions whose names imply inversion (`inverse_*`, `un*`, `de*`, reversal kwargs) are reserved for R1. R2 inversions get explicit approximate names (`rotate_back`, never `derotate`). R3 gets no inversion entry point at all.

## Consequences

- **Core stays R1-only.** `patchcraft` keeps its bit-exact character: only transforms with an exact inverse (or no inverse claim, like `resize` whose contract never claimed one) are core candidates. R2/R3 features land in auxiliary packages (`patchcraft-quant` for quantization; rotation-arbitrária onde couber), consistent with the consumer gate.
- **Lab tasks become falsification instruments.** Before any R2 feature ships, a lab script must publish its measured round-trip error curve (e.g. PSNR vs. angle), and that measurement goes in the doc. T7 exists precisely for this.
- **Tests mirror the class.** R1 → `torch.equal`; R2 → threshold assertions on measured error; R3 → error-metric assertions only.
- **No retroactive renames.** Existing functions already satisfy their implied class: `reconstruct` is R1, `stitch` makes no inversion claim, `resize` makes no inversion claim. Nothing changes in 0.2.x behavior.
