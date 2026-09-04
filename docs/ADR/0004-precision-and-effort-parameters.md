# ADR 0004: naming the precision policy, and an effort preset that may not change values

- **Status:** Proposed
- **Date:** 2026-09-03
- **Deciders:** Leonardo Marques de Souza
- **Relates to:** [ADR 0003](0003-reversibility-classes.md) (which this amends on one point), [`STUDIES/2026-09-03-precision-measurements.md`](../STUDIES/2026-09-03-precision-measurements.md) (every measurement cited here), [`FOCO-1.0.md`](../FOCO-1.0.md) §1

## Context, in four sentences

The library already decides precision for the caller in five places, none of them visible
from a call site: half precision is promoted to a `float32` accumulator, `stitch` floors its
denominator, `metrics` accumulates in `float64` whatever came in, and `resize` routes every
cross-backend conversion through a normalised `float32`. So a knob named `accum_dtype`
introduces no concept; it exposes one that is already applied by default.

What forced the question is that a `uint8` user cannot use the library at all, and the only
route is `.float()` first, at four times the memory. The owner's framing rules out the easy
answer: `input(byte) -> operation(???) -> output(byte)` is not the same thing as
`input(byte) -> operation(byte) -> output(byte)`, because returning the input dtype after
widening internally gives back neither the memory nor the time.

Every number this decision rests on is measured and lives in
[`STUDIES/2026-09-03-precision-measurements.md`](../STUDIES/2026-09-03-precision-measurements.md).

## What is decided here, and what is not

This document mixes two kinds of statement, and separating them is the point of the
rescoping.

**The names are not a decision.** Which identifier a parameter carries is a fact about the
ecosystem the caller already has imported, not a preference of this project. Choosing one by
taste would be guessing, and a guess on a frozen surface is permanent. So the names below are
**determined by precedent** and verified against the installed libraries, not proposed for
approval. Section 1 states them; there is nothing to approve there.

**What is a decision is whether to add any of them, and when.** Three of the steps in the
effort table put a new keyword on a surface that 1.0 freezes, and each is a permanent
commitment. The order also embeds a real choice, which section 4 states plainly rather than
burying. That is what this ADR is asking, and it is asking it later rather than now.

**Nothing here is implemented.**

## 1. The names, determined by precedent

| Name | Type | Values | Default | Governs | Borrowed from |
|---|---|---|---|---|---|
| `accum_dtype` | `torch.dtype \| None`, keyword-only | any dtype at least as wide as the input, of the same kind | `None`, meaning today's rule | where the numerator and count map live in `reconstruct` and `stitch`. Does not change the return dtype. | `numpy.sum(dtype=)`, documented as "the accumulator in which the elements are summed", and `torch.sum(dtype=)` |
| `rounding_mode` | `Literal["trunc","floor","nearest_even"] \| None` | as listed | `None` | the single rounding event on `stitch`'s final divide, once `stitch` accepts integers. Not offered on `reconstruct`. | `torch.div(rounding_mode=)`, same name, two of three values; the tie-break vocabulary from stdlib `decimal` |
| `antialias` | `bool \| None` | `True`, `False`, `None` | `None`, meaning whatever the backend does today | whether the torch `resize` backend prefilters before downsampling | `torch.nn.functional.interpolate` and `torchvision.transforms.v2.functional.resize`, verbatim |

Verified against the installed libraries on 2026-09-03, so "borrowed" is a checked fact:
`numpy.sum` and `torch.sum` both take `dtype=`, `torch.div` takes `rounding_mode=` and
documents exactly `None`, `"trunc"` and `"floor"`, and both `torch.nn.functional.interpolate`
and `torchvision.transforms.v2.functional.resize` take `antialias=`. `numpy.ufunc.accumulate`
also exists, which is the collision that rules out `accumulate=`.

`None` is a sentinel on every one of them, never a spelled-out default. Without it a preset
cannot distinguish "unspecified" from "explicitly asked for the current value", and the
resolver has nothing to key on.

**Not `accumulate=`.** It is a verb where every precedent is a noun, it does not say the
value is a dtype, and `numpy` already owns the identifier for `ufunc.accumulate`, which is a
running scan and a different operation.

**No `rounding_mode` on `reconstruct`.** For unmodified patches every covering value is the
same `v`, so the sum is exactly `k*v` and `k*v // k == v` for every `k`, in every dtype.
`reconstruct` does not round. A knob there would advertise a decision the function does not
make.

## 2. The preset, and the one constraint that makes it safe

`effort: Literal["balanced", "fast"] | None = None`, keyword-only, on `reconstruct`,
`stitch`, `extract` and `Patchify.__init__`. `Patchify` is the reason it exists: inside a
`Compose` there is no per-call keyword, so a stored string is the only way a policy reaches
a data pipeline.

One axis, following x264 and x265, whose presets are all speed and whose quality is a
separate orthogonal knob. Not `fast`/`quality`/`precise`: that trio spans three axes and
admits no correct ordering, and it leaves the caller who wants exact-and-fast, which is the
common case here, nothing to pass. No superlative at either end, so a third rung can be
added later without a lie, which is the wall both `torch` ("highest") and JAX (`HIGHEST`)
hit.

**The constraint, and it is the whole decision:**

> `effort=` may change speed and memory. It may never change a returned value. Anything
> that changes values requires an explicit knob.

That constraint is what separates this from `-ffast-math`, which is structurally the same
proposal: a name promising speed whose effect changes results. Its failure modes are all
reachable here in miniature. It deletes the caller's own safety checks, a compiler update
silently reinterpreted it more aggressively and broke shipping software with no source
change, and it escapes the caller's scope entirely by setting process-wide floating point
mode. The constraint also makes the preset falsifiable, which is what this project's
contract style demands: "fast" is unprovable, but "`effort="fast"` is bit-identical to
`effort="balanced"`" is a test id.

`effort="fast"` may therefore expand only into: permitting the native accelerator without
an environment variable, which structurally replaces the `PATCHCRAFT_ACCEL` global that is
already an unnamed preset today; permitting the shift-and-mask integer path where the
coverage map makes it bit-identical to the divide; and returning views where aliasing is
provably safe, which is a memory choice and needs its own sub-knob rather than hiding here.
`effort="balanced"` expands to nothing at all: it is today's behaviour, exhaustively.

Ship `patchcraft.effort_options(name) -> dict` returning the literal expansion, after
`torch._inductor.list_mode_options`, so the preset is never a black box and the knobs stay
the source of truth.

**Precedence: the preset seeds defaults, explicit keyword arguments override, and argument
order is irrelevant.** After x265, which promises exactly that in prose. Not mutual
exclusion: `torch.compile` raises on preset-plus-options and the bill is visible, because
`max-autotune-no-cudagraphs` exists as a whole extra public name purely to work around it.

## 3. Amendment to ADR 0003

**The exactness predicate is a property of the geometry *and the accumulator*, not of the
geometry alone.** ADR 0003 states the round trip is bit-exact iff every coverage count is a
power of two. Measured over 76 legal square geometries on `float32` input: with a `float32`
accumulator, 45 of 76 are exact, and those 45 are exactly the power-of-two geometries, so
the predicate is perfectly predictive. With a `float64` accumulator, **76 of 76 are exact**.

The predicate as written carries an unstated premise, which is the accumulator the library
happens to use today. ADR 0003 should say so. It also means `accum_dtype=torch.float64`
would offer unconditional bit-exactness for `float32` input at twice the accumulator memory,
which is a stronger guarantee than the library currently advertises.

**And the predicate does not govern integers at all.** A `uint8` round trip through an
`int32` accumulator was exact on all six geometries tested, including three whose count maps
are not powers of two, for the reason given above: the sum is exactly `k*v`. Gating an
integer path on the float predicate would be a category error.

## 4. The decision being asked: whether, and when

### Effort, each step independently shippable

| Step | Size | Breaks frozen surface | When |
|---|---|---|---|
| 0. Record this ADR, and qualify the four copies of "widening the dtype does not help" | 0.5 to 1 day | no | before 1.0, and it is the only item that must be |
| A. Fix three defects as bugs, no knob | done, 2026-09-03 | no signature change | done before 1.0, as planned |
| 1. `extract` accepts the five integer dtypes, by adding them to the fast-path set | ~1 day | no | after 1.0, never alone |
| 2. Exact integer `reconstruct`, unrestricted, `int32` accumulator, every legal geometry | 4 to 6 days | no signature change | after 1.0, shipped with step 1 as one 1.1 |
| 3. Byte-native `uint8` quotient-and-remainder path | 2 to 3 days | no | do not build yet: 1.7x over an `int32` accumulator, at the cost of a second, different predicate on the doc surface |
| 4. `accum_dtype=` on `reconstruct` and `stitch` | 1 day, 2 alongside step 2 | yes, signature | after 1.0 |
| 5. `rounding_mode=` on `stitch` | 1 day | yes, signature | after 1.0, and only if a real caller appears |
| 6. `effort=` and `effort_options()` | 1 day of code, 3+ of contract | yes, signature, four sites | after 1.0, after 1 and 2 have shipped |
| 7. Bool support, where the inverse is a majority vote | 3 to 4 days | it is a 21st public name | 1.1 at the earliest, a function rather than a parameter |

### Order, and the choice inside it

The order below embeds a real alternative, and it is the substance of what is being asked.
It recommends freezing 1.0 at 20 names with no new parameters, and only then waiting for a
user to appear. **The defensible opposite is to add the knobs before the freeze**, because
after it each one costs a minor release. Neither is settled by any measurement in the study.

Record the design. Fix the three defects as bugs on their own merits, because folding them
into a `precise` tier would sell the absence of a bug as a quality level. **Step A landed on
2026-09-03**, and the three were the `stitch` denominator floor, the `reconstruct`
single-patch aliasing and the `resize` inbound clamp on a dtype that cannot hold the bound.
The remaining items named in the audit, the `bfloat16` justification in THEORY 9.2 and the
resize antialias gap, are still open and are not defects of the same kind: the first is a
documentation correction and the second is a user-facing choice that needs its own release. Document, at zero
cost, the rounding policy already in force. Freeze 1.0 at 20 names with no new parameters.
Then 1.1 brings steps 1 and 2 together and unrestricted. Then stop and wait for a user:
only if one appears do the knobs follow, and the preset last of all.

### What not to do

- Do not gate the integer path on the power-of-two predicate. It is a float predicate.
- Do not let any preset change a returned value.
- Do not ship step 1 without step 2. `extract` accepting `uint8` while `reconstruct` rejects
  it makes THEORY §9.1 and §9.2 false as the matched pair they are written as.
- Do not make any of this a module global, an environment variable, or a context manager
  that mutates torch state.
- Do not offer `rounding="nearest"` without saying which tie-break.
- Do not add a knob to `accel_available()`. It is one of the 20 frozen names with a pinned
  zero-argument signature.

### Open questions, for the decision this ADR defers

1. Does `accum_dtype` change only where the sum lives, or also what comes back?
   `reconstruct` casts back to `patches.dtype`; `metrics` returns `float64` unconditionally.
   The two are inconsistent today and the knob would expose it.
2. If `accum_dtype=float64` makes every geometry exact, does the headline contract become
   "exact, at this accumulator width" rather than "exact on these geometries"? That reads
   as a stronger promise and needs its own falsification suite before it is made.
3. Are steps 1 to 3 a `0.Y.0` or a `z` bump? The project shipped the mirror-image change
   once: 0.2.1 added the `reconstruct` integer guard, turning a raw `NotImplementedError`
   into a framed `ValueError`.
4. Integers with `dilation > 1`: a framed `ValueError`, or torch's raw error in that corner?
5. CUDA. Everything here was measured on CPU, and the accelerator refuses non-CPU tensors
   outright. No accept text may say "integers work" without a device qualifier.
