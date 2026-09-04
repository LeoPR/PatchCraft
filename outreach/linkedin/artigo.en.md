<!-- l10n: doc_id=patchcraft-outreach-linkedin-artigo · lang=en · translation_of=artigo.pt-BR.md · source_lang=pt-BR -->
**English** · [Português](artigo.pt-BR.md)

# Cutting an image into pieces and putting it back without losing pixels on the way

*Technical article. Every number here has a command that reproduces it in the repository.
Where the library does not help, the text says it does not help.*

---

To a computer an image is a matrix of numbers, and working on the image means doing
arithmetic on that matrix. A large image rarely goes into a neural network whole: it is cut
into pieces, each piece is processed, and at the end everything is glued back. The pieces
are called patches.

PyTorch ships `unfold` and `fold` for this, and for simple cuts the two are enough. Beyond
that case, details start asking for care, and two of them return the wrong image without
raising anything. This text is about those details and about what PatchCraft does with
each.

## Two defects that never announce themselves

The first one lives in the reshape. Torch's `F.unfold` returns a `(1, C*ph*pw, L)` tensor,
and the intuitive reshape into `(L, C, ph, pw)` hands you the right shape with the wrong
pixels. The shape assertion passes, training runs, the loss comes down a little less, and
there is no error message anywhere.

The second lives in the stride. On a 128 by 128 image with patch 32 and stride 20, the grid
stops at pixel 112 and leaves 3840 of the 16384 pixels at zero. A hand-rolled `fold` returns
that partly black image without complaining.

Neither is hard to fix. Both are easy to miss, and that difference is what justifies writing
it once, with tests around it, rather than rewriting it per project.

## The numeric contract, and why it is evaluable before the call

The library makes a numerical claim: under what conditions the round trip returns the same
tensor, bit for bit.

> The round trip is exact if and only if **every** value in the coverage count map is a
> power of two. Outside that, the per-pixel error is bounded by `(k+1)·eps·|v|`, with `k`
> that pixel's coverage count.

There is one reason for it: dividing a float by a power of two is the one division that
never rounds. `stride == patch_size` always satisfies it, because every count is 1.

What makes this a contract rather than a promise is the second half: the condition is
computed from the geometry alone, without running anything. The caller knows which regime
they are in beforehand.

The rule looks severe, and the obvious alternative is looser: just keep the maximum overlap
count small. It does not work, and the gap is wide. Over a sweep of 14,969 rectangular
geometries, the maximum rule mispredicts 3,936 cases; the power-of-two rule mispredicts 8,
and those 8 err by promising less than they deliver.

That asymmetry is what settles which of the two is worth having. A contract may
under-promise. It may not over-promise, because whoever trusts it has no way to notice the
difference: outside the rule the per-pixel error grows with the coverage and reaches 19 ULP
in float32 with nothing to signal it.

Notice that the previous version looked at the maximum of the map and the correct one looks
at all of its values. I tested both against a sweep of rectangular geometries: the maximum
rule mispredicts 3936 of 14969 cases, the power-of-two rule mispredicts 8. Those 8 fail in
the safe direction, promising less than they deliver. A contract may under-promise. It may
not over-promise.

## The test that exists to bring the claim down

The contract above has, in the suite, a test whose explicit job is to falsify it.

It enumerates the 126,736 legal geometries of the space without consulting the predicate,
so that the enumerator and the thing under test are independent. Over a seeded sample it
hunts for the two counterexamples that would break the contract: a case inside the rule
that is not exact, and a case outside the rule that is exact across every seed, which would
say the predicate is drawn too tight. The full sweep of all 126,736 sits behind an
environment variable, because it takes a little over a minute.

Notice that it is the same silent failure as the opening, one layer up. The code was right;
the guarantee about the code was wrong, and it passed the tests for the same reason the two
defects at the start did: the question being asked was not the one that mattered.

One detail of the data generator is worth carrying anywhere patches get tested: the images
are full-mantissa noise drawn directly in the target dtype, never an integer ramp. Integer
data in a float round-trips exactly on geometries where random data does not, so a suite
built with `torch.arange` passes without verifying anything.

I think a numerical library is worth less for the guarantee it announces and more for the
test it keeps pointed at that guarantee.

## The accelerator, and the objection it deserves

The hot path is the overlapping fold, which is where the time goes when patches overlap.
Five of the six wheels carry a Rust kernel for it, compiled into the wheel itself, with no
extra to enable. Every other platform gets the universal wheel and runs the torch path,
which returns the same values.

| Geometry | Call | Pure torch | Accelerated | Speedup |
|---|---|---|---|---|
| 3x512x512, patch 32, stride 16 | `reconstruct` | 16.3 ms | 2.3 ms | 7.1x |
| 3x1024x1024, patch 64, stride 32 | `reconstruct` | 53.5 ms | 7.8 ms | 6.9x |
| 3x2048x2048, patch 64, stride 32 | `reconstruct` | 453.7 ms | 32.1 ms | 14.1x |
| 3x2048x2048, patch 64, stride 32 | `stitch` hann | 460.9 ms | 37.9 ms | 12.2x |

The obvious objection is that the kernel simply uses more cores than torch does. That
deserves a measurement rather than an answer. Forcing torch to 4, 8, 16 and 36 threads on
the largest case, the pure path lands between 365 ms and 465 ms, and 36 threads is no better
than 8. `F.fold` does not scale at batch size 1, and that is why the kernel exists.

The benchmark runs each case twice, once with the accelerator and once with it disabled, and
compares the two results with `torch.equal` **before** reporting any timing. If they differ
it prints the table, says so, and exits non-zero. A benchmark of two different computations
is not a benchmark.

One trap worth knowing if you benchmark a native kernel: an editable install compiles in
debug by default, because the tooling follows the build command unless told otherwise. Here
that turned a 14x speedup into 2.1x, and nothing in the output tells you which binary is
running.

## What the project does not claim

No external project has consumed the library yet, and that is the criterion it chose for
itself before calling the shape settled.

No CUDA path in it has ever executed, in CI or anywhere else. The accelerator declines any
tensor that is not on the CPU and hands the work back to torch.

Three of the five accelerated wheels have never had their kernel executed in CI. The macOS
and aarch64 ones are built and have their contents checked, and that is all.

And every claim on this page is measured, which is not the same as proven. It is reasonable
to assume there is one I have not measured yet.

## Practical

Python 3.12 to 3.14, torch 2.6 or newer, MIT, pre-1.0. 1619 tests pass and 1656 are
collected, with CI on {Ubuntu, Windows} x {3.12, 3.13, 3.14} plus an accelerated job on
both systems. The public
surface is 20 names, frozen by test.

```
pip install patchcraft
```

The code, the measurements and the documentation of what does not work are open.

https://github.com/LeoPR/PatchCraft

#Python #PyTorch #OpenSource #ComputerVision #SoftwareEngineering
