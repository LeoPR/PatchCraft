# Performance

Five of the six wheels PatchCraft publishes carry a Rust kernel for one hot
path. This page is the measurement behind that, and it is technology
information rather than usage guidance: nothing here changes how you call the
library, and the manual is [GUIDE.md](GUIDE.md).

## What is measured

The accelerator touches exactly one thing, which is the overlapping fold
inside `reconstruct` and `stitch`. That is where the time goes when patches
overlap, because `F.fold` does the scatter-add serially.

Non-overlapping geometries never reach it. Where `stride == patch_size` the
library takes a closed-form path that is already a pure rearrangement, so
there is nothing to accelerate and nothing below measures it.

## How to read the timings

They are the median of 25 runs after a warm-up call, on one machine, on one
torch build. The ratio depends on your CPU, on how many threads torch decides
to use, and on the geometry, so a ratio here is not a promise about your
machine. Run the script.

The comparison is PatchCraft against PatchCraft. There is no other library in
these tables, because the baseline that matters is the pure-torch path the
same install falls back to.

## Test machine

Intel Xeon E5-2697 v4 at 2.30 GHz, 36 logical cores, Windows.

## Versions

`patchcraft` 0.5.1, torch 2.14.0+cpu, Python 3.13.13. Measured 2026-09-03.

## The overlapping fold

| Geometry | Call | Pure torch | Accelerated | Speedup |
|---|---|---|---|---|
| 3x512x512, patch 32, stride 16 | `reconstruct` | 16.3 ms | 2.3 ms | 7.1x |
| 3x512x512, patch 32, stride 16 | `stitch, weight="hann"` | 16.2 ms | 6.3 ms | 2.6x |
| 3x1024x1024, patch 64, stride 32 | `reconstruct` | 53.5 ms | 7.8 ms | 6.9x |
| 3x1024x1024, patch 64, stride 32 | `stitch, weight="hann"` | 83.3 ms | 15.7 ms | 5.3x |
| 3x2048x2048, patch 64, stride 32 | `reconstruct` | 453.7 ms | 32.1 ms | 14.1x |
| 3x2048x2048, patch 64, stride 32 | `stitch, weight="hann"` | 460.9 ms | 37.9 ms | 12.2x |

The gain grows with the image because the kernel parallelizes over output
rows, and each output pixel is written by exactly one worker, so there are no
atomics and no contention.

**The gain is algorithmic rather than a thread count.** The obvious objection
is that the accelerator simply uses more cores than torch does, and it is
worth answering with a measurement. Forcing torch to 4, 8, 16 and 36 threads
on the largest case above moved the pure path between 365 ms and 465 ms, with
36 threads no better than 8. `F.fold` does not scale here with batch size 1,
which is the whole reason a native kernel was worth writing.

## Earlier gains, in pure torch

These are a different measurement, of different code, on an earlier release,
with no accelerator involved. They are listed separately on purpose and must
not be added to the table above.

Version 0.3.0 rewrote three paths in torch alone. `extract` takes a strided
window view instead of im2col plus a permute and a copy, measured 13 to 21
times faster on CPU and bit-exact. `reconstruct` skips `F.fold` entirely on
non-overlapping grids, since the operation there is a pure rearrangement,
measured 27 times faster, and it computes the overlap count map in closed form
in O(H+W) rather than folding a tensor of ones. `stitch` builds its denominator
from two 1-D window folds, because the kernels are separable.

## Reproducing

The script is in the repository and the numbers above are its output, pasted
rather than retyped:

```
python tools/benchmark.py
python tools/benchmark.py --markdown          # the table as it appears here
python tools/benchmark.py --repeats 50        # more samples
```

It runs every case twice, once with the accelerator and once with
`PATCHCRAFT_ACCEL=0`, and it compares the two results with `torch.equal`
**before** reporting any timing. If they differ it prints the table, says so,
and exits non-zero, because a benchmark of two different computations is not a
benchmark. That check is the reason to trust the ratios more than the ratios
themselves.

It exits early on an install with no accelerator, which is what the universal
wheel is, and says so.

## What these numbers do not say

**Nothing here was measured on a GPU.** No CUDA path in this library has ever
executed, in CI or anywhere else. The accelerator declines any tensor that is
not CPU-resident and torch handles it.

**One hot path only.** `extract`, `pair`, `resize`, the metrics and the cache
are untouched by the accelerator, and a pipeline dominated by any of those
will not move.

**Three of the five accelerated wheels have never had their kernel executed in
CI.** The equivalence test runs on Ubuntu and Windows x86_64. The macOS arm64,
macOS x86_64 and manylinux aarch64 wheels are built and their contents are
checked, and that is all.

**Timings from a development checkout are not these timings** unless the
extension was built in release mode. It is, since 0.5.1, and the reason the
setting is pinned in `setup.py` is that a debug kernel measured here turned a
14x speedup into 2.1x.

**I wrote both sides of this comparison.** The pure path and the accelerated
path are both mine, which removes the usual conflict of interest in a
benchmark and introduces a different one. Run the script.
