# The patchcraft accelerator

A Rust/pyo3 kernel for the overlapping fold in `reconstruct` and `stitch`,
compiled into the `patchcraft` wheel as `patchcraft._accel_native`.

This is not a separate package. There is nothing to install for it and no
extra to enable. `pip install patchcraft` brings it on Windows x64, Linux
x86_64 and aarch64, and both macOS architectures; every other platform gets
the universal wheel and runs the same operation in torch.

## Is it active here

```bash
python -c "import patchcraft; print(patchcraft.accel_available())"
```

`PATCHCRAFT_ACCEL=0` in the environment forces the pure path at runtime, which
is the switch to reach for when comparing the two.

## Build it in a checkout

Requires a Rust toolchain (<https://rustup.rs>) at 1.82 or newer. An editable
install compiles it automatically when cargo is on PATH:

```bash
uv sync                 # or: pip install -e .
```

Set `PATCHCRAFT_REQUIRE_EXTENSION=1` to make a build failure fatal rather than
a silent fall back to the pure path, and `PATCHCRAFT_PURE_PYTHON=1` to skip the
extension entirely. If your checkout sits under a path with non-ASCII
characters, point cargo at an ASCII build directory with
`CARGO_TARGET_DIR=/tmp/pc-target`, because setuptools-rust mis-decodes the
artifact path otherwise.

The kernel has its own tests, which need no Python:

```bash
cargo test --manifest-path accel/Cargo.toml
```

## What it may and may not do

It takes only a plain CPU tensor, float32 or float64, and returns `None` for
anything else, so that torch handles the call: a tensor that requires grad or
is a forward-mode dual, a tensor subclass, a tensor with the negative bit set,
a call under `torch.jit.trace` or a dispatch mode such as `make_fx`, with or
without `pre_dispatch`, a tensor whose `data_ptr()` raises or is null, as
`torch.vmap`, `torch.func` transforms, `FakeTensorMode` and wrapper
subclasses produce, and a tensor whose pointer is misaligned, as a view of a
byte buffer at an odd offset can be. The native entry point itself also
refuses a null or misaligned pointer with `ValueError`. It never raises
for its own absence. Its summation order per output pixel matches
ATen `col2im`, which is what makes the two paths bit-identical rather than
merely close.
