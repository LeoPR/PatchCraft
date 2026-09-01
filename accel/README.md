# patchcraft-accel

Optional native accelerator for [patchcraft](https://github.com/LeoPR/PatchCraft):
a Rust/pyo3 kernel for the overlap fold in `reconstruct`/`stitch`.

## Install

```bash
pip install patchcraft[accel]
```

Prebuilt abi3 wheels (one per OS, covering Python 3.12+): Windows x64,
Linux x86_64 (manylinux), macOS arm64, macOS x86_64. Wheels are
self-contained (statically linked Rust; no system dependencies) and the
package never imports or links against torch.

## Build from source

Requires a Rust toolchain (https://rustup.rs) >= 1.82 and maturin:

```bash
pip install maturin
maturin develop --release   # from this directory, inside your virtualenv
```

## Debug

`PATCHCRAFT_ACCEL=0` forces the pure-torch path;
`patchcraft.accel_available()` reports whether the accelerator is active.
