## What this changes, and why

<!-- One or two sentences. If it fixes a defect, say what the defect did with
     a number: this project's habit is to quote the measurement rather than
     describe the symptom. -->

## Does it change a returned value?

<!-- Delete the ones that do not apply. This decides the version bump, and the
     rule is in CONTRIBUTING.md: the question is not whether values changed but
     whether the contract moved or the implementation stopped violating it. -->

- [ ] No: documentation, tests, tooling or packaging only.
- [ ] No: new behaviour that breaks nothing.
- [ ] Yes, and the documentation always promised the new behaviour. A fix, so a `0.y.Z`.
- [ ] Yes, and behaviour the documentation endorsed has moved. A `0.Y.0`.

## Checks

- [ ] `ruff check src tests tools` and `mypy --strict src` pass.
- [ ] `pytest -m "not gpu"` passes.
- [ ] If it touches the exactness predicate, the fold, or the accelerator:
      `PATCHCRAFT_SWEEP_FULL=1 pytest tests/test_exactness.py` passes.
- [ ] If it touches `accel/`: `cargo test --manifest-path accel/Cargo.toml` passes.
- [ ] A test fails on the old code and passes on the new one. A fix without one
      is a fix that comes back.

## Documents that had to move

<!-- THEORY.md section 9 is the arbiter of the per-function contract, so a
     behaviour change usually starts there. CHANGELOG.md takes an entry with
     the measurement behind it. Say "none" if none, which is a real answer. -->
