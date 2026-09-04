# Security policy

## Reporting

Report a suspected vulnerability privately through GitHub's
[security advisories](https://github.com/LeoPR/PatchCraft/security/advisories/new),
or by email to leonardo.marques.souza@gmail.com.

Please do not open a public issue for something you believe is exploitable.
For anything else, including a wrong numeric result, a public issue is the
right place: a wrong answer is a correctness bug here, and this project prefers
to discuss those in the open with the measurement attached.

This is a one-maintainer project and there is no service-level agreement. A
report will be acknowledged, and if it holds it will be fixed in the next
release with the entry naming what was wrong.

## Supported versions

The most recent release, only. The project is pre-1.0 and there are no
maintenance branches; a fix goes into the next version rather than being
backported.

## What the attack surface actually is

Boilerplate would be less useful than naming the real edges, so here they are.

**The native kernel writes through raw pointers.** On the platforms whose wheel
carries it, `reconstruct` and `stitch` hand `data_ptr()` values to Rust and the
kernel writes the output buffer directly. It does not re-validate the geometry:
the Python side validates with `check_fold_geometry` first, and the kernel
trusts that. This is the sharpest edge in the library. A defect that let an
unvalidated geometry reach `patchcraft._accel.fold_weighted` would be an
out-of-bounds write rather than an exception, and it is the thing most worth
reporting. `PATCHCRAFT_ACCEL=0` disables the native path entirely, and the
universal wheel does not contain it at all.

**`Cache` uses the path you give it, and validates nothing beyond expanding
`~`.** That is deliberate and it is what every established library does with a
caller-supplied cache directory: `torch.hub.set_dir` calls `expanduser` and
stops, pytest's cacheprovider does not validate, pip passes its `--cache-dir`
through. A cache location is an argument the calling code chooses, not data
read from an untrusted source, so `Cache(root, namespace="../elsewhere")` will
write outside `root` and that is the caller's decision.

The boundary, therefore: **do not derive a cache root or namespace from
untrusted input.** If you need per-user or per-request caches, hash or
otherwise sanitise the value in the layer that has the context to do it. This
library is not the place for path policing, and adding it here would be
inventing a security mechanism where the ecosystem has a settled convention.

**Nothing in the shipped package opens a network connection.** The dataset
helpers that download anything live in `tests/`, which is not part of the
wheel.

**Image decoding happens before this library sees anything.** `resize` accepts
a `PIL.Image` or a tensor. If you decode untrusted image files, the exposure is
Pillow's, and Pillow is the project to watch and to keep current.

**`Cache` stores and returns opaque bytes.** `put` takes `bytes` and `get`
returns `bytes`, so the library never deserialises anything itself and adds no
pickle surface of its own. The exposure is whatever *you* do with those bytes:
if you feed them to `torch.load` or `pickle`, then a cache root someone else
can write to becomes a code-execution path through your own deserialiser, not
through this one. A sidecar checksum is verified on read, which catches
corruption but is not a defence against an attacker who can write both files.

## What is not a security issue here

A wrong numeric result, a geometry accepted that should have been rejected, or
a documented claim that turns out to be false. Those are correctness bugs, this
project has published and retracted such a claim before, and they belong in a
public issue with the measurement that shows the problem.
