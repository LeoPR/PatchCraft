"""Check that a dist directory holds the artifacts the release intends.

The project publishes two shapes of wheel for one version, and each shape has
a failure mode that no other gate catches.

A platform wheel is built with ``optional=True`` unless CI sets
``PATCHCRAFT_REQUIRE_EXTENSION``. If that variable is ever lost, a Rust
failure stops being fatal and the build still produces a wheel, tagged for the
platform, with no extension inside it. Installers prefer that wheel over the
universal one, so the fastest platforms would silently get the slowest build.

The universal wheel has the mirror problem: a stray extension in it would make
a file tagged ``py3-none-any`` platform-specific, and it would be installed on
machines it cannot load on.

It also guards the version, which setuptools-scm derives from the git tag
rather than from any literal in the source. Two things can go wrong there and
both are silent. A checkout without the tag in reach makes setuptools-scm fall
back to something like ``0.1.dev1+g1234567``, a version that walks backwards;
a dirty tree turns ``0.5.1`` into ``0.5.2.dev0+g1234567.d20260902``. Both carry
a local segment, which PyPI refuses outright ("it MUST NOT allow the use of
local version identifiers"), so the upload would fail late and confusingly.
Catching it here fails early and says why.

Run it against a directory of artifacts::

    python tools/check_dist.py dist/
    python tools/check_dist.py dist/ --expect-version 0.5.1
"""

from __future__ import annotations

import re
import sys
import tarfile
import zipfile
from pathlib import Path

EXT_STEM = "patchcraft/_accel_native"
# `patchcraft-0.5.1-cp312-abi3-win_amd64.whl` and `patchcraft-0.5.1.tar.gz`
# both put the version in the second dash-separated field.
_VERSION_RE = re.compile(r"^patchcraft-(?P<v>[^-]+?)(?:-.*)?\.(?:whl|tar\.gz)$")
SDIST_MUST_CARRY = ("accel/src/lib.rs", "accel/src/kernel.rs", "accel/Cargo.toml")


def _artifact_version(path: Path) -> str | None:
    name = path.name
    if name.endswith(".tar.gz"):
        if not name.startswith("patchcraft-"):
            return None
        return name[len("patchcraft-") : -len(".tar.gz")]
    m = _VERSION_RE.match(name)
    return m.group("v") if m else None


def _check_versions(paths: list[Path], expected: str | None, problems: list[str]) -> None:
    seen: dict[str, list[str]] = {}
    for p in paths:
        v = _artifact_version(p)
        if v is None:
            problems.append(f"{p.name}: cannot read a version out of the filename")
            continue
        seen.setdefault(v, []).append(p.name)
        if "+" in v:
            problems.append(
                f"{p.name}: version {v!r} carries a local segment, which PyPI refuses. "
                "setuptools-scm produces this from a dirty tree or a checkout with no "
                "tag in reach (use fetch-depth: 0)"
            )
    if len(seen) > 1:
        problems.append(f"artifacts disagree about the version: {dict(seen)}")
    if expected is not None:
        wrong = sorted(v for v in seen if v != expected)
        if wrong:
            problems.append(f"expected version {expected!r} but the artifacts say {wrong}")


def _wheel_has_extension(path: Path) -> bool:
    with zipfile.ZipFile(path) as zf:
        return any(n.startswith(EXT_STEM) for n in zf.namelist())


def _check_wheel(path: Path, problems: list[str]) -> str:
    universal = path.name.endswith("-py3-none-any.whl")
    has_ext = _wheel_has_extension(path)
    if universal and has_ext:
        problems.append(f"{path.name}: universal wheel carries a native extension")
    elif not universal and not has_ext:
        problems.append(
            f"{path.name}: platform wheel has no {EXT_STEM} inside it, so the "
            "Rust build degraded silently (is PATCHCRAFT_REQUIRE_EXTENSION set?)"
        )
    return "universal, pure" if universal else "platform, native"


def _check_sdist(path: Path, problems: list[str]) -> str:
    with tarfile.open(path) as tf:
        names = {n.split("/", 1)[-1] for n in tf.getnames()}
    missing = [w for w in SDIST_MUST_CARRY if w not in names]
    if missing:
        problems.append(f"{path.name}: sdist cannot build the extension, missing {missing}")
    leaked = sorted(n for n in names if n.startswith(("accel/target/", "lab/", ".superpowers/")))
    if leaked:
        problems.append(
            f"{path.name}: sdist carries {len(leaked)} file(s) it should not, "
            f"e.g. {leaked[0]}"
        )
    return f"sdist, {len(names)} entries"


def main(argv: list[str]) -> int:
    args = argv[1:]

    expected: str | None = None
    if "--expect-version" in args:
        i = args.index("--expect-version")
        if i + 1 >= len(args):
            print("--expect-version needs a value", file=sys.stderr)
            return 2
        expected = args[i + 1]
        del args[i : i + 2]

    # Only the complete set is required to hold a universal wheel. The native
    # job checks one platform wheel at a time and legitimately has none.
    require_universal = "--require-universal" in args
    if require_universal:
        args.remove("--require-universal")

    where = Path(args[0] if args else "dist")
    if not where.is_dir():
        print(f"not a directory: {where}", file=sys.stderr)
        return 2

    wheels = sorted(where.glob("*.whl"))
    sdists = sorted(where.glob("*.tar.gz"))
    if not wheels and not sdists:
        print(f"no artifacts in {where}", file=sys.stderr)
        return 2

    problems: list[str] = []
    _check_versions(wheels + sdists, expected, problems)
    for w in wheels:
        print(f"  {w.name}\n      {_check_wheel(w, problems)}")
    for sd in sdists:
        print(f"  {sd.name}\n      {_check_sdist(sd, problems)}")

    if require_universal and not any(w.name.endswith("-py3-none-any.whl") for w in wheels):
        problems.append(
            "no py3-none-any wheel: platforms without a native build would "
            "fall back to the sdist and need a Rust toolchain"
        )

    if problems:
        print("\nFAILED:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    found = _artifact_version((wheels or sdists)[0]) or "?"
    print(f"\nOK: {len(wheels)} wheel(s), {len(sdists)} sdist(s), version {found}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
