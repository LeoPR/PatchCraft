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

Run it against a directory of artifacts::

    python tools/check_dist.py dist/
"""

from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path

EXT_STEM = "patchcraft/_accel_native"
SDIST_MUST_CARRY = ("accel/src/lib.rs", "accel/src/kernel.rs", "accel/Cargo.toml")


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
    where = Path(argv[1] if len(argv) > 1 else "dist")
    if not where.is_dir():
        print(f"not a directory: {where}", file=sys.stderr)
        return 2

    wheels = sorted(where.glob("*.whl"))
    sdists = sorted(where.glob("*.tar.gz"))
    if not wheels:
        print(f"no wheels in {where}", file=sys.stderr)
        return 2

    problems: list[str] = []
    for w in wheels:
        print(f"  {w.name}\n      {_check_wheel(w, problems)}")
    for s in sdists:
        print(f"  {s.name}\n      {_check_sdist(s, problems)}")

    if not any(w.name.endswith("-py3-none-any.whl") for w in wheels):
        problems.append(
            "no py3-none-any wheel: platforms without a native build would "
            "fall back to the sdist and need a Rust toolchain"
        )

    if problems:
        print("\nFAILED:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    print(f"\nOK: {len(wheels)} wheel(s), {len(sdists)} sdist(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
