"""Build hook for the optional native accelerator.

Everything declarative lives in ``pyproject.toml``. This file exists for the
one decision that cannot be declared: whether this particular build carries
the Rust extension.

The project publishes both shapes of wheel for the same version. On the
platforms with a build in CI the wheel is tagged ``cp312-abi3-<platform>`` and
contains ``patchcraft._accel_native``; everywhere else the wheel is
``py3-none-any`` and the library runs its pure-torch paths. Installers prefer
the more specific tag and fall back to the universal one, so a user gets the
accelerator where it exists and a working install where it does not.

Two environment variables drive the choice:

``PATCHCRAFT_PURE_PYTHON=1``
    Skip the extension entirely. This is how the universal wheel is built.

``PATCHCRAFT_REQUIRE_EXTENSION=1``
    Make a failed Rust build fatal. CI sets this for the platform wheels, so a
    toolchain problem fails the release instead of quietly shipping a
    platform-tagged wheel with no extension inside it.

With neither set, which is the case for an end user building from the sdist,
the extension is attempted when a Rust toolchain is on PATH and skipped when
there is none. Skipping rather than failing matters for the wheel tag: a build
that declares the extension and then loses it still produces a wheel named for
the platform, and such a wheel would be preferred over the real universal one
by anything that caches or mirrors it, while being the slow build inside.
"""

from __future__ import annotations

import os
import shutil

from setuptools import setup


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in ("", "0", "false", "no")


PURE_PYTHON = _truthy("PATCHCRAFT_PURE_PYTHON")
REQUIRE_EXTENSION = _truthy("PATCHCRAFT_REQUIRE_EXTENSION")
HAVE_CARGO = shutil.which("cargo") is not None

# Without a toolchain there is nothing to build, so the extension is not
# declared at all and the wheel is honestly tagged `py3-none-any`. CI asks for
# the extension explicitly, and there a missing toolchain has to be an error.
BUILD_EXTENSION = not PURE_PYTHON and (HAVE_CARGO or REQUIRE_EXTENSION)

setup_kwargs: dict[str, object] = {}

if BUILD_EXTENSION:
    # Imported lazily so a pure build works without setuptools-rust present.
    from setuptools_rust import Binding, RustExtension

    setup_kwargs["rust_extensions"] = [
        RustExtension(
            "patchcraft._accel_native",
            path="accel/Cargo.toml",
            binding=Binding.PyO3,
            # `optional` is the graceful-degradation switch: a build failure
            # drops the extension instead of aborting the install.
            optional=not REQUIRE_EXTENSION,
        )
    ]
    # One abi3 wheel per platform covers every Python from 3.12 up, which is
    # the whole supported range. setuptools-rust reads this back and adds the
    # matching `pyo3/abi3-py312` cargo feature.
    setup_kwargs["options"] = {"bdist_wheel": {"py_limited_api": "cp312"}}

setup(**setup_kwargs)
