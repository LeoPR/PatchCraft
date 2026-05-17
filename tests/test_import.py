"""Sanity check: the package imports and exposes a version."""
import patchforge


def test_version_is_defined() -> None:
    assert isinstance(patchforge.__version__, str)
    assert patchforge.__version__ != ""
