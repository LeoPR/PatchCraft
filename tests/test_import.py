"""Sanity check: the package imports and exposes a version."""
import patchkit


def test_version_is_defined() -> None:
    assert isinstance(patchkit.__version__, str)
    assert patchkit.__version__ != ""
