"""Sanity check: the package imports and exposes a version."""
import patchcraft


def test_version_is_defined() -> None:
    assert isinstance(patchcraft.__version__, str)
    assert patchcraft.__version__ != ""
