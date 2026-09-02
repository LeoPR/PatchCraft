"""Sanity check: the package imports and exposes a version."""
import patchcraft


def test_version_is_defined() -> None:
    assert isinstance(patchcraft.__version__, str)
    assert patchcraft.__version__ != ""


def test_weight_kind_is_public() -> None:
    """`WeightKind` appears in `stitch`'s signature, so it must be reachable
    from the public namespace (unreachable in 0.2.0)."""
    assert "WeightKind" in patchcraft.__all__
    assert patchcraft.WeightKind is not None


def test_accel_available_is_public() -> None:
    """Support/debug helper for the optional native accelerator."""
    assert "accel_available" in patchcraft.__all__
    assert isinstance(patchcraft.accel_available(), bool)
