"""PatchCraft: image patch extraction, pairing and reconstruction utilities."""

from patchcraft._accel import accel_available
from patchcraft.cache import Cache
from patchcraft.extract import Patchify, extract
from patchcraft.geometry import (
    PairedTilingSpec,
    TilingSpec,
    num_patches,
    paired_tilings,
    scale_factor,
    tilings,
)
from patchcraft.metrics import patch_metrics, per_patch_mse, per_patch_psnr
from patchcraft.pair import PatchMeta, PatchPair, pair
from patchcraft.reconstruct import reconstruct
from patchcraft.resize import resize
from patchcraft.stitch import WeightKind, stitch

try:
    # The redundant alias is the PEP 484 form for an explicit re-export, so
    # `mypy --strict` in a consuming project accepts `patchcraft.__version__`.
    from patchcraft._version import __version__ as __version__
except ImportError:  # pragma: no cover - source tree that was never built
    # importlib.metadata is the slow path and is only reached in a checkout
    # with no build behind it, so the import cost never lands on real users.
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _installed_version

    try:
        __version__ = _installed_version("patchcraft")
    except PackageNotFoundError:
        __version__ = "0+unknown"

__all__ = [
    "Cache",
    "PairedTilingSpec",
    "PatchMeta",
    "PatchPair",
    "Patchify",
    "TilingSpec",
    "WeightKind",
    "accel_available",
    "extract",
    "num_patches",
    "pair",
    "paired_tilings",
    "patch_metrics",
    "per_patch_mse",
    "per_patch_psnr",
    "reconstruct",
    "resize",
    "scale_factor",
    "stitch",
    "tilings",
]
