"""PatchKit — image patch extraction, pairing and reconstruction utilities."""

from patchkit.cache import Cache
from patchkit.extract import Patchify, extract
from patchkit.geometry import (
    PairedTilingSpec,
    TilingSpec,
    num_patches,
    paired_tilings,
    scale_factor,
    tilings,
)
from patchkit.metrics import patch_metrics, per_patch_mse, per_patch_psnr
from patchkit.pair import PatchMeta, PatchPair, pair
from patchkit.reconstruct import reconstruct
from patchkit.resize import resize
from patchkit.stitch import stitch

__version__ = "0.2.0"
__all__ = [
    "Cache",
    "PairedTilingSpec",
    "PatchMeta",
    "PatchPair",
    "Patchify",
    "TilingSpec",
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
