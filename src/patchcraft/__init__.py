"""PatchCraft — image patch extraction, pairing and reconstruction utilities."""

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
from patchcraft.stitch import stitch

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
