"""PatchForge — image patch extraction, pairing and reconstruction utilities."""

from patchforge.cache import Cache
from patchforge.extract import Patchify, extract
from patchforge.geometry import (
    PairedTilingSpec,
    TilingSpec,
    num_patches,
    paired_tilings,
    scale_factor,
    tilings,
)
from patchforge.metrics import patch_metrics, per_patch_mse, per_patch_psnr
from patchforge.pair import PatchMeta, PatchPair, pair
from patchforge.reconstruct import reconstruct
from patchforge.resize import resize
from patchforge.stitch import stitch

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
