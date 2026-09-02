"""Frozen public surface (B3/D4): 20 names, signatures, carrier fields.

A failure here means the 1.0-freeze surface moved. Changing it on purpose
means updating this file in the same commit and noting it in the CHANGELOG.
"""
from __future__ import annotations

import inspect
import typing
from dataclasses import fields

import patchcraft
from patchcraft import Cache, Patchify

P = inspect.Parameter
POK = P.POSITIONAL_OR_KEYWORD
KO = P.KEYWORD_ONLY
VP = P.VAR_POSITIONAL

EXPECTED_ALL = {
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
}
# 20 names: the 19 of FOCO-1.0.md plus `accel_available` (0.4.0).


def _params(fn: object) -> list[tuple[object, ...]]:
    """(name, kind, has_default, default_or_None) per parameter."""
    out: list[tuple[object, ...]] = []
    for p in inspect.signature(fn).parameters.values():  # type: ignore[union-attr]
        out.append((
            p.name,
            p.kind,
            p.default is not P.empty,
            None if p.default is P.empty else p.default,
        ))
    return out


def _returns(fn: object) -> object:
    return inspect.signature(fn).return_annotation  # type: ignore[union-attr]


class TestAll:
    def test_exact_set(self) -> None:
        assert set(patchcraft.__all__) == EXPECTED_ALL

    def test_no_duplicates(self) -> None:
        assert len(patchcraft.__all__) == len(EXPECTED_ALL)

    def test_every_name_reachable(self) -> None:
        for name in EXPECTED_ALL:
            assert getattr(patchcraft, name) is not None


class TestSignatures:
    def test_extract(self) -> None:
        assert _params(patchcraft.extract) == [
            ("image", POK, False, None),
            ("patch_size", POK, False, None),
            ("stride", POK, False, None),
            ("dilation", POK, True, 1),
        ]
        assert _returns(patchcraft.extract) == "torch.Tensor"

    def test_reconstruct(self) -> None:
        assert _params(patchcraft.reconstruct) == [
            ("patches", POK, False, None),
            ("image_shape", POK, False, None),
            ("stride", POK, False, None),
            ("dilation", POK, True, 1),
        ]
        assert _returns(patchcraft.reconstruct) == "torch.Tensor"

    def test_stitch(self) -> None:
        assert _params(patchcraft.stitch) == [
            ("patches", POK, False, None),
            ("image_shape", POK, False, None),
            ("stride", POK, False, None),
            ("weight", KO, True, "uniform"),
            ("dilation", KO, True, 1),
        ]
        assert _returns(patchcraft.stitch) == "torch.Tensor"

    def test_resize(self) -> None:
        assert _params(patchcraft.resize) == [
            ("image", POK, False, None),
            ("target_size", POK, False, None),
            ("backend", POK, True, "pil"),
            ("resample", POK, True, None),
        ]
        assert _returns(patchcraft.resize) == "torch.Tensor | PILImage"

    def test_pair(self) -> None:
        assert _params(patchcraft.pair) == [
            ("lr_image", POK, False, None),
            ("hr_image", POK, False, None),
            ("lr_patch_size", POK, False, None),
            ("scale_factor", POK, False, None),
            ("stride", POK, False, None),
            ("image_id", KO, True, None),
        ]
        assert _returns(patchcraft.pair) == "PatchPair"

    def test_num_patches(self) -> None:
        assert _params(patchcraft.num_patches) == [
            ("image_shape", POK, False, None),
            ("patch_size", POK, False, None),
            ("stride", POK, False, None),
            ("dilation", POK, True, 1),
        ]
        assert _returns(patchcraft.num_patches) == "tuple[int, int]"

    def test_tilings(self) -> None:
        assert _params(patchcraft.tilings) == [
            ("image_shape", POK, False, None),
            ("allow_overlap", KO, True, False),
            ("min_patch_size", KO, True, 2),
            ("max_patch_size", KO, True, None),
        ]
        assert _returns(patchcraft.tilings) == "list[TilingSpec]"

    def test_paired_tilings(self) -> None:
        assert _params(patchcraft.paired_tilings) == [
            ("lr_shape", POK, False, None),
            ("hr_shape", POK, False, None),
            ("allow_overlap", KO, True, False),
            ("min_patch_size", KO, True, 2),
            ("max_patch_size", KO, True, None),
        ]
        assert _returns(patchcraft.paired_tilings) == "list[PairedTilingSpec]"

    def test_scale_factor(self) -> None:
        assert _params(patchcraft.scale_factor) == [
            ("lr_shape", POK, False, None),
            ("hr_shape", POK, False, None),
        ]
        assert _returns(patchcraft.scale_factor) == "int | None"

    def test_patch_metrics(self) -> None:
        assert _params(patchcraft.patch_metrics) == [
            ("a", POK, False, None),
            ("b", POK, False, None),
            ("max_value", KO, True, 1.0),
        ]
        assert _returns(patchcraft.patch_metrics) == "dict[str, float]"

    def test_per_patch_mse(self) -> None:
        assert _params(patchcraft.per_patch_mse) == [
            ("a", POK, False, None),
            ("b", POK, False, None),
        ]
        assert _returns(patchcraft.per_patch_mse) == "torch.Tensor"

    def test_per_patch_psnr(self) -> None:
        assert _params(patchcraft.per_patch_psnr) == [
            ("a", POK, False, None),
            ("b", POK, False, None),
            ("max_value", KO, True, 1.0),
        ]
        assert _returns(patchcraft.per_patch_psnr) == "torch.Tensor"

    def test_accel_available(self) -> None:
        assert _params(patchcraft.accel_available) == []
        assert _returns(patchcraft.accel_available) == "bool"


class TestCarriers:
    def test_tiling_spec_fields(self) -> None:
        assert patchcraft.TilingSpec._fields == (
            "patch_size", "stride", "dilation",
            "num_patches", "total_patches", "overlap",
        )

    def test_paired_tiling_spec_fields(self) -> None:
        assert patchcraft.PairedTilingSpec._fields == ("lr", "hr", "scale_factor")

    def test_patch_pair_fields(self) -> None:
        assert [f.name for f in fields(patchcraft.PatchPair)] == [
            "lr_patches", "hr_patches", "metas",
        ]

    def test_patch_meta_fields(self) -> None:
        assert [f.name for f in fields(patchcraft.PatchMeta)] == [
            "patch_index", "row", "col",
            "lr_patch_size", "hr_patch_size", "image_id",
        ]


class TestCacheSurface:
    def test_init_signature(self) -> None:
        assert _params(Cache.__init__) == [
            ("self", POK, False, None),
            ("root", POK, False, None),
            ("namespace", POK, False, None),
            ("version", POK, True, 1),
        ]

    def test_method_signatures(self) -> None:
        assert _params(Cache.key_for) == [("self", POK, False, None), ("parts", VP, False, None)]
        assert _returns(Cache.key_for) == "str"
        assert _params(Cache.put) == [
            ("self", POK, False, None),
            ("key", POK, False, None),
            ("data", POK, False, None),
        ]
        assert _params(Cache.get) == [("self", POK, False, None), ("key", POK, False, None)]
        assert _returns(Cache.get) == "bytes | None"

    def test_properties(self) -> None:
        for name in ("root", "namespace", "version"):
            assert isinstance(inspect.getattr_static(Cache, name), property)


class TestPatchifySurface:
    def test_init_signature(self) -> None:
        # positional-or-keyword on purpose (verified at head): Patchify is a
        # transforms-style callable, `Patchify(4, 2)` is idiomatic.
        assert _params(Patchify.__init__) == [
            ("self", POK, False, None),
            ("patch_size", POK, False, None),
            ("stride", POK, False, None),
            ("dilation", POK, True, 1),
        ]


class TestWeightKind:
    def test_current_values(self) -> None:
        # D3: the set is open (new windows are a compatible addition); this
        # pins the starting point, not an exhaustive contract.
        assert typing.get_args(patchcraft.WeightKind) == ("uniform", "hann", "gaussian")
