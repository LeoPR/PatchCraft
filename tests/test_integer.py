"""Integer images through extract and reconstruct (THEORY §9.1, §9.2; ADR 0004).

extract only copies, so it takes every dtype. reconstruct takes the five
integer dtypes and returns, at each pixel, the exact mean of the covering
values rounded half to even. For patches straight from extract that mean is
the pixel itself in every legal geometry, with no count-map rule, and the
round trips below assert it on random data over each dtype's whole range,
extremes included, on both code paths. Modified patches are checked against
the same mean computed in Python integers, which cannot overflow.
"""
from __future__ import annotations

import os

import pytest
import torch

from patchcraft import extract, pair, reconstruct, stitch
from tests._rng import bit_equal
from tests.test_exactness import _SAMPLE, _SPACE

INT_DTYPES = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]

# (h, w, ph, pw, sh, sw), each covering its image exactly. They hold counts of
# 1 only, 1/2/4, a 3 on one axis, 3 and 6 and 9, and a rectangular mix.
GEOMETRIES = [
    (8, 8, 2, 2, 2, 2),
    (16, 16, 4, 4, 2, 2),
    (9, 7, 3, 3, 1, 2),
    (6, 6, 3, 3, 1, 1),
    (12, 10, 4, 3, 2, 1),
    (11, 13, 5, 3, 3, 2),
]


@pytest.fixture(params=["1", "0"], ids=["accel", "pure"])
def accel(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("PATCHCRAFT_ACCEL", request.param)
    return str(request.param)


def _random(dtype: torch.dtype, shape: tuple[int, ...], seed: int) -> torch.Tensor:
    """Uniform over the dtype's whole range, with its two extremes planted."""
    info = torch.iinfo(dtype)
    g = torch.Generator().manual_seed(seed)
    out = torch.randint(info.min, info.max, shape, generator=g, dtype=torch.int64).to(dtype)
    out.view(-1)[0] = info.max
    out.view(-1)[-1] = info.min
    return out


def _python_mean(
    patches: torch.Tensor, shape: tuple[int, int, int], sh: int, sw: int
) -> list[list[list[int]]]:
    """The covering values' exact mean, rounded half to even, in Python integers."""
    n, c, ph, pw = patches.shape
    _, h, w = shape
    nw = (w - pw) // sw + 1
    total = [[[0] * w for _ in range(h)] for _ in range(c)]
    count = [[0] * w for _ in range(h)]
    values = patches.tolist()
    for k in range(n):
        top, left = (k // nw) * sh, (k % nw) * sw
        for i in range(ph):
            for j in range(pw):
                count[top + i][left + j] += 1
                for ch in range(c):
                    total[ch][top + i][left + j] += values[k][ch][i][j]
    out = []
    for ch in range(c):
        plane = []
        for y in range(h):
            row = []
            for x in range(w):
                floor, rest = divmod(total[ch][y][x], count[y][x])
                if 2 * rest > count[y][x] or (2 * rest == count[y][x] and floor % 2 == 1):
                    floor += 1
                row.append(floor)
            plane.append(row)
        out.append(plane)
    return out


@pytest.mark.parametrize("dtype", INT_DTYPES)
@pytest.mark.parametrize("geom", GEOMETRIES)
def test_round_trip_is_exact(accel: str, dtype: torch.dtype, geom: tuple[int, ...]) -> None:
    h, w, ph, pw, sh, sw = geom
    img = _random(dtype, (2, h, w), seed=h * w + ph)
    patches = extract(img, (ph, pw), (sh, sw))
    assert patches.dtype == dtype
    back = reconstruct(patches, img.shape, stride=(sh, sw))
    assert back.dtype == dtype
    assert bit_equal(back, img)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int64])
def test_round_trip_is_exact_over_the_geometry_space(dtype: torch.dtype) -> None:
    """Every sampled legal geometry, not only the power-of-two ones the float rule allows.

    PATCHCRAFT_SWEEP_FULL=1 runs all 126,736."""
    space = _SPACE if os.environ.get("PATCHCRAFT_SWEEP_FULL") == "1" else _SAMPLE
    for h, w, ph, pw, sh, sw in space:
        img = _random(dtype, (1, h, w), seed=h * 31 + w)
        back = reconstruct(extract(img, (ph, pw), (sh, sw)), img.shape, stride=(sh, sw))
        assert bit_equal(back, img), (h, w, ph, pw, sh, sw)


@pytest.mark.parametrize("dtype", INT_DTYPES)
@pytest.mark.parametrize("geom", GEOMETRIES)
def test_disagreeing_patches_get_the_exact_mean_rounded_half_to_even(
    accel: str, dtype: torch.dtype, geom: tuple[int, ...]
) -> None:
    h, w, ph, pw, sh, sw = geom
    n = ((h - ph) // sh + 1) * ((w - pw) // sw + 1)
    patches = _random(dtype, (n, 2, ph, pw), seed=n + w)
    got = reconstruct(patches, (2, h, w), stride=(sh, sw))
    assert got.tolist() == _python_mean(patches, (2, h, w), sh, sw)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [(1, 2, 2), (2, 3, 2), (-1, -2, -2), (-2, -3, -2), (0, 3, 2), (-3, 0, -2)],
)
def test_ties_go_to_even(left: int, right: int, expected: int) -> None:
    """Two 1x2 patches overlapping on the middle pixel of a 1x3 strip."""
    patches = torch.tensor([[[[0, left]]], [[[right, 0]]]], dtype=torch.int16)
    assert reconstruct(patches, (1, 1, 3), stride=1)[0, 0, 1].item() == expected


@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_no_intermediate_overflows_at_the_extremes(accel: str, dtype: torch.dtype) -> None:
    """Counts up to 9 of the largest and smallest values, and mixes of the two."""
    info = torch.iinfo(dtype)
    for value in (info.max, info.min):
        img = torch.full((1, 6, 6), value, dtype=dtype)
        assert bit_equal(reconstruct(extract(img, 3, 1), img.shape, stride=1), img)
    mixed = torch.where(torch.arange(16).view(16, 1, 1, 1) % 2 == 0, info.max, info.min)
    patches = mixed.expand(16, 1, 3, 3).to(dtype)
    got = reconstruct(patches, (1, 6, 6), stride=1)
    assert got.tolist() == _python_mean(patches, (1, 6, 6), 1, 1)


@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_dilated_integer_extract_matches_the_float_one(dtype: torch.dtype) -> None:
    """Integers at dilation 2 are the same gather the float path makes."""
    img = _random(torch.int16, (2, 17, 19), seed=3).to(dtype)
    got = extract(img, (3, 4), (2, 1), dilation=(2, 3))
    ref = extract(img.to(torch.float64), (3, 4), (2, 1), dilation=(2, 3))
    assert got.dtype == dtype
    assert torch.equal(got.to(torch.float64), ref)


def test_pair_takes_integer_images() -> None:
    lr = _random(torch.uint8, (1, 8, 8), seed=5)
    hr = lr.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
    result = pair(lr, hr, lr_patch_size=2, scale_factor=2, stride=2)
    assert result.lr_patches.dtype == result.hr_patches.dtype == torch.uint8
    upsampled = result.lr_patches.repeat_interleave(2, 2).repeat_interleave(2, 3)
    assert torch.equal(result.hr_patches, upsampled)


def test_vmap_over_integer_reconstruct_equals_a_loop() -> None:
    batch = _random(torch.int32, (3, 1, 12, 10), seed=7)

    def round_trip(im: torch.Tensor) -> torch.Tensor:
        return reconstruct(extract(im, (4, 3), (2, 1)), im.shape, stride=(2, 1))

    got = torch.vmap(round_trip)(batch)
    assert torch.equal(got, torch.stack([round_trip(im) for im in batch]))
    assert torch.equal(got, batch)


@pytest.mark.parametrize("dtype", [torch.bool, torch.complex64])
def test_bool_and_complex_patches_are_refused(dtype: torch.dtype) -> None:
    patches = torch.zeros(4, 1, 2, 2, dtype=dtype)
    with pytest.raises(ValueError, match="floating-point or integer"):
        reconstruct(patches, (1, 4, 4), stride=2)


def test_stitch_stays_float_only() -> None:
    patches = torch.zeros(4, 1, 2, 2, dtype=torch.uint8)
    with pytest.raises(ValueError, match="floating-point"):
        stitch(patches, (1, 4, 4), stride=2)


def test_output_does_not_alias_the_patches() -> None:
    patches = extract(_random(torch.uint8, (1, 4, 4), seed=9), 2, 2)
    out = reconstruct(patches, (1, 4, 4), stride=2)
    before = patches.clone()
    out.fill_(0)
    assert torch.equal(patches, before)
