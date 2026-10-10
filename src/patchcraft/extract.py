"""Patch extraction: a strided window view and one copy, for every dtype.

Contract: docs/THEORY.md §1 and §9.1, docs/ADR/0001-patch-extraction-api.md,
docs/ADR/0002-patchify-transform.md.
"""
from __future__ import annotations

import torch

__all__ = ["Patchify", "extract"]


def _as_pair(value: int | tuple[int, int], name: str) -> tuple[int, int]:
    if isinstance(value, int) and not isinstance(value, bool):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        return (value, value)
    if isinstance(value, tuple) and len(value) == 2:
        h, w = value
        h_ok = isinstance(h, int) and not isinstance(h, bool)
        w_ok = isinstance(w, int) and not isinstance(w, bool)
        if not (h_ok and w_ok):
            raise ValueError(f"{name} must contain ints, got {value!r}")
        if h <= 0 or w <= 0:
            raise ValueError(f"{name} must be positive, got {value!r}")
        return (h, w)
    raise ValueError(f"{name} must be int or (int, int), got {value!r}")


def extract(
    image: torch.Tensor,
    patch_size: int | tuple[int, int],
    stride: int | tuple[int, int],
    dilation: int | tuple[int, int] = 1,
) -> torch.Tensor:
    """Extract rectangular patches from a `(C, H, W)` image.

    Returns `Tensor[L, C, ph, pw]` in row-major order. Patch `k` has its
    top-left at `(k // num_w * sh, k % num_w * sw)`. Truncation is the only
    boundary policy: if the geometry fits no patch, returns `Tensor[0, C, ph, pw]`.
    Dtype and device of `image` are preserved. Every dtype is accepted, the
    integer ones and bool included, at every dilation: extraction only copies,
    so each patch holds the image's own values bit for bit.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"image must be torch.Tensor, got {type(image).__name__}")
    if image.ndim != 3:
        raise ValueError(f"image must have ndim==3 (C, H, W), got ndim={image.ndim}")

    ph, pw = _as_pair(patch_size, "patch_size")
    sh, sw = _as_pair(stride, "stride")
    dh, dw = _as_pair(dilation, "dilation")

    c, h, w = image.shape
    eff_h = dh * (ph - 1) + 1
    eff_w = dw * (pw - 1) + 1

    if h < eff_h or w < eff_w:
        return torch.empty(0, c, ph, pw, dtype=image.dtype, device=image.device)

    # A strided window view and exactly one copy, for every dtype and dilation.
    # The window spans the dilated footprint and the slice keeps every d-th
    # pixel of it. Against F.unfold (im2col), which this replaced: the same
    # values bit for bit, 13 to 21 times faster at dilation 1 and 3.9 times at
    # dilation 2 on 3x1024x1024, and a kernel for every dtype where im2col has
    # none for integers.
    nh = (h - eff_h) // sh + 1
    nw = (w - eff_w) // sw + 1
    windows = image.unfold(1, eff_h, sh).unfold(2, eff_w, sw)  # (C, nh, nw, eh, ew)
    if dh > 1 or dw > 1:
        windows = windows[..., ::dh, ::dw]                      # (C, nh, nw, ph, pw)
    # Exactly one copy in every case. A reshape would copy too, except on
    # degenerate geometries (e.g. ph == pw == 1), where the permuted window
    # is already contiguous and reshape would return a view aliasing the
    # image, which the contract forbids. Cloning unconditionally covers
    # that case without asking the tensor for its storage, which a
    # torch.vmap batched tensor does not have.
    return (
        windows.permute(1, 2, 0, 3, 4)
        .clone(memory_format=torch.contiguous_format)
        .view(nh * nw, c, ph, pw)
    )

class Patchify:
    """Callable that extracts patches with a frozen geometry.

    Drop-in for `torchvision.transforms.Compose([...])`:

        transform = Compose([
            ToTensor(),
            GaussianBlur(kernel_size=3),
            Patchify(patch_size=4, stride=2),
        ])
        patches = transform(pil_image)  # Tensor[L, C, 4, 4]

    Equivalent to `lambda img: extract(img, patch_size, stride, dilation)`,
    but composable, repr-friendly, and validates the geometry at construction
    instead of at first call. Holds no state beyond the geometry: no cache,
    no fixed image size, no device. See ADR 0002.

    Output shape is `(L, C, ph, pw)`, the same as `extract`. Subsequent
    transforms in the Compose chain receive the patch stack, not a single
    patch; they must accept `(N, C, H, W)` or be wrapped.
    """

    __slots__ = ("_dh", "_dw", "_ph", "_pw", "_sh", "_sw")

    def __init__(
        self,
        patch_size: int | tuple[int, int],
        stride: int | tuple[int, int],
        dilation: int | tuple[int, int] = 1,
    ) -> None:
        self._ph, self._pw = _as_pair(patch_size, "patch_size")
        self._sh, self._sw = _as_pair(stride, "stride")
        self._dh, self._dw = _as_pair(dilation, "dilation")

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return extract(
            image,
            patch_size=(self._ph, self._pw),
            stride=(self._sh, self._sw),
            dilation=(self._dh, self._dw),
        )

    def __repr__(self) -> str:
        return (
            f"Patchify(patch_size=({self._ph}, {self._pw}), "
            f"stride=({self._sh}, {self._sw}), "
            f"dilation=({self._dh}, {self._dw}))"
        )
