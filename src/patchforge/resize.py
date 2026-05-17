"""Image resizing with PIL or torch backends.

Output type matches input: PIL → PIL, Tensor → Tensor. Cross-backend
conversions go through a normalized float32 [0, 1] intermediate (with a
uint8 hop into PIL because PIL's standard modes are byte-typed).

Contract: docs/THEORY.md §5 and §9.4.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

__all__ = ["resize"]


_PIL_RESAMPLE_NAMES = {
    "nearest", "bilinear", "bicubic", "lanczos", "box", "hamming",
}
_TORCH_RESAMPLE_NAMES = {
    "nearest", "bilinear", "bicubic", "area", "nearest-exact",
}


def _validate_target_size(target_size: object) -> tuple[int, int]:
    if not (isinstance(target_size, tuple) and len(target_size) == 2):
        raise ValueError(
            f"target_size must be a 2-tuple (H, W), got {target_size!r}"
        )
    for axis_name, val in zip(("H", "W"), target_size, strict=True):
        if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
            raise ValueError(
                f"target_size[{axis_name}] must be a positive int, got {val!r}"
            )
    return (target_size[0], target_size[1])


def _pil_to_tensor_f32(pil_image: PILImage) -> torch.Tensor:
    """PIL → ``Tensor[C, H, W]`` float32 in [0, 1]. Copy is forced (PIL buffers
    are not safely sharable with torch)."""
    arr = np.asarray(pil_image, dtype=np.float32) / 255.0
    arr = arr[np.newaxis, ...] if arr.ndim == 2 else arr.transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(arr))


def _tensor_to_pil_u8(tensor: torch.Tensor) -> PILImage:
    """``Tensor[C, H, W]`` → PIL.Image (uint8 modes: L, RGB, RGBA only)."""
    from PIL import Image
    if tensor.is_floating_point():
        arr = (tensor.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
    else:
        arr = tensor.cpu().numpy().astype(np.uint8)
    c = arr.shape[0]
    if c == 1:
        return Image.fromarray(arr[0], mode="L")
    if c == 3:
        return Image.fromarray(arr.transpose(1, 2, 0), mode="RGB")
    if c == 4:
        return Image.fromarray(arr.transpose(1, 2, 0), mode="RGBA")
    raise ValueError(
        f"cannot convert tensor with {c} channels to PIL "
        f"(supported: C=1 → L, C=3 → RGB, C=4 → RGBA)"
    )


def _resize_pil(
    pil_image: PILImage,
    target_size: tuple[int, int],
    resample: str | None,
) -> PILImage:
    from PIL import Image
    h, w = target_size
    if resample is None:
        pil_resample = Image.Resampling.LANCZOS
    else:
        if not isinstance(resample, str):
            raise ValueError(
                f"resample must be str or None, got {type(resample).__name__}"
            )
        key = resample.lower()
        if key not in _PIL_RESAMPLE_NAMES:
            raise ValueError(
                f"resample {resample!r} not supported by PIL backend; "
                f"valid: {sorted(_PIL_RESAMPLE_NAMES)}"
            )
        pil_resample = Image.Resampling[key.upper()]
    return pil_image.resize((w, h), pil_resample)


def _resize_torch(
    tensor: torch.Tensor,
    target_size: tuple[int, int],
    resample: str | None,
) -> torch.Tensor:
    if resample is None:
        mode = "bilinear"
    else:
        if not isinstance(resample, str):
            raise ValueError(
                f"resample must be str or None, got {type(resample).__name__}"
            )
        key = resample.lower()
        if key not in _TORCH_RESAMPLE_NAMES:
            raise ValueError(
                f"resample {resample!r} not supported by torch backend; "
                f"valid: {sorted(_TORCH_RESAMPLE_NAMES)}"
            )
        mode = key

    original_dtype = tensor.dtype
    x = tensor.unsqueeze(0)
    if mode in {"bilinear", "bicubic"} and not x.is_floating_point():
        x = x.to(torch.float32)

    kwargs: dict[str, Any] = {"size": target_size, "mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    out = F.interpolate(x, **kwargs)
    return out[0].to(original_dtype)


def resize(
    image: torch.Tensor | PILImage,
    target_size: tuple[int, int],
    backend: Literal["pil", "torch"] = "pil",
    resample: str | None = None,
) -> torch.Tensor | PILImage:
    """Resize a single image, preserving input type.

    ``image`` is a ``PIL.Image`` or ``Tensor[C, H, W]``. ``target_size`` is
    ``(H, W)``. ``backend`` selects the resize algorithm family
    (``"pil"`` → ``PIL.Image.resize``; ``"torch"`` → ``F.interpolate``).
    Cross-backend conversions go through a normalized float32 [0, 1]
    intermediate. CUDA tensors are only accepted with ``backend="torch"``.

    ``resample=None`` picks each backend's default: LANCZOS for ``"pil"``,
    bilinear for ``"torch"``. The accepted resample strings differ between
    backends; an unsupported choice raises ``ValueError``.

    Rejects (per §9.4): non-2-tuple or non-positive ``target_size``;
    ``backend`` not in ``{"pil", "torch"}``; CUDA tensor with
    ``backend="pil"``; unsupported ``resample`` for the chosen backend;
    PIL tensor conversion of an unsupported channel count.
    """
    target_size = _validate_target_size(target_size)
    if backend not in {"pil", "torch"}:
        raise ValueError(f"backend must be 'pil' or 'torch', got {backend!r}")

    if isinstance(image, torch.Tensor):
        if image.ndim != 3:
            raise ValueError(
                f"tensor image must have ndim==3 (C, H, W), got ndim={image.ndim}"
            )
        if backend == "pil":
            if image.device.type != "cpu":
                raise ValueError(
                    f"backend='pil' cannot accept tensors on {image.device}; "
                    "move to CPU explicitly with .cpu() first"
                )
            original_dtype = image.dtype
            pil_in = _tensor_to_pil_u8(image)
            pil_out = _resize_pil(pil_in, target_size, resample)
            tensor_out = _pil_to_tensor_f32(pil_out)
            if not torch.empty(0, dtype=original_dtype).is_floating_point():
                tensor_out = (tensor_out * 255).round()
            return tensor_out.to(original_dtype)
        return _resize_torch(image, target_size, resample)

    # PIL branch.
    try:
        from PIL.Image import Image as PILImageCls
    except ImportError as exc:  # pragma: no cover — pillow is a runtime dep
        raise ImportError("Pillow is required for resize") from exc
    if not isinstance(image, PILImageCls):
        raise TypeError(
            f"image must be torch.Tensor or PIL.Image, got {type(image).__name__}"
        )
    if backend == "pil":
        return _resize_pil(image, target_size, resample)
    # PIL + backend == "torch"
    tensor = _pil_to_tensor_f32(image)
    resized = _resize_torch(tensor, target_size, resample)
    return _tensor_to_pil_u8(resized)
