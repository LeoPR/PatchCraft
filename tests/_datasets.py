"""Test-only helpers for loading example images from public datasets.

Lazy: data lives in ``Z:\\caches\\datasets\\<name>\\`` and is downloaded on
first use. PatchKit core does NOT depend on torchvision — only tests and
``lab/`` do. This module is NOT part of the public API (underscore prefix).
"""
from __future__ import annotations

from pathlib import Path

import torch

DATASETS_ROOT = Path(r"Z:\caches\datasets")


def mnist_subset(
    n_per_label: int = 5,
    seed: int = 0,
    train: bool = True,
) -> list[tuple[torch.Tensor, int]]:
    """Return a small balanced MNIST subset as a list of ``(image, label)``.

    Each image is a ``(1, 28, 28)`` ``float32`` tensor in ``[0, 1]``.
    The dataset is downloaded on first call into
    ``Z:\\caches\\datasets\\mnist\\``; subsequent calls hit the cache.

    Parameters
    ----------
    n_per_label
        Samples per digit class (0-9). Total length is ``10 * n_per_label``.
    seed
        RNG seed for the per-class index sampling.
    train
        ``True`` uses the 60 000-image train split; ``False`` the 10 000-image
        test split.

    Raises
    ------
    ImportError
        If ``torchvision`` is not installed. Install with
        ``uv pip install -e ".[dev]"``.
    """
    try:
        from torchvision.datasets import MNIST
        from torchvision.transforms import functional as TF  # noqa: N812 (torchvision convention)
    except ImportError as exc:
        raise ImportError(
            "tests._datasets needs torchvision; install with "
            'uv pip install -e ".[dev]"'
        ) from exc

    root = DATASETS_ROOT / "mnist"
    root.mkdir(parents=True, exist_ok=True)
    ds = MNIST(root=str(root), train=train, download=True)

    by_label: dict[int, list[int]] = {}
    for idx, label in enumerate(ds.targets.tolist()):
        by_label.setdefault(int(label), []).append(idx)

    rng = torch.Generator().manual_seed(seed)
    picks: list[int] = []
    for label in sorted(by_label):
        available = by_label[label]
        take = min(n_per_label, len(available))
        perm = torch.randperm(len(available), generator=rng)[:take].tolist()
        picks.extend(available[i] for i in perm)

    out: list[tuple[torch.Tensor, int]] = []
    for idx in picks:
        pil_img, label = ds[idx]
        tensor = TF.pil_to_tensor(pil_img).to(torch.float32) / 255.0
        out.append((tensor, int(label)))
    return out
