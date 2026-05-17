"""Test-only helpers for loading example images from public datasets.

Lazy: data lives in ``Z:\\caches\\datasets\\<name>\\`` and is downloaded on
first use. PatchCraft core does NOT depend on torchvision or on dataset
abstractions — only tests and ``lab/`` do. This module is NOT part of the
public API (underscore prefix).
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

DATASETS_ROOT = Path(r"Z:\caches\datasets")


def label_subset(
    labels: Sequence[int],
    n_per_label: int,
    seed: int = 0,
) -> list[int]:
    """Return a deterministic balanced subset of indices into ``labels``.

    For each distinct label, picks up to ``n_per_label`` indices uniformly
    at random (without replacement). If a label has fewer than
    ``n_per_label`` samples, takes all of them.

    Pure function: no torchvision, no dataset abstractions. Caller composes
    with ``torch.utils.data.Subset`` if needed.

    Parameters
    ----------
    labels
        Sequence of integer class labels, one per sample.
    n_per_label
        Max samples to take per distinct label.
    seed
        RNG seed for the per-class sampling.

    Returns
    -------
    list[int]
        Indices into the original ``labels`` sequence.

    Raises
    ------
    ValueError
        If ``n_per_label`` is not a positive int, or ``seed`` is not int.
    """
    if not isinstance(n_per_label, int) or isinstance(n_per_label, bool) or n_per_label <= 0:
        raise ValueError(f"n_per_label must be a positive int, got {n_per_label!r}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"seed must be int, got {seed!r}")

    by_label: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        by_label.setdefault(int(label), []).append(idx)

    rng = torch.Generator().manual_seed(seed)
    picks: list[int] = []
    for label in sorted(by_label):
        available = by_label[label]
        take = min(n_per_label, len(available))
        perm = torch.randperm(len(available), generator=rng)[:take].tolist()
        picks.extend(available[i] for i in perm)
    return picks


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
        RNG seed forwarded to :func:`label_subset`.
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

    picks = label_subset(ds.targets.tolist(), n_per_label=n_per_label, seed=seed)

    out: list[tuple[torch.Tensor, int]] = []
    for idx in picks:
        pil_img, label = ds[idx]
        tensor = TF.pil_to_tensor(pil_img).to(torch.float32) / 255.0
        out.append((tensor, int(label)))
    return out
