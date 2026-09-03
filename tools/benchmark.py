"""Measure the accelerated fold against the pure-torch one, and prove they agree.

The numbers quoted in the documentation come from this script. Run it yourself
rather than trusting them, because the ratio depends on your CPU, your torch
build and how many threads torch decides to use:

    python tools/benchmark.py
    python tools/benchmark.py --markdown     # the table as the docs carry it

It measures the one hot path the accelerator touches, which is the overlapping
fold inside ``reconstruct`` and ``stitch``. Non-overlapping geometries never
reach it: they take a closed-form path that is already a pure rearrangement, so
there is nothing to accelerate and they are not measured here.

Every case is run twice, once with the accelerator and once with
``PATCHCRAFT_ACCEL=0``, and the two results are compared with ``torch.equal``
before any timing is reported. A benchmark that did not check that would be
measuring two different computations.
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

import patchcraft
from patchcraft import extract, reconstruct, stitch


@dataclass(frozen=True)
class Case:
    shape: tuple[int, int, int]
    patch: int
    stride: int

    @property
    def label(self) -> str:
        c, h, w = self.shape
        return f"{c}x{h}x{w}, patch {self.patch}, stride {self.stride}"


CASES = (
    Case((3, 512, 512), 32, 16),
    Case((3, 1024, 1024), 64, 32),
    Case((3, 2048, 2048), 64, 32),
)


def _median_ms(fn: Callable[[], object], repeats: int) -> float:
    fn()  # warm caches and any first-call import
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def _run(fn: Callable[[], torch.Tensor], repeats: int) -> tuple[float, float, bool]:
    """Time ``fn`` accelerated and pure, and say whether they agree bit for bit."""
    previous = os.environ.get("PATCHCRAFT_ACCEL")
    try:
        os.environ["PATCHCRAFT_ACCEL"] = "1"
        accelerated = fn()
        fast = _median_ms(fn, repeats)

        os.environ["PATCHCRAFT_ACCEL"] = "0"
        pure = fn()
        slow = _median_ms(fn, repeats)
    finally:
        if previous is None:
            os.environ.pop("PATCHCRAFT_ACCEL", None)
        else:
            os.environ["PATCHCRAFT_ACCEL"] = previous

    return slow, fast, torch.equal(accelerated, pure)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=15, help="timed runs per case")
    parser.add_argument("--markdown", action="store_true", help="emit a markdown table")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv[1:])

    if not patchcraft.accel_available():
        print(
            "This install has no native accelerator, so there is nothing to compare.\n"
            "The wheels for Windows x64, Linux x86_64 and aarch64, and both macOS\n"
            "architectures carry one; other platforms run the pure path only.",
            file=sys.stderr,
        )
        return 2

    torch.manual_seed(args.seed)
    rows: list[tuple[str, str, float, float, bool]] = []

    for case in CASES:
        image = torch.rand(*case.shape)
        patches = extract(image, patch_size=case.patch, stride=case.stride)
        shape = case.shape

        # Bound as defaults: the closures outlive the loop variable otherwise.
        def run_reconstruct(
            p: torch.Tensor = patches, sh: tuple[int, int, int] = shape, st: int = case.stride
        ) -> torch.Tensor:
            return reconstruct(p, sh, stride=st)

        def run_stitch(
            p: torch.Tensor = patches, sh: tuple[int, int, int] = shape, st: int = case.stride
        ) -> torch.Tensor:
            return stitch(p, sh, stride=st, weight="hann")

        rows.append((case.label, "reconstruct", *_run(run_reconstruct, args.repeats)))
        rows.append((case.label, 'stitch, weight="hann"', *_run(run_stitch, args.repeats)))

    disagreed = [r for r in rows if not r[4]]

    env = (
        f"torch {torch.__version__}, Python {platform.python_version()}, "
        f"{platform.system()} {platform.machine()}, "
        f"{os.cpu_count()} logical cores, torch using {torch.get_num_threads()} threads"
    )

    if args.markdown:
        print("| Geometry | Call | Pure torch | Accelerated | Speedup |")
        print("|---|---|---|---|---|")
        for label, call, slow, fast, _ in rows:
            print(
                f"| {label} | `{call}` | {slow:.1f} ms | {fast:.1f} ms | {slow / fast:.1f}x |"
            )
        print()
        print(f"Measured with `python tools/benchmark.py --markdown` on {env}.")
    else:
        print(env)
        print(f"patchcraft {patchcraft.__version__}, median of {args.repeats} runs\n")
        head = f"{'Geometry':30s} {'Call':22s} {'pure':>10s} {'accel':>10s} {'x':>7s}  equal"
        print(head)
        print("-" * len(head))
        for label, call, slow, fast, same in rows:
            print(
                f"{label:30s} {call:22s} {slow:9.1f}m {fast:9.1f}m "
                f"{slow / fast:6.1f}x  {'yes' if same else 'NO'}"
            )

    if disagreed:
        print(
            f"\nFAILED: {len(disagreed)} case(s) where the accelerated and pure results "
            "are not bit-identical. The timings above are meaningless.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
