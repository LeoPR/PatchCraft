"""The images in `docs/GUIDE.md` are drawn from the library, and this keeps them so.

`tools/make_guide_figures.py` draws every image from computed tensors, the
library's own output wherever an image shows what a call returns, and
asserts each number its caption states before it draws. Its
`--check` mode renders them again in memory and fails if a committed PNG has
drifted, if a caption no longer states a number the script measured, or if an
image and the GUIDE no longer link to each other. Running it here means a
change that moves a figure fails the suite, the way a stale line in
`docs/USAGE.md` does in `test_docs_usage.py`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools" / "make_guide_figures.py"


def test_guide_figures_match_a_fresh_render():
    assert TOOL.exists(), "tools/make_guide_figures.py is missing"
    result = subprocess.run(
        [sys.executable, str(TOOL), "--check"],
        capture_output=True,
        text=True,
        errors="replace",
        cwd=TOOL.parent.parent,
    )
    if result.returncode and os.environ.get("GITHUB_ACTIONS") == "true":
        # Printed lines of this form become annotations on the run, which name
        # the figure without opening the log.
        lines = [line for line in result.stdout.splitlines() if line.startswith("PROBLEM")]
        lines += result.stderr.strip().splitlines()[-1:]
        for line in lines:
            print(f"::error file=tools/make_guide_figures.py::{line}")
    assert result.returncode == 0, (
        "the GUIDE figures or their captions no longer match a fresh render:\n"
        f"{result.stdout[-4000:]}\n{result.stderr[-2000:]}"
    )
