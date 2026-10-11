"""The images in `docs/GUIDE.md` are drawn from the library, and this keeps them so.

`tools/make_guide_figures.py` draws every image from tensors the library
returned, and asserts each number its caption states before it draws. Its
`--check` mode renders them again in memory and fails if a committed PNG has
drifted, if a caption no longer states a number the script measured, or if an
image and the GUIDE no longer link to each other. Running it here means a
change that moves a figure fails the suite, the way a stale line in
`docs/USAGE.md` does in `test_docs_usage.py`.
"""

from __future__ import annotations

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
    assert result.returncode == 0, (
        "the GUIDE figures or their captions no longer match a fresh render:\n"
        f"{result.stdout[-4000:]}\n{result.stderr[-2000:]}"
    )
