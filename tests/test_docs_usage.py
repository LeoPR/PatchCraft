"""`docs/USAGE.md` is executable, and this is what makes that true.

The page is a REPL transcript of the public surface. Until 0.5.3 nothing ran
it, and it spent three releases describing an API that had moved: it claimed
eighteen public names when there were twenty, and eighteen of its examples no
longer reproduced. Collecting it as a doctest means a stale line fails here
instead of misleading a reader.

`--doctest-glob` is not set repository-wide on purpose. Only this page is a
transcript; the other documents are prose with illustrative blocks, and
collecting them would assert outputs they never promised.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

USAGE = Path(__file__).resolve().parents[1] / "docs" / "USAGE.md"


def test_usage_page_runs_as_written():
    assert USAGE.exists(), "docs/USAGE.md is missing"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--doctest-glob=*.md", str(USAGE), "-q"],
        capture_output=True,
        text=True,
        cwd=USAGE.parent.parent,
    )
    assert result.returncode == 0, (
        "docs/USAGE.md no longer reproduces its own output:\n"
        f"{result.stdout[-4000:]}\n{result.stderr[-2000:]}"
    )
