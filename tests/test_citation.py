"""The citation metadata must not drift from the release it claims.

`CITATION.cff` is what GitHub reads for the "Cite this repository" button, so
a stale version there is wrong in public. It cannot be compared against
`patchcraft.__version__`, which setuptools-scm resolves to a development
version in a checkout; the comparison that means something is against the most
recent released section of the changelog.

Parsed with a regex rather than a YAML library on purpose: this repository has
no YAML runtime dependency and adding one to assert four lines would cost more
than it checks.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CITATION = ROOT / "CITATION.cff"
CHANGELOG = ROOT / "CHANGELOG.md"

REQUIRED_KEYS = ("cff-version", "message", "title", "authors", "type", "version")


def _field(text: str, key: str) -> str | None:
    m = re.search(rf'^{re.escape(key)}:\s*"?([^"\n]+)"?\s*$', text, re.M)
    return m.group(1).strip() if m else None


@pytest.mark.skipif(not CITATION.exists(), reason="CITATION.cff is not in this tree")
def test_citation_carries_the_required_keys():
    text = CITATION.read_text(encoding="utf-8")
    missing = [k for k in REQUIRED_KEYS if not re.search(rf"^{re.escape(k)}:", text, re.M)]
    assert not missing, f"CITATION.cff is missing {missing}"
    assert _field(text, "cff-version") == "1.2.0"


@pytest.mark.skipif(
    not (CITATION.exists() and CHANGELOG.exists()),
    reason="needs both CITATION.cff and CHANGELOG.md",
)
def test_citation_version_matches_the_newest_released_changelog_section():
    released = re.findall(r"^## \[(\d+\.\d+\.\d+)\]", CHANGELOG.read_text(encoding="utf-8"), re.M)
    assert released, "no released section found in CHANGELOG.md"
    assert _field(CITATION.read_text(encoding="utf-8"), "version") == released[0], (
        "CITATION.cff names a different version than the newest released section of "
        "CHANGELOG.md; step 2 of the release checklist updates it"
    )
