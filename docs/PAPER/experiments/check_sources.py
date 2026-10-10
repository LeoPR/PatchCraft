"""Re-open every reference in sources.tsv and confirm the expected text is there.

Each row names a key, a kind (doi, url, pdf), a locator and a piece of text
the source must contain. A doi row is checked against the Crossref record
title; a url row against the page text (tags stripped, entities unescaped,
whitespace collapsed); a pdf row against the pdftotext output of the
downloaded file.

    python docs/PAPER/experiments/check_sources.py [download dir]

Downloads go to the system temp folder unless a directory is given, never
into the repository. The pdf rows need pdftotext on PATH (Git for Windows
ships one in /mingw64/bin). Reading the sources needs network access; nothing
is sent but the requests themselves.
"""

import hashlib
import html
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import ClassVar

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
DL = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path(tempfile.gettempdir()) / "patchcraft-paper-sources"
)
DL.mkdir(parents=True, exist_ok=True)
# Some hosts answer a browser-like agent with an anti-bot page and others refuse a plain one,
# so each locator is tried with both and counts as confirmed if either response holds the text.
AGENTS = ("reference-check/1.0", "Mozilla/5.0 (reference check)")


class _Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self.skip = [], 0

    BLOCK: ClassVar[set[str]] = {
        "p",
        "div",
        "br",
        "li",
        "tr",
        "td",
        "th",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "pre",
        "dt",
        "dd",
        "table",
        "section",
        "article",
        "header",
        "footer",
        "ul",
        "ol",
        "blockquote",
    }

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1
        if tag in self.BLOCK:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def norm(s):
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


def fetch(url, agent=AGENTS[0]):
    req = urllib.request.Request(url, headers={"User-Agent": agent})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.status, r.geturl(), r.read()


_cache = {}


def text_of(kind, loc, agent=AGENTS[0]):
    if (kind, loc, agent) in _cache:
        return _cache[(kind, loc, agent)]
    if kind == "doi":
        status, final, body = fetch(f"https://api.crossref.org/works/{loc}")
        msg = json.loads(body)["message"]
        authors = ", ".join(
            f"{a.get('given', '')} {a.get('family', '')}".strip() for a in msg.get("author", [])
        )
        out = (status, final, norm(" ".join(msg.get("title", []))), authors)
    elif kind == "pdf":
        status, final, body = fetch(loc, agent)
        d = DL / hashlib.sha1(loc.encode()).hexdigest()[:12]
        d.mkdir(exist_ok=True)
        pdf = d / "source.pdf"
        pdf.write_bytes(body)
        txt = d / "source.txt"
        subprocess.run([shutil.which("pdftotext") or "pdftotext", str(pdf), str(txt)], check=True)
        out = (status, final, norm(txt.read_text(encoding="utf-8", errors="replace")), "")
    else:
        status, final, body = fetch(loc, agent)
        raw = body.decode("utf-8", errors="replace")
        if "<html" in raw[:2000].lower() or "<!doctype" in raw[:200].lower():
            p = _Text()
            p.feed(raw)
            raw = "".join(p.parts)
        out = (status, final, norm(raw), "")
    _cache[(kind, loc, agent)] = out
    return out


rows = [
    ln.rstrip("\n").split("\t")
    for ln in (HERE.parent / "sources.tsv").read_text(encoding="utf-8").splitlines()
    if ln.strip() and not ln.startswith("#")
]
ok = 0
for key, kind, loc, expected in rows:
    try:
        for agent in AGENTS[: 1 if kind == "doi" else 2]:
            try:
                status, final, text, authors = text_of(kind, loc, agent)
            except Exception:
                if agent == AGENTS[-1]:
                    raise
                continue
            found = norm(expected).lower() in text.lower()
            if found:
                break
        if not found and kind != "doi":
            # one slow retry: a host that rate-limits a burst can answer with a stub page
            time.sleep(10)
            _cache.pop((kind, loc, AGENTS[-1]), None)
            status, final, text, authors = text_of(kind, loc, AGENTS[-1])
            found = norm(expected).lower() in text.lower()
        ok += found
        extra = f" | Crossref authors: {authors}" if kind == "doi" else ""
        print(f"{'OK  ' if found else 'MISS'} {key:28s} {kind:4s} HTTP {status} {loc}{extra}")
        if not found:
            print(f"     expected: {expected!r} | got {len(text)} chars starting {text[:120]!r}")
    except Exception as e:  # report and continue
        print(f"FAIL {key:28s} {kind:4s} {loc}: {type(e).__name__}: {e}")
print(f"{ok} of {len(rows)} rows confirmed")
