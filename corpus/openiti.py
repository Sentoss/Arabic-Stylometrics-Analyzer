"""Download OpenITI texts and convert mARkdown into labelled paragraphs.

Output per book: JSONL, one record per paragraph:
  {"type": "prose" | "poetry" | "heading", "page": "V01P015", "text": "..."}

What is removed: the #META# header, page markers (PageV01P015), milestones
(ms12), footnote markers («1»), editorial tags (@QB@ etc.).
What is labelled, not removed: headings (often editorial), poetry (%~%
hemistich separators), so later steps can decide what counts as the
author's own prose.
"""
import csv
import json
import re
import time
import urllib.request
from pathlib import Path

META_CSV = Path("/home/claude/data_ext/openiti/meta/output/OpenITI_Github_clone_metadata_light.csv")

PAGE = re.compile(r"PageV(\d+)P(\d+)")
MILESTONE = re.compile(r"\bms\d+\b")
FOOTNOTE = re.compile(r"«\s*\d+\s*»|\(\s*\d+\s*\)|\[\s*\d+\s*\]")
TAGS = re.compile(r"@[A-Z]{2,4}\d*@|@\w*\d+\b")
SPACES = re.compile(r"[ \t ]+")


def load_metadata(path=META_CSV):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def primary_version(rows, book_uri):
    cands = [r for r in rows if r["book"] == book_uri]
    pri = [r for r in cands if r["status"] == "pri"] or cands
    return max(pri, key=lambda r: int(r["tok_length"] or 0)) if pri else None


def download(url, dest, retries=3):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                dest.write_bytes(r.read())
            return dest
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(3 * (i + 1))


AR_WORD = re.compile(r"[ء-ي]+")
OLD_VERSE = re.compile(r"^\s*\(?([^%()]+)%([^%()]*)\)?\s*$")     # ( hemistich % hemistich )
DOTS = re.compile(r"\s(?:\.\.\.|…)\s")


def is_verse(line: str) -> bool:
    """Verse lines come in three notations: OpenITI %~%, older OpenITI
    '( a % b )', and Shamela-style 'a ... b' with two short hemistichs."""
    if "%~%" in line:
        return True
    if OLD_VERSE.match(line):
        return True
    parts = DOTS.split(line)
    if len(parts) == 2:
        a, b = (len(AR_WORD.findall(p)) for p in parts)
        return 2 <= a <= 12 and 0 <= b <= 12
    return False


def parse_markdown(raw: str):
    """Return labelled paragraphs from an OpenITI mARkdown file. Verse lines
    are split out of the paragraphs they sit in, so prose stays prose."""
    body = raw.split("#META#Header#End#", 1)[-1]
    paras, cur, cur_type, page = [], [], "prose", ""

    def flush():
        nonlocal cur
        if cur:
            text = " ".join(cur)
            text = FOOTNOTE.sub(" ", TAGS.sub(" ", MILESTONE.sub(" ", text)))
            text = SPACES.sub(" ", text).strip()
            if text:
                paras.append({"type": cur_type, "page": page, "text": text})
        cur = []

    for line in body.splitlines():
        m = PAGE.search(line)
        line_page = f"V{m.group(1)}P{m.group(2)}" if m else None
        line = PAGE.sub(" ", line).rstrip()
        if line.startswith("#META#") or not line.strip():
            if line_page:
                page = line_page
            continue
        if line.startswith("### "):                 # heading (### | title)
            flush()
            cur_type = "heading"
            cur.append(line.lstrip("#").lstrip(" |").strip())
            flush()
            cur_type = "prose"
        else:
            new_para = line.startswith("# ")
            text = line[2:] if (new_para or line.startswith("~~")) else line
            verse = is_verse(MILESTONE.sub(" ", text))
            text = text.replace("%~%", " ... ").replace("%", " ... ")
            kind = "poetry" if verse else "prose"
            if new_para or kind != cur_type:
                flush()
                cur_type = kind
            cur.append(text)
        if line_page:
            page = line_page
    flush()
    return paras
