"""Small orthographic helpers."""
import re

MARKS_RE = re.compile(r"[ً-ْٰٓ-ٕـ]")
SHADDA = "ّ"


def bare(word: str) -> str:
    """Letters only: strip diacritics and tatweel (letters are NOT normalized)."""
    return MARKS_RE.sub("", word)


def split_marks(word: str):
    """'عَلِيّ' -> [('ع',{'َ'}), ('ل',{'ِ'}), ('ي',{'ّ'})]. Tatweel dropped."""
    out = []
    for ch in word:
        if ch == "ـ":
            continue
        if MARKS_RE.match(ch):
            if out:
                out[-1][1].add(ch)
        else:
            out.append((ch, set()))
    return out
