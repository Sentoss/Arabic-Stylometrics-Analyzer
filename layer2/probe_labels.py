"""Probe: how many Tashkeela words get a unique label by matching their vowels
against the analyzer's readings?

usage: python layer2/probe_labels.py TASHKEELA.txt N_WORDS
"""
import random
import re
import sys
from collections import Counter

from camel_tools.morphology.analyzer import Analyzer
from camel_tools.morphology.database import MorphologyDB

FATHA, DAMMA, KASRA = "َ", "ُ", "ِ"
SHADDA, SUKUN, DAGGER = "ّ", "ْ", "ٰ"
CASE = set("ًٌٍَُِ")      # tanwin + short vowels
MARK = re.compile("[ً-ْٰ]")
WORD = re.compile("[ء-يً-ْٰٱ]+")


def norm(d):
    """Reconcile vowel-writing conventions between Tashkeela and CAMeL:
    CAMeL omits sukun and the short vowel before a long vowel (قالَ vs قَالَ)."""
    d = d.replace(SUKUN, "").replace("ٱ", "ا")
    d = re.sub(FATHA + "(?=[اى])", "", d)
    d = re.sub(KASRA + "(?=ي(?![ً-ّ]))", "", d)
    d = re.sub(DAMMA + "(?=و(?![ً-ّ]))", "", d)
    return d


def split(d):
    out = []
    for ch in d:
        if MARK.match(ch):
            if out:
                out[-1][1].add(ch)
        else:
            out.append([ch, set()])
    return out


def compatible(written, analysis, full):
    """Every mark the text gives must be in the reading (text may be partial).
    In fully vowelled text a missing shadda is evidence too. Case ending ignored."""
    w, a = split(norm(written)), split(norm(analysis))
    if [c for c, _ in w] != [c for c, _ in a]:
        return False
    for i, ((_, wm), (_, am)) in enumerate(zip(w, a)):
        if i == len(w) - 1:
            wm, am = wm - CASE, am - CASE
        if not wm <= am - {DAGGER}:
            return False
        if full and (SHADDA in am) != (SHADDA in wm):
            return False
    return True


def main():
    path, limit = sys.argv[1], int(sys.argv[2])
    an = Analyzer(MorphologyDB.builtin_db())
    lines = open(path, encoding="utf-8").read().split("\n")
    random.seed(0)
    random.shuffle(lines)
    c, ex, n = Counter(), {}, 0
    for line in lines:
        for w in WORD.findall(line):
            bare = MARK.sub("", w)
            n += 1
            ans = [a for a in an.analyze(bare)
                   if a.get("source") != "backoff" and a["pos"] != "abbrev"]
            if not ans:
                c["oov"] += 1
                continue
            full = len(MARK.findall(w)) >= len(bare) - 1
            before = {(a["lex"], a["pos"]) for a in ans}
            labels = {(a["lex"], a["pos"]) for a in ans if compatible(w, a["diac"], full)}
            k = "no_match" if not labels else "unique" if len(labels) == 1 else "still_ambiguous"
            c[k] += 1
            c["ambiguous_without_vowels"] += len(before) > 1
            ex.setdefault(k, []).append((w, len(before), sorted(labels)[:3]))
        if n >= limit:
            break
    print("words:", n)
    for k in ["ambiguous_without_vowels", "unique", "still_ambiguous", "no_match", "oov"]:
        print(f"  {k:26} {c[k]:6}  {c[k] / n:.1%}")
    for k in ["still_ambiguous", "no_match"]:
        print(k, ex.get(k, [])[:10])


if __name__ == "__main__":
    main()
