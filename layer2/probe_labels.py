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


TANWIN_FATH = "ً"
SUN = set("تثدذرزسشصضطظلن")


def norm(d):
    """Reconcile vowel-writing conventions between Tashkeela and CAMeL:
    - CAMeL omits sukun and the short vowel before a long vowel (قالَ / قَالَ)
    - CAMeL writes long ā as a dagger alif (هٰذا / هَذَا, الرَحْمٰن / الرَّحْمَن)
    - tanwīn al-fatḥ sits on the alif in CAMeL, before it in Tashkeela (ثَلاثاً / ثَلَاثًا)
    """
    d = d.replace(SUKUN, "").replace("ٱ", "ا")
    # Tashkeela vowels the connecting alif (وَاَللَّهُ, اُنْظُرْ); CAMeL does not
    d = re.sub("ا[َُِ]", "ا", d)
    d = d.replace(DAGGER, FATHA)
    d = re.sub(TANWIN_FATH + "([اى])", r"\1" + TANWIN_FATH, d)
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
    # CAMeL leaves out the shadda of a sun letter after the article (الثَلاث);
    # Tashkeela writes it (الثَّلَاث). Drop it on both sides.
    for i in range(2, len(out)):
        if out[i][0] in SUN and out[i - 1][0] == "ل" and not out[i - 1][1] \
                and out[i - 2][0] == "ا":
            out[i][1].discard(SHADDA)
    return out


def compatible(written, analysis, full, enc_len=0):
    """Every mark the text gives must be in the reading (text may be partial).
    In fully vowelled text a missing shadda is evidence too. Case ending ignored,
    also when it sits before a pronoun suffix of enc_len letters (يَلْزَمُهُ)."""
    w, a = split(norm(written)), split(norm(analysis))
    if [c for c, _ in w] != [c for c, _ in a]:
        return False
    for i, ((_, wm), (_, am)) in enumerate(zip(w, a)):
        if i == len(w) - 1 or i == len(w) - 1 - enc_len:
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
