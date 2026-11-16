"""Harvest Classical vowelings from vowelled text that the MSA analyzer misses.

For each word whose printed vowels contradict every reading (no_match in the
labels file), try each reading of the same bare word: keep its letters, prefix
and suffix, and put the text's vowels on the *stem*. If that makes the reading
agree with the text, the stem has a Classical vowelling the dictionary lacks:

  الثُّلُثِ   reading ثُلْث      → stem ثُلُث      (same lemma)
  الْجُمُعَةِ reading جُمْعَة    → stem جُمُعَة
  يُوسُفَ    reading يُوسِف     → stem يُوسُف
  حَدَّثَنَا  reading حَدَث+نا   → stem حَدَّث      (verb: new lemma حَدَّث, Form II)
  عَجَزَ     reading عَجُز      → stem عَجَز       (verb: new lemma عَجَز)

A variant is kept if it is attested at least --min times and accounts for at
least --share of the harvest for its (lemma, POS, stem). Nouns keep their
lemma (same word, Classical vowelling); verbs get the new perfect stem as lemma.

usage: python layer2/harvest_variants.py LABELS.tsv [--min 5] [--out resources/classical_variants.tsv]
"""
import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

from probe_labels import MARK, compatible, norm
from supplement import ClassicalAnalyzer, SHADDA, UNIT, join, stem_of, units

CASE = set("ًٌٍ")   # tanwīn on a stem's last letter is case, not stem


def bare(s):
    return MARK.sub("", s).replace("ٱ", "ا")


def stem_span(a):
    """(start, end) letter positions of the stem inside a['diac']."""
    st = stem_of(a)
    if not st:
        return None
    i, stem, tag, parts = st
    before = "".join(t for t, _ in parts[:i] if t != "(null)")
    allp = "".join(t for t, _ in parts if t != "(null)")
    if bare(allp) != bare(a["diac"]):
        return None
    s = len(bare(before))
    return s, s + len(bare(stem)), stem


def diffs(old, new):
    """[(position, old marks, new marks)] after reconciling conventions."""
    ou, nu = units(norm(old)), units(norm(new))
    if len(ou) != len(nu):
        return None
    return [(k, set(m1), set(m2)) for k, ((_, m1), (_, m2)) in enumerate(zip(ou, nu)) if set(m1) != set(m2)]


VOW = set("\u064e\u064f\u0650")


def epenthetic(old, new):
    """Only vowels filled in where there was none (ثُلْث → ثُلُث)."""
    d = diffs(old, new)
    return bool(d) and all(not (m1 & VOW) and (m2 & VOW) and SHADDA not in m2 - m1 for _, m1, m2 in d)


def changes(old, new):
    """Letters whose marks differ, after reconciling writing conventions,
    and whether a shadda was added."""
    ou, nu = units(norm(old)), units(norm(new))
    if len(ou) != len(nu):
        return 99, False
    diff = [k for k, ((_, m1), (_, m2)) in enumerate(zip(ou, nu)) if set(m1) != set(m2)]
    added = any(SHADDA in nu[k][1] and SHADDA not in ou[k][1] for k in diff)
    return len(diff), added


def allowed(a, old, new):
    n, added = changes(old, new)
    if n == 0:
        return None
    if a["pos"] == "verb":
        if str(a.get("source", "")).startswith("classical"):
            return None
        d = diffs(old, new) or []
        if n != 1 or len(d) != 1 or d[0][0] != 1:
            return None          # one change, on the middle radical only
        k, m1, m2 = d[0]
        if added:                # Form II: حَدَث → حَدَّث
            return n
        if SHADDA in m1 - m2:
            return None          # removing a shadda makes another word (غَرَّر / غَرَر)
        return n if (m1 & VOW) and (m2 & VOW) else None   # stem vowel: عَجُز → عَجَز
    return None if added else n


def try_variant(written, a, full, enc):
    """New stem if putting the text's stem vowels on reading a makes it fit."""
    sp = stem_span(a)
    if not sp:
        return None
    s, e, stem = sp
    wu, au = units(written), units(a["diac"])
    if [bare(c) for c, _ in wu] != [bare(c) for c, _ in au] or e > len(au):
        return None
    su = units(stem)
    if len(su) != e - s:
        return None
    new = []
    for k, (c, m) in enumerate(su):
        wm = wu[s + k][1]
        if k == 0 and s > 0 and SHADDA in wm and SHADDA not in m:
            wm = wm.replace(SHADDA, "")   # sun-letter shadda after the article
        if k == len(su) - 1:
            # the stem's last letter: its vowel is usually the suffix/case vowel;
            # take only a shadda from the text, keep the stem's own marks
            nm = m + (SHADDA if SHADDA in wm and SHADDA not in m else "")
        else:
            nm = "".join(ch for ch in wm if ch not in CASE) if wm else m
        new.append([c, nm])
    new_stem = join(new)
    if new_stem == stem:
        return None
    n = allowed(a, stem, new_stem)
    if n is None:
        return None
    cand = a["diac"].replace(stem, new_stem, 1)
    if not compatible(written, cand, full, enc):
        return None
    return stem, new_stem, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    ap.add_argument("--min", type=int, default=5)
    ap.add_argument("--share", type=float, default=0.2)
    ap.add_argument("--out", default=str(Path(__file__).parent / "resources" / "classical_variants.tsv"))
    a = ap.parse_args()
    from build_labels import enc_len
    words = Counter()
    for r in csv.DictReader(open(a.labels, encoding="utf-8"), delimiter="\t"):
        if r.get("status") == "no_match":
            words[r["written"]] += 1
    print(f"{len(words)} distinct no_match forms, {sum(words.values())} tokens")
    an = ClassicalAnalyzer(variants=None)
    found = defaultdict(Counter)       # (lex, pos, stem) -> new_stem -> tokens
    asp = {}
    skipped = Counter()
    for w, n in words.items():
        b = bare(w)
        forms = [b] + ([b[:-1] + "ي"] if b.endswith("ى") else [])
        full = len(MARK.findall(w)) >= len(b) - 1
        hits = {}
        for f in forms:
            for r in an.analyze(f):
                v = try_variant(w if f == b else re.sub("ى(?=[ً-ْ]*$)", "ي", w), r, full, enc_len(r))
                if v:
                    key = (r["lex"], r["pos"], v[0])
                    hits[(key, v[1])] = v[2]
                    asp[key] = (r.get("asp"), r.get("bw", ""))
        if not hits:
            continue
        best = min(hits.values())
        top = {k for k, c in hits.items() if c == best}
        if len({(key[0], key[1]) for key, _ in top}) != 1:
            skipped[w] = n      # several lemmas/POS explain it equally well: skip
            continue
        for key, new in top:
            found[key][new] += n
    rows = []
    for key, c in found.items():
        tot = sum(c.values())
        for new, n in c.most_common(2):
            if n >= a.min and n / tot >= a.share:
                lex, pos, stem = key
                new_lex = ""
                if pos == "verb":
                    if asp[key][0] == "p" and bare(lex) == bare(stem):
                        new_lex = new                 # حَدَث → حَدَّث, عَجُز → عَجَز
                elif not epenthetic(stem, new):
                    if stem not in lex:
                        continue                      # broken plural etc.: too risky
                    new_lex = lex.replace(stem, new, 1)   # مُكاتِب → مُكاتَب: another word
                rows.append((lex, pos, stem, new, new_lex, n))
    rows.sort(key=lambda r: -r[5])
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("lex\tpos\tstem\tnew_stem\tnew_lex\tcount\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")
    print(f"{len(rows)} variants written to {a.out}; ambiguous forms skipped: "
          f"{len(skipped)} ({sum(skipped.values())} tokens), e.g. {' '.join(w for w, _ in skipped.most_common(12))}")
    for r in rows[:40]:
        print("  ", r)


if __name__ == "__main__":
    main()
