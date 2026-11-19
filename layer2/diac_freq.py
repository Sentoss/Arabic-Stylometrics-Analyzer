"""Count vowelled word forms in Classical text (Tashkeela), to break exact ties
between readings the tagger cannot tell apart (same features, different
vowels: حَدَّثَنا / حَدَثَنا). Case endings and sukun are left out of the key.

usage: python layer2/diac_freq.py TASHKEELA_TRAIN.txt resources/diac_freq.tsv
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from probe_labels import MARK, WORD, norm  # noqa: E402

CASE = "ًٌٍَُِ"


def key(d):
    """Vowelled form without the final letter's marks (case/mood ending)."""
    d = norm(d)
    while d and MARK.match(d[-1]):
        d = d[:-1]
    out, marks = [], []
    for ch in d:                       # one order for the marks on a letter (shadda first)
        if MARK.match(ch):
            marks.append(ch)
        else:
            out += sorted(marks, key=lambda m: (m != "ّ", m)) + [ch]
            marks = []
    return "".join(out + sorted(marks, key=lambda m: (m != "ّ", m)))


def main():
    src, out = sys.argv[1], sys.argv[2]
    c = Counter()
    for line in open(src, encoding="utf-8"):
        for w in WORD.findall(line):
            if len(MARK.findall(w)) >= len(MARK.sub("", w)) - 1:      # fully vowelled words only
                c[key(w)] += 1
    with open(out, "w", encoding="utf-8") as f:
        for k, n in c.most_common():
            if n < 2:
                break
            f.write(f"{k}\t{n}\n")
    print(len(c), "forms")


if __name__ == "__main__":
    main()
