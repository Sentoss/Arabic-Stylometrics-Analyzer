"""Count fully vowelled word forms in the full Tashkeela corpus (Classical
books only: the top-level texts, not texts.txt/msa), as the attestation list
for adding Lane's words. Same key as diac_freq.py (case/mood ending dropped).

usage: python layer2/lane/tashkeela_forms.py TASHKEELA_DIR/texts.txt OUT.tsv
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probe_labels import MARK, WORD  # noqa: E402
from diac_freq import key  # noqa: E402


def fully_vowelled(w):
    """Every letter that needs a mark has one (one may be missing). Long-vowel
    letters need none: ا ى آ, and و / ي after ُ / ِ (فَاعْلَمُوا counts)."""
    need, prev = 0, ""
    for i, ch in enumerate(w):
        if MARK.match(ch):
            continue
        nxt = w[i + 1] if i + 1 < len(w) else ""
        nxt_letter = next((c for c in w[i + 1:] if not MARK.match(c)), "")
        long_vowel = ch in "اىآٱ" or (ch == "و" and prev == "ُ") or (ch == "ي" and prev == "ِ") \
            or nxt_letter == "ا" or (ch == "ل" and i > 0 and w[i - 1] in "اٱ" and i <= 3)
        if not long_vowel and not MARK.match(nxt or " "):
            need += 1
        prev = w[i - 1] if i else ""
        prev = ""
        j = i + 1
        while j < len(w) and MARK.match(w[j]):
            prev = w[j]
            j += 1
    return need <= 1


def main(src, out):
    c, books, words, voc = Counter(), 0, 0, 0
    for p in sorted(Path(src).glob("*.txt")):        # top level only: Classical books
        books += 1
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                for w in WORD.findall(line):
                    words += 1
                    if fully_vowelled(w):
                        voc += 1
                        c[key(w)] += 1
        print(books, p.name, words, voc, len(c), flush=True)
    with open(out, "w", encoding="utf-8") as f:
        for k, n in c.most_common():
            f.write(f"{k}\t{n}\n")
    print("books", books, "words", words, "fully vowelled", voc, "forms", len(c))


if __name__ == "__main__":
    main(*sys.argv[1:3])
