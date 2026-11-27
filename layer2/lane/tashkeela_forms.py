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


def main(src, out):
    c, books, words, voc = Counter(), 0, 0, 0
    for p in sorted(Path(src).glob("*.txt")):        # top level only: Classical books
        books += 1
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                for w in WORD.findall(line):
                    words += 1
                    if len(MARK.findall(w)) >= len(MARK.sub("", w)) - 1:
                        voc += 1
                        c[key(w)] += 1
        print(books, p.name, words, voc, len(c), flush=True)
    with open(out, "w", encoding="utf-8") as f:
        for k, n in c.most_common():
            f.write(f"{k}\t{n}\n")
    print("books", books, "words", words, "fully vowelled", voc, "forms", len(c))


if __name__ == "__main__":
    main(*sys.argv[1:3])
