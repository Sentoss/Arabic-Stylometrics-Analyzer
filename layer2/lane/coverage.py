"""How much of Lane is already in our dictionary? For each Lane word (nouns,
plurals, maṣdars, feminines; verbs separately), compare its vowelling with
the readings our analyzer gives its bare form.

  known      a reading with the same vowelling exists
  shared     the bare form has readings, but not this vowelling
  new        the bare form has no reading at all (only backoff)

usage: python layer2/lane/coverage.py lane_rows.jsonl OUT.jsonl
"""
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from diac_freq import key  # noqa: E402
from probe_labels import MARK  # noqa: E402
from supplement import ClassicalAnalyzer, VARIANTS  # noqa: E402

an = ClassicalAnalyzer(passives=True, variants=VARIANTS)


def bare(w):
    return MARK.sub("", w).replace("ٱ", "ا")


def lex_key(w):
    k = key(w)
    if k.endswith("ة") or k.endswith("ى"):
        return k
    return k


def main(src, out):
    stats = Counter()
    cache = {}
    with open(out, "w", encoding="utf-8") as f:
        for line in open(src, encoding="utf-8"):
            r = json.loads(line)
            w = r["word"].strip()
            if not w or " " in w or len(bare(w)) < 2:
                continue
            if r["kind"] == "verb":
                if w.endswith("هُ"):                     # Lane writes transitive verbs with the object هُ
                    w = w[:-2]
            b = bare(w)
            if b not in cache:
                cache[b] = [a for a in an.analyze(b) if a.get("source") != "backoff"]
            reads = cache[b]
            k = key(w)
            if not reads:
                status = "new"
            elif any(key(a["diac"]) == k for a in reads):
                status = "known"
            else:
                status = "shared"
            n_marks = len(MARK.findall(w))
            if n_marks == 0:
                status += "_unvowelled"
            stats[(r["kind"], status)] += 1
            f.write(json.dumps({**r, "w": w, "bare": b, "key": k, "status": status,
                                "readings": sorted({a["diac"] for a in reads})[:6]}, ensure_ascii=False) + "\n")
    for kind in ("noun", "plural", "masdar", "fem", "verb"):
        tot = sum(v for (k, s), v in stats.items() if k == kind)
        print(kind, tot, {s: v for (k, s), v in sorted(stats.items()) if k == kind})


if __name__ == "__main__":
    main(*sys.argv[1:3])
