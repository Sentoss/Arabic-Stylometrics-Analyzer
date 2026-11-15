"""Synthetic evaluation: corrupt clean Classical text with printing
conventions, restore it, and measure how much comes back.

usage: python evaluate.py CLEAN.txt [--words N] [--no-prior]

Corruption (all at 100%, i.e. harsher than most real editions):
  hamza on alif dropped  أ/إ -> ا   (anywhere in the word; آ is kept, as old prints do)
  final ي printed ى
  final ة printed ه
"""
import argparse
import re
from collections import Counter

from restorer import Restorer
from restorer.text import bare

CONVS = {"hamza_dropped_on_alif": True, "final_ya_as_alif_maqsura": True,
         "ha_for_ta_marbuta": True}


def corrupt_word(w):
    w = w.replace("أ", "ا").replace("إ", "ا")
    if w.endswith("ي"):
        w = w[:-1] + "ى"
    if w.endswith("ة"):
        w = w[:-1] + "ه"
    return w


def eligible(w):
    """Would a restorer have to make a decision about this written word?"""
    return "ا" in w or w.endswith("ى") or w.endswith("ه")


def load_clean(path, n_words):
    lines, count = [], 0
    for line in open(path, encoding="utf-8"):
        line = bare(line.strip())
        if not line:
            continue
        lines.append(line)
        count += len(line.split())
        if count >= n_words:
            break
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clean")
    ap.add_argument("--words", type=int, default=3000)
    ap.add_argument("--no-prior", action="store_true")
    a = ap.parse_args()

    gold_text = load_clean(a.clean, a.words)
    corrupted = re.sub(r"[ء-ي]+", lambda m: corrupt_word(m.group(0)), gold_text)
    R = Restorer(prior=None) if a.no_prior else Restorer()
    _, _, res = R.run(corrupted, overrides=CONVS)
    gold = [m.group(0) for m in re.finditer(r"[ء-غف-ي]+", gold_text)]
    words = [r for r in res if r.status != "not_word"]
    assert len(words) == len(gold), (len(words), len(gold))

    c = Counter()
    errors = Counter()
    for g, r in zip(gold, words):
        w, out = bare(r.raw), bare(r.restored)
        if not eligible(w):
            continue
        kind = "corrupted" if w != g else "untouched"
        c[kind] += 1
        c[kind + "_baseline_ok"] += (w == g)
        if out == g:
            c[kind + "_ok"] += 1
        else:
            errors[(w, out, g)] += 1
        c[f"conf_{r.confidence}"] += 1
        c[f"conf_{r.confidence}_ok"] += (out == g)

    tot = c["corrupted"] + c["untouched"]
    ok = c["corrupted_ok"] + c["untouched_ok"]
    base = c["corrupted_baseline_ok"] + c["untouched_baseline_ok"]
    print(f"prior: {'off' if a.no_prior else 'on'}   words: {len(gold)}   decisions: {tot}")
    print(f"accuracy on decisions: {ok/tot:.1%}   (do-nothing baseline {base/tot:.1%})")
    print(f"  corrupted words restored correctly: {c['corrupted_ok']}/{c['corrupted']}"
          f" = {c['corrupted_ok']/c['corrupted']:.1%}")
    print(f"  already-correct words left correct: {c['untouched_ok']}/{c['untouched']}"
          f" = {c['untouched_ok']/c['untouched']:.1%}")
    for k in ("certain", "high", "low", ""):
        n = c[f"conf_{k}"]
        if n:
            print(f"  confidence {k or 'none':8}: {n:5} words, {c[f'conf_{k}_ok']/n:.1%} correct")
    print("most frequent errors (written -> restored, gold):")
    for (w, out, g), n in errors.most_common(25):
        print(f"  {n:3}  {w} -> {out}   gold {g}")


if __name__ == "__main__":
    main()
