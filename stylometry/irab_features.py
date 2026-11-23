"""Stylometric features from the layer-2b i'rab model's output (layer2b/irab.py).

Three families, counted per chunk from its .jsonl (one tagged segment per line):

  nfunc  the naḥw function of each word, and of each pair of neighbours
         ("فعل ماض>فاعل", "مبتدأ>خبر"): which constructions the author builds.
  nlink  the links: kind, direction and distance ("تعلق:-1", "نعت:+2-3"), what
         a link attaches to ("تعلق→فعل ماض", "تعلق→محذوف خبر"), and whether a
         verb's subject is hidden ("hidden:yes").
  nrole  MASAQ's i'rab roles, single and in neighbouring pairs.

Words, lemmas and particles are not used: the features describe construction.
Words the model is unsure of (function probability below MIN_P) are left out of
the function features, so guesses do not swamp them.
"""
import json
from collections import Counter

MIN_P = 0.5


def dist_bin(d):
    a = abs(d)
    b = "1" if a == 1 else "2-3" if a <= 3 else "4-7" if a <= 7 else "8+"
    return ("+" if d > 0 else "-") + b


def read_tagged(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def irab_features(segments, min_p=MIN_P):
    nfunc, nlink, nrole = Counter(), Counter(), Counter()
    for seg in segments:
        seg = [w for w in seg if "func" in w]
        funcs = [w["func"] if w.get("p_func", 1) >= min_p else None for w in seg]
        for i, w in enumerate(seg):
            f = funcs[i]
            if f:
                nfunc["F:" + f] += 1
                if i and funcs[i - 1]:
                    nfunc[f"{funcs[i - 1]}>{f}"] += 1
            nrole["R:" + w["role"]] += 1
            if i:
                nrole[f"{seg[i - 1]['role']}>{w['role']}"] += 1
            if w["rel"] != "none" and w.get("target") is not None:
                t = w["target"]
                if isinstance(t, int):
                    nlink[f"{w['rel']}:{dist_bin(t - i)}"] += 1
                    tf = funcs[t] if t < len(funcs) else None
                    if tf:
                        nlink[f"{w['rel']}→{tf}"] += 1
                else:
                    nlink[f"{w['rel']}→{t}"] += 1
            if w.get("hidden") is not None:
                nlink["hidden:" + ("yes" if w["hidden"] else "no")] += 1
    return {"nfunc": nfunc, "nlink": nlink, "nrole": nrole}
