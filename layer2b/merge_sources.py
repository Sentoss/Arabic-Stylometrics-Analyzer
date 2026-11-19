"""Merge converted iʿrāb sources into one treebank, field by field, by vote.

For every MASAQ word and every field (function; link = relation + target), each
source that analyses the word casts a vote:

  agreed     two or more sources, none against (a more specific label counts as
             agreeing with a general one: خبر إنّ with خبر)
  single     one source only
  disputed   sources disagree; every source's answer is kept for review

The merged record keeps `votes` ({source: value}) so any source can be dropped
and the treebank rebuilt (licensing, or a source found unreliable).

usage: python layer2b/merge_sources.py OUT.jsonl.gz NAME=FILE.jsonl.gz [NAME=FILE ...]
"""
import gzip
import json
import sys
from collections import Counter, defaultdict

MERGE = {"نائب الفاعل": "نائب فاعل", "صفه": "نعت", "مضارع": "فعل مضارع", "امر": "فعل امر",
         "موصول": "اسم موصول", "مفعول لاجله": "مفعول لأجله", "مبتدا مؤخر": "مبتدا", "خبر مقدم": "خبر",
         "مفعول به اول": "مفعول به", "عطف بيان": "بدل"}
GENERAL = {"خبر ان": "خبر", "خبر كان": "خبر", "خبر لا": "خبر", "خبر ما": "خبر", "خبر ثان": "خبر",
           "مفعول به ثان": "مفعول به", "فعل ماض": "فعل", "فعل مضارع": "فعل", "فعل امر": "فعل"}
PARTICLE_ROLES = {"NON_INFLECT", "PREP", "EXPLET", "CONJ", "CERT_PART", "NEG"}
UNINFORMATIVE = {None, "", "حرف", "حرف جر", "ضمير", "اسم موصول", "اسم اشاره", "اسم استفهام"}


def norm_fn(f):
    return MERGE.get(f, f)


def compatible(a, b):
    """The more specific of two compatible labels, or None."""
    if a == b:
        return a
    if GENERAL.get(a) == b:
        return a
    if GENERAL.get(b) == a:
        return b
    return None


def link(w, ws):
    def at(k):
        return [ws[k]["verse"], ws[k]["n"]] if isinstance(k, int) and 0 <= k < len(ws) else None
    if "mutaallaq" in w:
        m = w["mutaallaq"]
        return ["تعلق", "محذوف:" + m["محذوف"]] if isinstance(m, dict) else ["تعلق", at(m)]
    if "matbu" in w:
        return [w["matbu"]["type"], at(w["matbu"]["word"])]
    if "sahib_hal" in w:
        return ["حال", at(w["sahib_hal"])]
    return None


def vote(values):
    """values: {source: value}. Returns (value, status)."""
    vals = {s: v for s, v in values.items() if v is not None}
    if not vals:
        return None, "none"
    if len(vals) == 1:
        return next(iter(vals.values())), "single"
    items = list(vals.values())
    best = items[0]
    for v in items[1:]:
        best = compatible(best, v) if isinstance(v, str) and isinstance(best, str) else (best if best == v else None)
        if best is None:
            break
    if best is not None:
        return best, "agreed"
    counts = Counter(json.dumps(v, ensure_ascii=False) for v in vals.values())
    top, n = counts.most_common(1)[0]
    if n * 2 > len(vals) and len(vals) >= 3:
        return json.loads(top), "majority"
    return None, "disputed"


def main():
    out = sys.argv[1]
    sources = [a.split("=", 1) for a in sys.argv[2:]]
    words = {}                                  # (sura, verse, n) -> merged record
    order = []
    for name, path in sources:
        for line in gzip.open(path, "rt", encoding="utf-8"):
            d = json.loads(line)
            ws = d["words"]
            for w in ws:
                k = (d["sura"], w["verse"], w["n"])
                if k not in words:
                    words[k] = {"sura": d["sura"], "verse": w["verse"], "n": w["n"], "word": w["word"],
                                "role": w["role"], "fn_votes": {}, "link_votes": {}, "gloss": {}, "src": {}}
                    order.append(k)
                r = words[k]
                f = norm_fn(w.get("function"))
                if f == "توكيد" and w["role"] in PARTICLE_ROLES:
                    f = None                   # «زائدة لتأكيد النفي»: the particle's meaning, not a تابع
                if f not in UNINFORMATIVE:
                    r["fn_votes"][name] = f
                lk = link(w, ws)
                if lk:
                    r["link_votes"][name] = lk
                if w.get("jadwal"):
                    r["gloss"][name] = w["jadwal"][:200]
                if w.get("src"):
                    r["src"][name] = w["src"]
    stats = Counter()
    with gzip.open(out, "wt", encoding="utf-8") as o:
        for k in sorted(order):
            r = words[k]
            r["function"], r["fn_status"] = vote(r["fn_votes"])
            r["link"], r["link_status"] = vote(r["link_votes"])
            stats["fn_" + r["fn_status"]] += 1
            stats["link_" + r["link_status"]] += 1
            o.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(order)
    print(f"{n} words")
    for k, v in sorted(stats.items()):
        print(f"  {k:18} {v:6}  {v / n:.1%}")


if __name__ == "__main__":
    main()
