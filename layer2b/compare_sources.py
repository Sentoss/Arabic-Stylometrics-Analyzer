"""Compare two converted iʿrāb sources word by word (same MASAQ words).

For every word both sources analyse, compare:
  function   the naḥw function (after merging spelling variants)
  link       the relation and its target: تعلق (word or محذوف+kind), نعت/عطف/بدل/توكيد (word), حال (صاحب)
Writes a JSON report with agreement rates and examples of each kind of disagreement.

usage: python layer2b/compare_sources.py A.jsonl.gz B.jsonl.gz OUT.json [--names الجدول دار_الصحابة]
"""
import argparse
import gzip
import json
import random
from collections import Counter, defaultdict

MERGE = {"نائب الفاعل": "نائب فاعل", "صفه": "نعت", "مضارع": "فعل مضارع", "امر": "فعل امر",
         "موصول": "اسم موصول", "مفعول لاجله": "مفعول لأجله", "مبتدا مؤخر": "مبتدا", "خبر مقدم": "خبر",
         "خبر ثان": "خبر", "مفعول به ثان": "مفعول به", "مفعول به اول": "مفعول به", "عطف بيان": "بدل",
         "فعل ماض": "فعل", "فعل مضارع": "فعل", "فعل امر": "فعل"}
VERB_ROLES = {"VERB", "IV", "CV", "IV_PASS", "VERB_PASS", "IV_COP", "CV_COP", "VERB_COP"}
SKIP = {None, "حرف", "حرف جر", "ضمير", "اسم موصول", "اسم اشاره", "اسم استفهام"}


def load(path):
    out = {}
    for line in gzip.open(path, "rt", encoding="utf-8"):
        d = json.loads(line)
        ws = d["words"]
        for i, w in enumerate(ws):
            out[(d["sura"], w["verse"], w["n"])] = (w, ws)
    return out


def link(w, ws):
    def at(k):
        return (ws[k]["verse"], ws[k]["n"]) if isinstance(k, int) and 0 <= k < len(ws) else None
    if "mutaallaq" in w:
        m = w["mutaallaq"]
        return ("تعلق", "محذوف:" + m["محذوف"]) if isinstance(m, dict) else ("تعلق", at(m))
    if "matbu" in w:
        return (w["matbu"]["type"], at(w["matbu"]["word"]))
    if "sahib_hal" in w:
        return ("حال", at(w["sahib_hal"]))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("out")
    ap.add_argument("--names", nargs=2, default=["A", "B"])
    a = ap.parse_args()
    A, B = load(a.a), load(a.b)
    common = sorted(set(A) & set(B))
    fn_n = fn_ok = 0
    by_fn = defaultdict(Counter)
    lk = defaultdict(Counter)
    examples = defaultdict(list)
    conf = Counter()
    for k in common:
        (wa, wsa), (wb, wsb) = A[k], B[k]
        mg = lambda f: MERGE.get(MERGE.get(f, f), MERGE.get(f, f))
        fa, fb = mg(wa.get("function")), mg(wb.get("function"))
        if wa.get("role") in VERB_ROLES:
            fa = fb = None                     # verbs: فعل either way; a pronoun subject's فاعل is not the verb's
        if fa not in SKIP and fb not in SKIP:
            fn_n += 1
            ok = fa == fb
            fn_ok += ok
            by_fn[fa]["n"] += 1
            by_fn[fa]["ok"] += ok
            if not ok:
                conf[(fa, fb)] += 1
                examples["function"].append({"loc": f"{k[0]}:{k[1]}:{k[2]}", "word": wa["word"],
                                             a.names[0]: fa, a.names[1]: fb,
                                             "gloss_" + a.names[0]: (wa.get("jadwal") or "")[:120],
                                             "gloss_" + a.names[1]: (wb.get("jadwal") or "")[:120],
                                             "verse": " ".join(w["word"] for w in wsa if w["verse"] == k[1])})
        la, lb = link(wa, wsa), link(wb, wsb)
        if la and lb and la[0] == lb[0]:
            lk[la[0]]["n"] += 1
            lk[la[0]]["ok"] += la[1] == lb[1]
            if la[1] != lb[1]:
                name = lambda x, ws: x if not isinstance(x, tuple) else next(
                    (w["word"] for w in ws if (w["verse"], w["n"]) == x), x)
                examples["link"].append({"loc": f"{k[0]}:{k[1]}:{k[2]}", "word": wa["word"], "type": la[0],
                                         a.names[0]: name(la[1], wsa), a.names[1]: name(lb[1], wsb),
                                         "verse": " ".join(w["word"] for w in wsa if w["verse"] == k[1])})
        elif la or lb:
            lk["type differs or one side only"]["n"] += 1
    random.seed(0)
    rep = {
        "words_in_both": len(common),
        "function": {"compared": fn_n, "agree": round(fn_ok / max(fn_n, 1), 3),
                     "by_function": {f: {"n": c["n"], "agree": round(c["ok"] / c["n"], 3)}
                                     for f, c in sorted(by_fn.items(), key=lambda x: -x[1]["n"]) if c["n"] >= 20},
                     "top_confusions": [[f"{x} / {y}", n] for (x, y), n in conf.most_common(25)]},
        "links": {t: ({"n": c["n"], "agree": round(c["ok"] / c["n"], 3)} if "ok" in c or c["n"] else c)
                  for t, c in lk.items()},
        "examples": {k: random.sample(v, min(40, len(v))) for k, v in examples.items()},
    }
    json.dump(rep, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: rep[k] for k in ("words_in_both", "links")}, ensure_ascii=False))
    print("function agree", rep["function"]["agree"], "of", fn_n)
    print(rep["function"]["top_confusions"][:12])


if __name__ == "__main__":
    main()
