"""Training data for the i'rab model (2b), from jadwal_convert.py's output.

Per verse block, for every word (unvowelled):
  role    MASAQ's role of the stem (all words)
  func    الجدول's naḥw function (where الجدول glosses the word), e.g. حال، تمييز
  rel     the word's link in naḥw terms: تعلق (a جار ومجرور / ظرف and its
          متعلَّق به), نعت / عطف / بدل / توكيد (a تابع and its متبوع), حال (a
          حال and صاحب الحال), or none; unknown where الجدول is silent
  target  the word it links to, or one of the "محذوف" slots placed after the
          words: محذوف خبر / صفة / حال / صلة / غير ذلك
  hidden  for verbs: 1 if the subject is a ضمير مستتر, 0 if expressed

Splits by sūra: test = Yūsuf (12), al-Kahf (18), al-Naba' (78);
validation = Maryam (19), Ṭā-Hā (20), Yā-Sīn (36); the rest is training data.

Internal use only (derived from الجدول in part).

usage: python layer2b/make_dataset.py CONVERTED.jsonl.gz OUT_DIR
"""
import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path

DIAC = re.compile("[ً-ْٰـۡ۟ۥۦ۪-ۭٓ-ٟ]")
REL = ["none", "تعلق", "نعت", "عطف", "بدل", "توكيد", "حال"]
SLOTS = ["خبر", "صفة", "حال", "صلة", "?"]
TEST, VAL = {12, 18, 78}, {19, 20, 36}
CANON = {"نائب الفاعل": "نائب فاعل", "صفه": "نعت", "مضارع": "فعل مضارع", "امر": "فعل امر",
         "موصول": "اسم موصول", "مفعول لاجله": "مفعول لأجله"}


def bare(w):
    return DIAC.sub("", w).replace("ٱ", "ا")


def word_targets(r, n):
    """(rel id or -1, target index or -1)."""
    if "mutaallaq" in r:
        m = r["mutaallaq"]
        if isinstance(m, dict):
            return REL.index("تعلق"), n + SLOTS.index(m["محذوف"] if m["محذوف"] in SLOTS else "?")
        return REL.index("تعلق"), m
    if "matbu" in r:
        return REL.index(r["matbu"]["type"]), r["matbu"]["word"]
    if "sahib_hal" in r:
        return REL.index("حال"), r["sahib_hal"]
    f = r.get("function") or ""
    by_func = {"نعت": "نعت", "صفه": "نعت", "معطوف": "عطف", "بدل": "بدل", "عطف بيان": "بدل",
               "توكيد": "توكيد", "حال": "حال", "جار ومجرور": "تعلق", "ظرف زمان": "تعلق", "ظرف مكان": "تعلق"}
    if f in by_func:
        return REL.index(by_func[f]), -1          # the link type is known, its target is not
    if "jadwal" in r:
        return REL.index("none"), -1
    return -1, -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("converted")
    ap.add_argument("out_dir")
    a = ap.parse_args()
    rows = [json.loads(l) for l in gzip.open(a.converted, "rt", encoding="utf-8")]
    for r in rows:
        for w in r["words"]:
            if w.get("function"):
                w["function"] = CANON.get(w["function"], w["function"])
    roles = sorted({w["role"] for r in rows for w in r["words"]})
    funcs = Counter(w["function"] for r in rows for w in r["words"] if w.get("function"))
    funcs = sorted(f for f, c in funcs.items() if c >= 3)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {s: gzip.open(out / f"{s}.jsonl.gz", "wt", encoding="utf-8") for s in ("train", "val", "test")}
    stats = Counter()
    for r in rows:
        split = "test" if r["sura"] in TEST else "val" if r["sura"] in VAL else "train"
        ws = r["words"]
        n = len(ws)
        item = {"sura": r["sura"], "verses": r["verses"], "w": [bare(w["word"]) for w in ws],
                "role": [roles.index(w["role"]) for w in ws],
                "func": [funcs.index(w["function"]) if w.get("function") in funcs else -1 for w in ws],
                "rel": [], "target": [],
                "hidden": [int(w["muqaddar"]["مستتر"]) if "muqaddar" in w else -1 for w in ws]}
        for w in ws:
            rel, tgt = word_targets(w, n)
            item["rel"].append(rel)
            item["target"].append(tgt)
        files[split].write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
        stats[split + "_blocks"] += 1
        stats[split + "_words"] += n
        stats[split + "_links"] += sum(t >= 0 for t in item["target"])
    for f in files.values():
        f.close()
    json.dump({"roles": roles, "funcs": funcs, "rels": REL, "slots": SLOTS}, open(out / "labels.json", "w"),
              ensure_ascii=False, indent=1)
    print(dict(stats), f"roles {len(roles)}, functions {len(funcs)}")


if __name__ == "__main__":
    main()
