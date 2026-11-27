"""Turn hand annotations (the "Iʿrāb annotation" page) into i'rab training rows.

Input: the batch file written by tools/select_annotation.py and the page's
saved documents (collection "ann", exported as a JSON list of {id, data}).
Only words the annotator checked are labelled; the rest stay -1. Skipped
sentences are dropped. Sentences in the test split go to prose_test.jsonl.gz
and are never trained on.

usage: python layer2b/annotations.py BATCH.json ANN.json DATASET_DIR [--repeat 3]
  adds the training sentences to DATASET_DIR/train.jsonl.gz (repeated), writes
  DATASET_DIR/prose_test.jsonl.gz, and extends labels.json with new labels.
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_dataset import NO_POSITION, PARTICLE_AS_FUNC, collapse

UNUSABLE = {"أخرى (في الملاحظات)", ""}
CLITICS = ("conj", "part", "subj", "enc")
# fields added with the shared label scheme (claude/irab-label-scheme.md); a word saved
# before them has no "nf" and its new fields are left unlabelled
NEW = ("ptype", "mn", "cm", "clause")
PAGE_KEY = {"ptype": "pt", "mn": "mn", "cm": "cm", "clause": "cl"}
VERB_CM = {"رفع": "مضارع مرفوع", "نصب": "مضارع منصوب", "جزم": "مضارع مجزوم"}


def page_cm(v, func):
    """The page's رفع/نصب/جر/جزم/مبني -> the model's case/mood label."""
    if not v:
        return None
    if v == "مبني":
        return "none"
    if func == "فعل مضارع":
        return VERB_CM.get(v)
    return v if v in ("رفع", "نصب", "جر") and func not in ("فعل ماض", "فعل أمر") else None


def rows(batch, ann, slots):
    by_id = {d["id"]: d["data"] for d in ann}
    for s in batch["sentences"]:
        a = by_id.get(s["id"])
        if not a or a.get("skip"):
            continue
        n = len(s["words"])
        words = a["words"]
        if len(words) != n:
            continue
        r = {"src": "prose", "id": s["id"], "split": s["split"], "w": s["words"], "role": [None] * n,
             "func": [None] * n, "rel": [None] * n, "target": [None] * n, "hidden": [None] * n,
             **{k: [None] * n for k in CLITICS + NEW}}
        checked = 0
        for i, w in enumerate(words):
            if not (w.get("ok") or a.get("all")):
                continue
            checked += 1
            f = w.get("f")
            f = NO_POSITION if f in PARTICLE_AS_FUNC or f == "لا وظيفة" else collapse(f)  # older saves
            r["func"][i] = None if f in UNUSABLE else f
            r["rel"][i] = w.get("r") or "none"
            t = w.get("t", "")
            if r["rel"][i] != "none" and t != "":
                r["target"][i] = int(t) if str(t).isdigit() else (n + slots.index(t) if t in slots else None)
            if w.get("h") is not None and w.get("f") in ("فعل ماض", "فعل مضارع", "فعل أمر"):
                r["hidden"][i] = 1 if w["h"] else 0
            for k in CLITICS:                 # the attached pieces ("none" = the word has none)
                if w.get(k):
                    r[k][i] = collapse({"اسم كان": "اسم فعل ناسخ"}.get(w[k], w[k]))
            if w.get("nf"):                   # checked on the page with the new fields
                for k in NEW:
                    v = w.get(PAGE_KEY[k])
                    if k == "cm":
                        v = page_cm(v, r["func"][i])
                    if v:
                        r[k][i] = v
        if checked:
            yield r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch"); ap.add_argument("ann"); ap.add_argument("dataset")
    ap.add_argument("--repeat", type=int, default=3)
    a = ap.parse_args()
    d = Path(a.dataset)
    labels = json.load(open(d / "labels.json", encoding="utf-8"))
    batch = json.load(open(a.batch, encoding="utf-8"))
    ann = json.load(open(a.ann, encoding="utf-8"))
    out = {"train": [], "test": []}
    added = set()
    for r in rows(batch, ann, labels["slots"]):
        for k, key in (("func", "funcs"), ("rel", "rels")) + tuple((c, c) for c in CLITICS + NEW):
            ids = []
            for x in r[k]:
                if x is None:
                    ids.append(-1)
                    continue
                if x not in labels[key]:
                    labels[key].append(x); added.add(x)
                ids.append(labels[key].index(x))
            r[k] = ids
        r["role"] = [-1] * len(r["w"])
        r["target"] = [-1 if t is None else t for t in r["target"]]
        r["hidden"] = [-1 if h is None else h for h in r["hidden"]]
        out[r.pop("split")].append(r)
    with gzip.open(d / "train.jsonl.gz", "at", encoding="utf-8") as f:
        for r in out["train"]:
            for _ in range(a.repeat):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with gzip.open(d / "prose_test.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in out["test"]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(labels, open(d / "labels.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print({"train sentences": len(out["train"]), "test sentences": len(out["test"]),
           "train words labelled": sum(1 for r in out["train"] for x in r["func"] if x >= 0),
           "test words labelled": sum(1 for r in out["test"] for x in r["func"] if x >= 0),
           "new labels": sorted(added)})


if __name__ == "__main__":
    main()
