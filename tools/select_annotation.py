"""Choose real corpus sentences for hand annotation of i'rab.

Sentences are cut from authorial chunks (quotations, isnads and verse already
removed) at real sentence ends, 6-30 words long. Each is tagged by the current
i'rab model; sentences are then chosen for the constructions the model gets
wrong (by word form) and for words it is unsure of, spread over genres and
authors. Each sentence marks 1-3 focus words: the trigger word(s) and the
least certain words.

usage: python tools/select_annotation.py MODEL DATA_DIR OUT.json [--n 100 --test 30]
"""
import argparse
import collections
import csv
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from layer2b.irab import IrabTagger, VOWELS  # noqa: E402

STOP = set(".!?؟")
# construction -> regex on the unvowelled word (with its clitics)
TRIGGERS = {
    "لولا": r"^[وف]?لولا$",
    "كأنّ": r"^[وف]?كأن",
    "أمّا … فـ": r"^[وف]?أما$",
    "كم": r"^[وف]?كم$",
    "أنّ / أنْ": r"^[وف]?(?:ب|ل)?أن(?:ه|ها|هم|ك|ني|نا)?$",
    "formula": r"^(?:صلى|رضي|رحمه|رحمهم|رحمها|كرم|عز|تعالى|تبارك|السلام|سبحانه)$",
    "ما (verb-like)": r"^[وف]?ما$",
    "إذ / إذا": r"^[وف]?إذا?$",
    "ليت / لعلّ": r"^[وف]?(?:ليت|لعل)",
    "لمّا": r"^[وف]?لما$",
}
# constructions recognised by the model's own (weak) labels
WEAK = {"حال", "بدل", "توكيد", "نائب فاعل", "مفعول به أول", "مفعول به ثان", "تمييز",
        "مفعول مطلق", "المفعول لأجله", "مستثني", "خبر كان", "اسم كان"}


def sentences(tokens, lo=6, hi=30):
    """Full sentences: between two sentence-final marks, with no removed
    material (<ISNAD>, <H-BLOCK>, <Q>…) inside."""
    out, cur, started, broken = [], [], False, False
    for t in tokens:
        if t["status"] == "not_word":
            if t["raw"] in STOP:
                if started and not broken and lo <= len(cur) <= hi:
                    out.append(cur)
                cur, started, broken = [], True, False
            elif "<" in t["raw"] or re.search(r"[A-Za-z]", t["raw"]):
                broken = True      # a removed quotation, isnad or verse: the sentence has a gap
            continue
        w = VOWELS.sub("", t["restored"] or t["raw"])
        if re.fullmatch(r"[ء-ي]+", w):
            cur.append({"w": w, "shown": t["restored"] or t["raw"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("data"); ap.add_argument("out")
    ap.add_argument("--sample", default="sample_syn.tsv")
    ap.add_argument("--n", type=int, default=100); ap.add_argument("--test", type=int, default=30)
    ap.add_argument("--pool", type=int, default=2500); ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    root = Path(a.data)
    meta = list(csv.DictReader(open(root / a.sample, encoding="utf-8"), delimiter="\t"))
    trig = {k: re.compile(v) for k, v in TRIGGERS.items()}

    pool = []
    for m in meta:
        p = root / "analyzed_l2" / f"{m['chunk_id']}.tsv"
        if not p.exists():
            continue
        toks = list(csv.DictReader(open(p, encoding="utf-8"), delimiter="\t"))
        for s in sentences(toks):
            hits = sorted({k for k, r in trig.items() for x in s if r.match(x["w"])})
            pool.append({"chunk": m["chunk_id"], "author": m["author"], "genre": m["genre"],
                         "words": s, "triggers": hits})
    rnd.shuffle(pool)
    with_t = [p for p in pool if p["triggers"]]
    without = [p for p in pool if not p["triggers"]]
    pool = with_t[: int(a.pool * 0.7)] + without[: a.pool - min(len(with_t), int(a.pool * 0.7))]
    print(f"pool: {len(pool)} sentences ({len(with_t)} with a trigger word available)", flush=True)

    tagger = IrabTagger(a.model)
    tagged = tagger.tag_many([[x["w"] for x in p["words"]] for p in pool])
    for p, t in zip(pool, tagged):
        p["model"] = t
        p["weak"] = sorted({w["func"] for w in t if w.get("func") in WEAK})
        p["unsure"] = sum(1 for w in t if w.get("p_func", 1) < 0.6)

    # choose: round-robin over constructions, preferring unsure sentences,
    # at most 6 sentences per author and at most 30% from one genre
    groups = collections.defaultdict(list)
    for i, p in enumerate(pool):
        for k in p["triggers"] + ["weak:" + w for w in p["weak"]]:
            groups[k].append(i)
        if p["unsure"] >= 2:
            groups["unsure"].append(i)
    for k in groups:
        groups[k].sort(key=lambda i: -pool[i]["unsure"])
    chosen, per_author, per_genre = [], collections.Counter(), collections.Counter()
    genre_cap = int(a.n * 0.3)
    keys = sorted(groups, key=lambda k: len(groups[k]))
    while len(chosen) < a.n and any(groups[k] for k in keys):
        for k in keys:
            while groups[k]:
                i = groups[k].pop(0)
                p = pool[i]
                if i in chosen or per_author[p["author"]] >= 6 or per_genre[p["genre"]] >= genre_cap:
                    continue
                chosen.append(i); per_author[p["author"]] += 1; per_genre[p["genre"]] += 1
                p["reason"] = k
                break
            if len(chosen) >= a.n:
                break

    out = []
    for n, i in enumerate(chosen):
        p = pool[i]
        focus = [j for j, x in enumerate(p["words"]) if any(trig[k].match(x["w"]) for k in p["triggers"])][:2]
        unsure = sorted((j for j, w in enumerate(p["model"]) if w.get("p_func", 1) < 0.6 and j not in focus),
                        key=lambda j: p["model"][j]["p_func"])
        focus = sorted((focus + unsure)[:3]) or [min(range(len(p["model"])), key=lambda j: p["model"][j].get("p_func", 1))]
        out.append({"id": f"a{n + 1:03d}", "chunk": p["chunk"], "author": p["author"], "genre": p["genre"],
                    "reason": p["reason"], "words": [x["w"] for x in p["words"]],
                    "shown": [x["shown"] for x in p["words"]], "focus": focus,
                    "model": [{k: w.get(k) for k in ("func", "p_func", "rel", "target", "hidden", "role")}
                              for w in p["model"]]})
    # test / train split, stratified by reason
    by = collections.defaultdict(list)
    for s in out:
        by[s["reason"]].append(s)
    test_ids, order = set(), [s for k in sorted(by) for s in by[k]]
    for s in order[::max(1, len(order) // a.test)][:a.test]:
        test_ids.add(s["id"])
    for s in out:
        s["split"] = "test" if s["id"] in test_ids else "train"
    rnd.shuffle(out)                       # mixed order on the page
    json.dump({"labels": tagger.labels, "sentences": out}, open(a.out, "w", encoding="utf-8"),
              ensure_ascii=False)
    print(json.dumps({"chosen": len(out), "test": len(test_ids),
                      "reasons": collections.Counter(s["reason"] for s in out),
                      "genres": collections.Counter(s["genre"] for s in out),
                      "authors": len({s["author"] for s in out})}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
