"""Syntactic features from dependency parses (CamelParser, CATiB scheme).

Three feature families, counted per chunk from its .conllx parse:

  dep    one feature per dependency: head POS, side of the dependent, relation
         and dependent POS.  VRB>SBJ:NOM = a noun subject after its verb;
         VRB<SBJ:NOM would be a subject before it;  NOM>IDF:NOM = an iḍāfa;
         ROOT:VRB / ROOT:NOM = what heads the segment.
  frame  the shape of each clause.  For a verb: its subject, objects and
         predicate in their order around it ("V S O", "V o" with a suffixed
         pronoun object, "Vp S" for a passive, "V" alone), and what introduces
         it ("open:ف>V", "open:أن>V", "open:لم>V").  For verbless clauses:
         "NOM P:NOM" (topic + predicate), "إن S P".
  dist   how far apart head and dependent are, per relation (SBJ:+1, OBJ:+2-3,
         MOD:-4-7 …): a measure of how long the author's constructions run.

Only particles are named by lemma; nouns and verbs are reduced to their POS,
so the features describe construction, not vocabulary.

usage (evaluation): python -m stylometry.syntax DATA_DIR [--noise] [--irab irab_tagged]
"""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CORE = {"SBJ": "S", "OBJ": "O", "PRD": "P", "TPC": "T"}
RELS = ["MOD", "OBJ", "IDF", "SBJ", "PRD", "TPC", "TMZ"]


def read_conllx(path):
    """List of sentences; each a list of dicts (id, form, lemma, pos, head, rel)."""
    sents, cur = [], []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            if cur:
                sents.append(cur)
                cur = []
            continue
        if line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) < 8 or not f[0].isdigit():
            continue
        cur.append({"id": int(f[0]), "form": f[1], "lemma": f[2], "pos": f[3] or "UNK",
                    "head": int(f[6]) if f[6].isdigit() else 0, "rel": f[7]})
    if cur:
        sents.append(cur)
    return sents


def dist_bin(d):
    a = abs(d)
    b = "1" if a == 1 else "2-3" if a <= 3 else "4-7" if a <= 7 else "8+"
    return ("+" if d > 0 else "-") + b


def particle(t):
    return t["form"].strip("+") if t["pos"] == "PRT" else t["pos"]


def syntax_features(sents):
    dep, frame, dist = Counter(), Counter(), Counter()
    for s in sents:
        by = {t["id"]: t for t in s}
        kids = defaultdict(list)
        for t in s:
            kids[t["head"]].append(t)
        for t in s:
            h = by.get(t["head"])
            if h is None:
                dep[f"ROOT:{t['pos']}"] += 1
                continue
            side = ">" if t["id"] > h["id"] else "<"
            dep[f"{h['pos']}{side}{t['rel']}:{t['pos']}"] += 1
            dist[f"{t['rel']}:{dist_bin(t['id'] - h['id'])}"] += 1
        for t in s:
            core = [k for k in kids[t["id"]] if k["rel"] in CORE]
            if t["pos"].startswith("VRB"):
                items = [(t["id"], "Vp" if t["pos"] == "VRB-PASS" else "V")]
                for k in core:
                    c = CORE[k["rel"]]
                    if k["form"].startswith("+"):
                        c = c.lower()                    # suffixed pronoun
                    elif k["pos"].startswith("VRB") or k["pos"] == "PRT":
                        c += "c"                         # a clause as object/predicate
                    items.append((k["id"], c))
                frame[" ".join(c for _, c in sorted(items))] += 1
                h = by.get(t["head"])
                frame[f"open:{particle(h) if h else 'ROOT'}>V"] += 1
            elif any(k["rel"] == "PRD" for k in core):
                if t["pos"] == "PRT":                    # إنّ and her sisters
                    items = [(t["id"], t["form"].strip("+"))] + [(k["id"], CORE[k["rel"]]) for k in core]
                    frame[" ".join(c for _, c in sorted(items))] += 1
                else:
                    for k in core:
                        if k["rel"] == "PRD":
                            frame[f"{t['pos']} P:{k['pos']}"] += 1
                    h = by.get(t["head"])
                    frame[f"open:{particle(h) if h else 'ROOT'}>N"] += 1
    return {"dep": dep, "frame": frame, "dist": dist}


def corrupt(sents, p, rng):
    """Give a share p of the tokens a wrong head and relation."""
    out = []
    for s in sents:
        new = []
        for t in s:
            if len(s) > 2 and rng.random() < p:
                t = dict(t)
                t["head"] = rng.choice([x["id"] for x in s if x["id"] not in (t["id"], t["head"])] + [0])
                t["rel"] = rng.choice(RELS)
            new.append(t)
        out.append(new)
    return out


def load(data_dir, sample="sample_syn.tsv", parsed="parsed", analyzed="analyzed_l2", noise=0.0, seed=7,
         irab=None):
    """(metas, feature dicts) with the usual families plus dep / frame / dist, and
    nfunc / nlink / nrole when irab names a folder of layer-2b output. With irab,
    only chunks that have both parses are kept, so the comparison is fair."""
    from .features import chunk_features, read_tokens
    from .irab_features import irab_features, read_tagged
    root = Path(data_dir)
    rng = random.Random(seed)
    metas, feats = [], []
    for r in csv.DictReader(open(root / sample, encoding="utf-8"), delimiter="\t"):
        pc, pa = root / parsed / f"{r['chunk_id']}.conllx", root / analyzed / f"{r['chunk_id']}.tsv"
        pi = root / irab / f"{r['chunk_id']}.jsonl" if irab else None
        if not (pc.exists() and pa.exists()) or (pi and not pi.exists()):
            continue
        sents = read_conllx(pc)
        if noise:
            sents = corrupt(sents, noise, rng)
        f = chunk_features(read_tokens(pa), (root / r["path"]).read_text(encoding="utf-8"))
        f.update(syntax_features(sents))
        if pi:
            f.update(irab_features(read_tagged(pi)))
        metas.append(r)
        feats.append(f)
    return metas, feats


CONFIGS = {
    "surface words (300)": [("surface", 300)],
    "function words + clitics (150)": [("function", 150)],
    "POS 1-3-grams (300)": [("pos", 300)],
    "lemma+function+POS+pattern": [("lemma", 300), ("function", 150), ("pos", 300), ("pattern", 200)],
    "syntax: dependencies (200)": [("dep", 200)],
    "syntax: clause frames (150)": [("frame", 150)],
    "syntax: distances (60)": [("dist", 60)],
    "syntax: all three": [("dep", 200), ("frame", 150), ("dist", 60)],
    "function words + syntax": [("function", 150), ("dep", 200), ("frame", 150), ("dist", 60)],
    "combined + syntax": [("lemma", 300), ("function", 150), ("pos", 300), ("pattern", 200),
                          ("dep", 200), ("frame", 150), ("dist", 60)],
}
SYN_ONLY = ["syntax: dependencies (200)", "syntax: clause frames (150)", "syntax: all three"]
IRAB_CONFIGS = {
    "iʿrāb: functions (200)": [("nfunc", 200)],
    "iʿrāb: links (150)": [("nlink", 150)],
    "iʿrāb: roles (150)": [("nrole", 150)],
    "iʿrāb: all three": [("nfunc", 200), ("nlink", 150), ("nrole", 150)],
    "CATiB + iʿrāb": [("dep", 200), ("frame", 150), ("dist", 60), ("nfunc", 200), ("nlink", 150), ("nrole", 150)],
    "function words + iʿrāb": [("function", 150), ("nfunc", 200), ("nlink", 150), ("nrole", 150)],
    "combined + iʿrāb": [("lemma", 300), ("function", 150), ("pos", 300), ("pattern", 200),
                         ("nfunc", 200), ("nlink", 150), ("nrole", 150)],
    "combined + syntax + iʿrāb": [("lemma", 300), ("function", 150), ("pos", 300), ("pattern", 200),
                                  ("dep", 200), ("frame", 150), ("dist", 60),
                                  ("nfunc", 200), ("nlink", 150), ("nrole", 150)],
}


def genre_split(metas, book_res):
    """(same-genre right, same-genre n, cross-genre right, cross-genre n)."""
    genres = defaultdict(set)
    for m in metas:
        if m["status"] == "secure":
            genres[m["author"]].add((m["book"], m["genre"]))
    bg = {m["book"]: m["genre"] for m in metas}
    out = [0, 0, 0, 0]
    for b, (truth, vote) in book_res.items():
        same = any(g == bg[b] for bb, g in genres[truth] if bb != b)
        k = 0 if same else 2
        out[k] += truth == vote
        out[k + 1] += 1
    return out


def main():
    from .evaluate import candidates, run
    from .methods import Families
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--noise", action="store_true", help="also run with corrupted parses")
    ap.add_argument("--irab", help="folder (inside DATA_DIR) of layer-2b output; adds the iʿrāb configurations")
    ap.add_argument("--out", default="reports/syntax/evaluation.json")
    a = ap.parse_args()
    configs = dict(CONFIGS, **(IRAB_CONFIGS if a.irab else {}))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    for noise in ([0.0, 0.1, 0.25, 0.5] if a.noise else [0.0]):
        metas, feats = load(a.data_dir, noise=noise, irab=a.irab)
        cands = candidates(metas)
        fams = Families(feats, families=sorted({f for spec in configs.values() for f, _ in spec}))
        tag = f"noise{int(noise * 100)}"
        results[tag] = {}
        print(f"== {tag}: {len(metas)} chunks, {len(cands)} candidates", flush=True)
        for name, spec in configs.items():
            if noise and name not in SYN_ONLY:
                continue
            res = run(metas, fams, spec, cands, methods=("logreg",))["logreg"]
            ch = float(np.mean(res["chunk"]))
            bk = float(np.mean([t == v for t, v in res["book"].values()]))
            sg, sn, cg, cn = genre_split(metas, res["book"])
            results[tag][name] = {"chunk": ch, "book": bk, "same": [sg, sn], "cross": [cg, cn],
                                  "books": {b: list(v) for b, v in res["book"].items()}}
            print(f"{name:34} chunks {ch:.1%}  books {bk:.1%}  same-genre {sg}/{sn}  cross-genre {cg}/{cn}",
                  flush=True)
        json.dump(results, open(out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
