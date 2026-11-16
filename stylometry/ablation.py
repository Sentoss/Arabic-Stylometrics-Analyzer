"""How much do morphological-analysis errors cost the attribution?

Before building layer 2, estimate what better analysis could buy. The same
leave-one-book-out attribution (logistic regression) is run on versions of
the analysed corpus that differ only in their morphology:

  base        the analyses as they are (CAMeL MSA disambiguator)
  no_unk      words the analyzer does not know are left out of the
              lemma / function / pattern counts (POS sequence marks them X)
  confident   only words whose bare form has a single reading (one lemma +
              POS) count; ambiguous words, where the MSA model had to choose,
              are left out (POS sequence marks them X)
  noise10/25/50  10% / 25% / 50% of the ambiguous words get a *wrong* reading instead
              (another reading of the same word, chosen at random)

If removing or corrupting analyses barely changes accuracy, a better
disambiguator will not change it much either. If added noise hurts, the
analysis matters and its current errors are costing us.

usage: python -m stylometry.ablation DATA_DIR [--sample sample40.tsv]
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.morphology.database import MorphologyDB

from .evaluate import candidates, run
from .features import AR_WORD, bare, chunk_features, read_tokens
from .methods import Families
import csv

CONFIGS = {
    "surface words (300) [control]": [("surface", 300)],
    "lemmas (300)": [("lemma", 300)],
    "function words + clitics (150)": [("function", 150)],
    "POS 1-3-grams (300)": [("pos", 300)],
    "patterns (300)": [("pattern", 300)],
    "lemma+function+POS+pattern": [("lemma", 300), ("function", 150), ("pos", 300), ("pattern", 200)],
}
CONDITIONS = ["base", "no_unk", "confident", "noise10", "noise25", "noise50"]


def readings_table(token_lists):
    an = Analyzer(MorphologyDB.builtin_db())
    types = {bare(t["restored"]) for toks in token_lists for t in toks}
    table = {}
    for w in types:
        if not AR_WORD.fullmatch(w or ""):
            continue
        seen = {}
        for a in an.analyze(w):
            if a.get("source") == "backoff":
                continue
            key = (a["lex"], a["pos"])
            seen.setdefault(key, {"lemma": a["lex"], "pos": a["pos"], "pattern": a.get("pattern", "")})
        table[w] = list(seen.values())
    return table


def transform(tokens, cond, table, rng):
    out = []
    for t in tokens:
        w = bare(t["restored"])
        rs = table.get(w, [])
        if cond == "no_unk" and not t["pos"]:
            out.append({**t, "lemma": "", "pos": "X", "pattern": "", "_skip": True})
            continue
        if cond == "confident" and len(rs) != 1:
            out.append({**t, "lemma": "", "pos": "X", "pattern": "", "_skip": True})
            continue
        if cond.startswith("noise") and len(rs) > 1 and t["pos"]:
            p = int(cond[5:]) / 100
            if rng.random() < p:
                wrong = [r for r in rs if (bare(r["lemma"]), r["pos"]) != (bare(t["lemma"]), t["pos"])]
                if wrong:
                    out.append({**t, **rng.choice(wrong)})
                    continue
        out.append(t)
    return out


def features(tokens, text):
    f = chunk_features([t for t in tokens if not t.get("_skip")], text)
    # POS n-grams keep the skipped words as X so sequences are not glued together
    if any(t.get("_skip") for t in tokens):
        seq = [t["pos"] or "UNK" for t in tokens if AR_WORD.fullmatch(bare(t["restored"]) or "")]
        f["pos"] = Counter()
        for n in (1, 2, 3):
            for i in range(len(seq) - n + 1):
                f["pos"]["_".join(seq[i:i + n])] += 1
    f["surface"] = chunk_features(tokens, "")["surface"]  # control is unaffected
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--sample", default="sample40.tsv")
    ap.add_argument("--out", default="reports/layer3_s40/ablation.json")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    a = ap.parse_args()
    root = Path(a.data_dir)
    rows = list(csv.DictReader(open(root / a.sample, encoding="utf-8"), delimiter="\t"))
    metas, toks, texts = [], [], []
    for r in rows:
        p = root / "analyzed" / f"{r['chunk_id']}.tsv"
        if p.exists():
            metas.append(r)
            toks.append(read_tokens(p))
            texts.append((root / r["path"]).read_text(encoding="utf-8") if r.get("path") else "")
    cands = candidates(metas)
    table = readings_table(toks)
    n_tok = sum(1 for ts in toks for t in ts if AR_WORD.fullmatch(bare(t["restored"]) or ""))
    amb = sum(1 for ts in toks for t in ts if len(table.get(bare(t["restored"]), [])) > 1)
    unk = sum(1 for ts in toks for t in ts if AR_WORD.fullmatch(bare(t["restored"]) or "") and not t["pos"])
    print(f"{len(metas)} chunks; words {n_tok}; ambiguous {amb / n_tok:.1%}; unknown {unk / n_tok:.1%}",
          flush=True)
    out = Path(a.out)
    results = json.load(open(out)) if out.exists() else {"runs": {}}
    results.update({"words": n_tok, "ambiguous": amb / n_tok, "unknown": unk / n_tok})
    for cond in a.conditions.split(","):
        rng = random.Random(7)
        fams = Families([features(transform(ts, cond, table, rng), tx) for ts, tx in zip(toks, texts)])
        results["runs"][cond] = {}
        for name, spec in CONFIGS.items():
            if name.endswith("[control]") and cond != "base":
                continue
            res = run(metas, fams, spec, cands, methods=("logreg",))
            ch = float(np.mean(res["logreg"]["chunk"]))
            bk = float(np.mean([t == v for t, v in res["logreg"]["book"].values()]))
            results["runs"][cond][name] = (round(ch, 3), round(bk, 3))
            print(f"{cond:9} {name:34} chunks {ch:.1%}  books {bk:.1%}", flush=True)
        del fams
        json.dump(results, open(out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
