"""Leave-one-book-out evaluation of authorship attribution.

usage: python -m stylometry.evaluate DATA_DIR [--out reports/layer3]

Candidates: authors with at least two secure books in the analysed sample.
For every secure book of a candidate, everything else (other books of all
candidates) is training data; the held-out book's samples are attributed by
  delta     nearest author centroid, Burrows's Delta
  cosine    nearest author centroid, Cosine Delta
  logreg    logistic regression
  svm       linear SVM
Features and z-score statistics come from the training rows only. A book is
attributed by majority vote of its samples.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from stylometry.features import load_sample
from stylometry.methods import Families, zscore, burrows_delta, chi_square, cosine_delta, centroids, fit_predict

CONFIGS = {
    "surface words (300)": [("surface", 300)],
    "lemmas (300)": [("lemma", 300)],
    "function words + clitics (150)": [("function", 150)],
    "POS 1-3-grams (300)": [("pos", 300)],
    "patterns (300)": [("pattern", 300)],
    "char 3-4-grams (1000)": [("char", 1000)],
    "lemma+function+POS+pattern": [("lemma", 300), ("function", 150), ("pos", 300), ("pattern", 200)],
}
METHODS = ("delta", "cosine", "chi2", "logreg", "svm")


def candidates(metas):
    books = defaultdict(set)
    for m in metas:
        if m["status"] == "secure":
            books[m["author"]].add(m["book"])
    return sorted(a for a, b in books.items() if len(b) >= 2)


def genre_center(Z, metas, train):
    """Subtract each genre's mean profile (from training rows): what remains
    is how a sample deviates from the norm of its genre."""
    Z = Z.copy()
    genres = defaultdict(list)
    for i in train:
        genres[metas[i]["genre"]].append(i)
    means = {g: Z[ii].mean(0) for g, ii in genres.items()}
    for i, m in enumerate(metas):
        if m["genre"] in means:
            Z[i] -= means[m["genre"]]
    return Z


def run(metas, fams, spec, cands, center=False, methods=METHODS):
    idx = [i for i, m in enumerate(metas) if m["status"] == "secure" and m["author"] in cands]
    test_books = sorted({metas[i]["book"] for i in idx})
    res = {meth: {"chunk": [], "book": {}} for meth in methods}
    for tb in test_books:
        te = [i for i in idx if metas[i]["book"] == tb]
        tr = [i for i in idx if metas[i]["book"] != tb]
        X, _ = fams.select(spec, tr)
        Z = zscore(X, tr)
        if center:
            # all known samples (incl. single-work authors) define genre norms
            norm_rows = [i for i, m in enumerate(metas)
                         if m["status"] in ("secure", "single") and m["book"] != tb]
            Z = genre_center(Z, metas, norm_rows)
        ytr = [metas[i]["author"] for i in tr]
        truth = metas[te[0]]["author"]
        C = centroids(Z[tr], ytr, cands)
        preds = {}
        if "delta" in methods:
            preds["delta"] = [cands[j] for j in burrows_delta(Z[te], C).argmin(1)]
        if "chi2" in methods:
            Cr = centroids(X[tr], ytr, cands)        # raw relative frequencies
            preds["chi2"] = [cands[j] for j in chi_square(X[te], Cr).argmin(1)]
        if "cosine" in methods:
            preds["cosine"] = [cands[j] for j in cosine_delta(Z[te], C).argmin(1)]
        for meth in ("logreg", "svm"):
            if meth in methods:
                preds[meth] = list(fit_predict(meth, Z[tr], ytr, Z[te])[0])
        for meth, p in preds.items():
            res[meth]["chunk"] += [x == truth for x in p]
            vote = Counter(p).most_common(1)[0][0]
            res[meth]["book"][tb] = (truth, vote)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--analyzed", default="analyzed",
                    help="folder of analysed chunks in DATA_DIR (analyzed_l2: with the Classical supplement)")
    ap.add_argument("--sample", default="sample.tsv",
                    help="chunk list in DATA_DIR (e.g. sample12.tsv, sample40.tsv)")
    ap.add_argument("--out", default="reports/layer3")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    metas, feats = load_sample(a.data_dir, analyzed=a.analyzed, manifest=a.sample)
    cands = candidates(metas)
    fams = Families(feats)
    n_books = len({m["book"] for m in metas if m["status"] == "secure" and m["author"] in cands})
    print(f"{len(metas)} samples, {len(cands)} candidate authors, {n_books} test books")
    summary, detail = {}, {}
    for name, spec in CONFIGS.items():
        res = run(metas, fams, spec, cands)
        summary[name] = {}
        for meth in METHODS:
            ch = np.mean(res[meth]["chunk"])
            bk = np.mean([t == v for t, v in res[meth]["book"].values()])
            summary[name][meth] = (round(float(ch), 3), round(float(bk), 3))
        detail[name] = {meth: res[meth]["book"] for meth in METHODS}
        print(f"{name:34}", "  ".join(f"{m}: {summary[name][m][0]:.0%}/{summary[name][m][1]:.0%}"
                                         for m in METHODS), flush=True)
    json.dump({"candidates": cands, "summary": summary, "detail": detail},
              open(out / "evaluation.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
