"""Feature profiles: which features characterise an author, a genre, a book?

For each group, every feature's rate in the group is compared with its rate in
the rest of the corpus:

  rate      occurrences per 1,000 (words, clitics, POS n-grams or patterns of
            that family) in the group, and in the rest
  z         how far the group's average sits from the corpus average, in
            units of the usual chunk-to-chunk spread (+ = used more)
  used_in   share of the group's chunks that contain the feature at all
  steady    for groups with several books: share of the books that lean the
            same way as the group (1.0 = every book; a habit, not one book's
            topic)

usage:
  python -m stylometry.profile DATA_DIR --by author|genre|book
         [--sample sample40.tsv] [--only 0255Jahiz] [--top 10] [--out reports/profiles]

Writes profiles_<by>.tsv (every group x feature) and profiles_<by>.md (the most
over- and under-used features of each group, per family).
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from .features import load_sample
from .methods import Families

FAMILIES = {"function": 150, "lemma": 300, "pos": 300, "pattern": 200}
LABEL = {"function": "Function words and clitics", "lemma": "Lemmas",
         "pos": "Part-of-speech sequences", "pattern": "Patterns (awzan)"}
ROOT_SLOTS = {"1": "ف", "2": "ع", "3": "ل", "4": "ل"}


def readable(fam, name):
    """Show features the way a grammarian would write them."""
    if fam == "function":
        if name.startswith("W:"):
            return name[2:]
        if name.startswith("PRC:"):
            return name[4:] + "ـ (proclitic)"
        if name.startswith("ENC:"):
            return "ـ" + name[4:] + " (suffix)"
    if fam == "pattern" and ":" in name:
        pos, pat = name.split(":", 1)
        return f"{re.sub('[1-4]', lambda m: ROOT_SLOTS[m.group()], pat)} ({pos})"
    if fam == "pos":
        return name.replace(" ", " + ")
    return name


def feature_matrix(fams, fam, top, ref):
    M = fams.mats[fam]
    mean = np.asarray(M[ref].mean(0)).ravel()
    cols = np.argsort(-mean)[:top]
    X = M[:, cols].toarray()
    return X, [fams.names[fam][j] for j in cols]


def profile_group(X, names, fam, grp, rest, books=None):
    """Rows of the table for one group and one family."""
    mu, sd = X[rest].mean(0), X[rest].std(0)
    sd[sd == 0] = 1
    Z = (X - mu) / sd
    zg = Z[grp].mean(0)
    used = (X[grp] > 0).mean(0)
    steady = np.full(len(names), np.nan)
    if books:
        by = defaultdict(list)
        for i, b in zip(grp, books):
            by[b].append(i)
        if len(by) >= 2:
            bm = np.vstack([Z[ii].mean(0) for ii in by.values()])
            steady = (np.sign(bm) == np.sign(zg)).mean(0)
    return [{"family": fam, "feature": readable(fam, n), "raw": n,
             "rate": 1000 * X[grp, j].mean(), "rest_rate": 1000 * X[rest, j].mean(),
             "z": float(zg[j]), "used_in": float(used[j]), "steady": float(steady[j])}
            for j, n in enumerate(names)]


def top_lines(rows, k, title):
    """Markdown: the k most over- and under-used features of one family."""
    def row(r):
        st = "" if np.isnan(r["steady"]) else f" | {r['steady']:.0%}"
        return (f"| {r['feature']} | {r['z']:+.2f} | {r['rate']:.2f} | {r['rest_rate']:.2f} "
                f"| {r['used_in']:.0%}{st} |")
    has_steady = not all(np.isnan(r["steady"]) for r in rows)
    head = "| feature | z | per 1,000 here | per 1,000 elsewhere | in % of chunks" \
           + (" | books agreeing |" if has_steady else " |")
    sep = "|---|---|---|---|---" + ("|---|" if has_steady else "|")
    srt = sorted(rows, key=lambda r: -r["z"])
    out = [f"**{title}: used more**", "", head, sep] + [row(r) for r in srt[:k]]
    out += ["", f"**{title}: used less**", "", head, sep] + [row(r) for r in srt[::-1][:k]]
    return out + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--by", choices=["author", "genre", "book"], default="author")
    ap.add_argument("--sample", default="sample40.tsv")
    ap.add_argument("--only", help="profile just this group")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out", default="reports/profiles")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    metas, feats = load_sample(a.data_dir, manifest=a.sample)
    keep = [i for i, m in enumerate(metas)
            if m["status"] in ("secure", "single") or a.by == "book"]
    fams = Families([feats[i] for i in keep], families=list(FAMILIES))
    metas = [metas[i] for i in keep]
    groups = defaultdict(list)
    for i, m in enumerate(metas):
        groups[m[a.by]].append(i)
    if a.only:
        groups = {a.only: groups[a.only]}
    allidx = list(range(len(metas)))
    mats = {fam: feature_matrix(fams, fam, n, allidx) for fam, n in FAMILIES.items()}
    tsv = open(out / f"profiles_{a.by}.tsv", "w", encoding="utf-8")
    tsv.write("group\tn_chunks\tn_books\tfamily\tfeature\traw\trate\trest_rate\tz\tused_in\tsteady\n")
    md = [f"# Feature profiles by {a.by}", "",
          f"Sample: {a.sample}. Each group is compared with all other chunks. "
          "z = how far the group's average is from the rest, in units of the usual "
          "chunk-to-chunk spread (|z| above 0.5 is a clear tendency, above 1 strong). "
          "'Books agreeing' = share of the group's books that lean the same way.", ""]
    for g in sorted(groups):
        grp = groups[g]
        rest = [i for i in allidx if i not in set(grp)]
        books = [metas[i]["book"] for i in grp]
        nb = len(set(books))
        md += [f"## {g}", "", f"{len(grp)} chunks from {nb} book(s)"
               + (f": {', '.join(sorted(set(books)))}" if nb <= 8 else ""), ""]
        for fam, (X, names) in mats.items():
            rows = profile_group(X, names, fam, grp, rest, books if a.by != "book" else None)
            for r in rows:
                tsv.write("\t".join(map(str, [g, len(grp), nb, fam, r["feature"], r["raw"],
                                              f"{r['rate']:.3f}", f"{r['rest_rate']:.3f}",
                                              f"{r['z']:.3f}", f"{r['used_in']:.3f}",
                                              "" if np.isnan(r["steady"]) else f"{r['steady']:.2f}"])) + "\n")
            md += top_lines(rows, a.top, LABEL[fam])
    tsv.close()
    (out / f"profiles_{a.by}.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out}/profiles_{a.by}.md and .tsv ({len(groups)} groups)")


if __name__ == "__main__":
    main()
