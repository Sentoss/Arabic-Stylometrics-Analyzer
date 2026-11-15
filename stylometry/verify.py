"""Authorship verification with the Impostors method: calibration on books of
known authorship, then the disputed works.

usage: python -m stylometry.verify DATA_DIR [--out reports/layer3]

Calibration. For every secure book T of an author A with two or more books:
  true trial   candidate A (profile from A's other books) vs. impostors =
               every other author
  false trial  candidate B != A vs. impostors = every author except A and B,
               i.e. the true author is NOT in the line-up (the realistic,
               harder case for disputed works)
The score is the share of rounds the candidate wins. From the two score
distributions we report AUC and the threshold with the best balanced accuracy.

Disputed works: each is scored against its traditional author, ranked
against all authors by Cosine Delta, and explained (features that separate
it from the candidate, and Zeta words).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from stylometry.features import load_sample
from stylometry.methods import (Families, zscore, cosine_delta, impostors_score,
                                contributions, zeta)

SPEC = [("lemma", 300), ("function", 150), ("pos", 300), ("pattern", 200)]
DISPUTED_AUTHOR = {
    "0255Jahiz.MahasinWaAddad": "0255Jahiz",
    "0255Jahiz.TajFiAkhlaq": "0255Jahiz",
    "0276IbnQutaybaDinawari.ImamaWaSiyasa": "0276IbnQutaybaDinawari",
    "0362AbuFarajIsbahani.AdabGhuraba": "0362AbuFarajIsbahani",
}


def shared_text_index(data_dir, books, n=8):
    """Hashes of all word n-grams in the clean prose of the given books."""
    import json as _json
    import re as _re
    idx = defaultdict(set)
    for b in books:
        for line in open(Path(data_dir) / "clean" / f"{b}.jsonl", encoding="utf-8"):
            p = _json.loads(line)
            if p["type"] != "prose":
                continue
            ws = _norm_words(p["text"])
            idx[b].update(hash(" ".join(ws[i:i + n])) for i in range(len(ws) - n + 1))
    return idx


def _norm_words(text):
    import re as _re
    t = _re.sub(r"[ً-ْٰـ]", "", text)
    t = _re.sub("[أإآ]", "ا", t).replace("ى", "ي").replace("ة", "ه")
    return _re.findall(r"[ء-ي]+", t)


def remove_shared(data_dir, chunk_ids, shared, n=8):
    """Recompute features of chunks with every word covered by an n-gram
    shared with another book removed. Returns (features, words kept, words total)."""
    from stylometry.features import read_tokens, chunk_features
    root = Path(data_dir)
    out, kept, total = [], 0, 0
    for cid in chunk_ids:
        rows = read_tokens(root / "analyzed" / f"{cid}.tsv")
        ws = [(_norm_words(r["raw"]) or [""])[0] for r in rows]
        drop = [False] * len(rows)
        for i in range(len(ws) - n + 1):
            if hash(" ".join(ws[i:i + n])) in shared:
                for k in range(i, i + n):
                    drop[k] = True
        keep = [r for r, d in zip(rows, drop) if not d]
        kept += len(keep)
        total += len(rows)
        out.append(chunk_features(keep))
    return out, kept, total


def profiles(Z, metas, idx, exclude_book=None):
    by = defaultdict(list)
    for i in idx:
        if metas[i]["book"] != exclude_book:
            by[metas[i]["author"]].append(i)
    return {a: Z[ii].mean(0) for a, ii in by.items()}


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def best_threshold(pos, neg):
    best = (0, 0.5)
    for t in np.unique(np.r_[pos, neg]):
        bal = ((np.asarray(pos) >= t).mean() + (np.asarray(neg) < t).mean()) / 2
        best = max(best, (bal, float(t)))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--out", default="reports/layer3")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--impostors", choices=["all", "same-genre"], default="all")
    ap.add_argument("--name", default="verification")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    metas, feats = load_sample(a.data_dir)
    fams = Families(feats)
    known = [i for i, m in enumerate(metas) if m["status"] in ("secure", "single")]
    books = defaultdict(set)
    genres = defaultdict(set)
    for i in known:
        genres[metas[i]["author"]].add(metas[i]["genre"])
        if metas[i]["status"] == "secure":
            books[metas[i]["author"]].add(metas[i]["book"])
    cands = sorted(au for au, b in books.items() if len(b) >= 2)
    rng = np.random.default_rng(0)

    def lineup(prof, exclude, genre):
        """Impostors: all other authors, or (same-genre mode) only authors who
        wrote in the questioned text's genre; at least 5, else everyone."""
        pool = {k: v for k, v in prof.items() if k not in exclude}
        if a.impostors == "same-genre":
            same = {k: v for k, v in pool.items() if genre in genres[k]}
            if len(same) >= 5:
                return same
        return pool

    # ------------------------------------------------ calibration
    pos, neg, trials = [], [], []
    for au in cands:
        for tb in sorted(books[au]):
            te = [i for i in known if metas[i]["book"] == tb]
            tr = [i for i in known if metas[i]["book"] != tb]
            X, _ = fams.select(SPEC, tr)
            Z = zscore(X, tr)
            prof = profiles(Z, metas, tr)
            g = metas[te[0]]["genre"]
            # does the candidate have another book in this genre?
            other_g = {metas[i]["genre"] for i in tr if metas[i]["author"] == au}
            s = impostors_score(Z[te], prof[au], lineup(prof, {au}, g), iters=a.iters, rng=rng)
            pos.append(s)
            trials.append({"book": tb, "candidate": au, "true": True, "score": s,
                           "same_genre": g in other_g})
            for b in [c for c in cands if c != au]:
                s = impostors_score(Z[te], prof[b], lineup(prof, {au, b}, g), iters=a.iters, rng=rng)
                neg.append(s)
                trials.append({"book": tb, "candidate": b, "true": False, "score": s,
                               "same_genre": g in genres[b]})
        print(f"calibrated {au}", flush=True)
    A = auc(pos, neg)
    bal, thr = best_threshold(pos, neg)
    print(f"AUC {A:.3f}; best threshold {thr:.2f} (balanced accuracy {bal:.1%}); "
          f"true trials {len(pos)}, false trials {len(neg)}")

    # ------------------------------------------------ disputed works
    results = []
    for db, au in DISPUTED_AUTHOR.items():
        te = [i for i, m in enumerate(metas) if m["book"] == db]
        if not te:
            continue
        tr = known
        X, names = fams.select(SPEC, tr)
        Z = zscore(X, tr)
        prof = profiles(Z, metas, tr)
        s = impostors_score(Z[te], prof[au], lineup(prof, {au}, metas[te[0]]["genre"]),
                            iters=a.iters, rng=rng)
        x = Z[te].mean(0)
        ranking = sorted(((float(cosine_delta(x[None], v[None])[0, 0]), k) for k, v in prof.items()))
        contrib = contributions(x, prof[au], names, k=15)
        cand_feats = [feats[i] for i in tr if metas[i]["author"] == au]
        z_hi, z_lo = zeta([feats[i] for i in te], cand_feats, family="lemma")
        results.append({
            "book": db, "candidate": au, "chunks": len(te), "impostors_score": s,
            "verdict": "consistent with the candidate" if s >= thr else "not supported",
            "nearest_authors": ranking[:6],
            "candidate_rank": [k for _, k in ranking].index(au) + 1,
            "largest_differences": contrib,
            "zeta_disputed_uses": z_hi, "zeta_candidate_uses": z_lo,
        })
        print(f"{db:42} chunks {len(te):3}  score {s:.2f}  rank of {au} = "
              f"{results[-1]['candidate_rank']}  nearest {ranking[0][1]}", flush=True)
    # ------------------------------------------------ disputed works, shared text removed
    # Compilations copy passages from common sources; those passages make two
    # books look alike for reasons other than the compiler's style. Remove every
    # 8-word sequence a disputed work shares with any benchmark book and
    # verify again on what remains.
    known_books = sorted({metas[i]["book"] for i in known})
    sh = shared_text_index(a.data_dir, known_books)
    union = set().union(*sh.values())
    for r in results:
        db, au = r["book"], r["candidate"]
        ids = [m["chunk_id"] for m in metas if m["book"] == db]
        own = shared_text_index(a.data_dir, [db])[db]
        overlap = {b: len(own & s) / max(len(own), 1) for b, s in sh.items()}
        r["shared_8grams_top"] = sorted(overlap.items(), key=lambda kv: -kv[1])[:5]
        clean_feats, kept, total = remove_shared(a.data_dir, ids, union)
        allf = feats + clean_feats
        allm = metas + [{"author": "Q", "book": "Q", "status": "q", "genre": metas[known[0]]["genre"]}] * len(ids)
        f2 = Families(allf)
        X, names = f2.select(SPEC, known)
        Z = zscore(X, known)
        te = list(range(len(feats), len(allf)))
        prof = profiles(Z, allm, known)
        g = next(m["genre"] for m in metas if m["book"] == db)
        s = impostors_score(Z[te], prof[au], lineup(prof, {au}, g), iters=a.iters, rng=rng)
        x = Z[te].mean(0)
        ranking = sorted((float(cosine_delta(x[None], v[None])[0, 0]), k) for k, v in prof.items())
        r["after_removing_shared_text"] = {
            "words_kept": kept, "words_total": total, "impostors_score": s,
            "candidate_rank": [k for _, k in ranking].index(au) + 1,
            "nearest_authors": ranking[:6],
            "largest_differences": contributions(x, prof[au], names, k=15),
        }
        print(f"{db:42} shared text removed: kept {kept}/{total} words; score {s:.2f}; rank of "
              f"{au} = {r['after_removing_shared_text']['candidate_rank']}; nearest {ranking[0][1]}; "
              f"top overlap {r['shared_8grams_top'][0][0]} {r['shared_8grams_top'][0][1]:.1%}", flush=True)

    same = [t for t in trials if t["same_genre"]]
    A_same = auc([t["score"] for t in same if t["true"]], [t["score"] for t in same if not t["true"]])
    print(f"same-genre trials only: AUC {A_same:.3f}")
    json.dump({"spec": SPEC, "impostors": a.impostors, "auc": A, "auc_same_genre": A_same,
               "threshold": thr, "balanced_accuracy": bal, "trials": trials, "disputed": results},
              open(out / f"{a.name}.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
