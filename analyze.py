"""Analyse a text from outside the corpus.

usage:
  python analyze.py MYTEXT.txt --data DATA_DIR --refs REFS_DIR
                    [--candidate 0255Jahiz] [--out out/mytext]

Steps
  1. read plain text (paragraphs separated by line breaks) or an OpenITI
     mARkdown file; verse lines are set aside
  2. remove editorial insertions (corpus/editorial.py)
  3. find quotations and isnads, keep the author's own words (corpus/quotations.py)
  4. layer 1: spelling restoration and morphological analysis, per ~2,000-word chunk
  5. compare with the benchmark authors:
       - ranking of authors by Cosine Delta to the text's profile
       - with --candidate: Impostors score against that author, the calibrated
         threshold from reports/layer3/verification.json, and the features
         that most separate the text from the candidate (plus Zeta words)

  6. the text's own feature profile: every feature's rate against the corpus
     (features.tsv), and the most over- and under-used ones in report.md

Outputs (in --out): authorial.txt, chunks/, tokens/, features.tsv, report.md
Candidate names are OpenITI author IDs as listed in corpus/books.tsv
(e.g. 0255Jahiz, 0276IbnQutaybaDinawari).
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from corpus.editorial import clean_editorial
from corpus.openiti import parse_markdown, is_verse
from corpus.build import chunk_paragraphs


def read_text(path):
    raw = Path(path).read_text(encoding="utf-8")
    if "#META#Header#End#" in raw:
        return parse_markdown(raw)
    paras = []
    for line in raw.splitlines():
        line = line.strip()
        if line:
            paras.append({"type": "poetry" if is_verse(line) else "prose", "page": "", "text": line})
    return paras


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--data", default="/home/claude/data_ext/corpus",
                    help="benchmark data dir (with analyzed/ and sample.tsv)")
    ap.add_argument("--refs", default="/home/claude/data_ext/refs",
                    help="reference texts dir (Qur'an, hadith, proverbs)")
    ap.add_argument("--candidate", help="OpenITI author ID to verify against")
    ap.add_argument("--out", default=None)
    ap.add_argument("--chunk", type=int, default=2000)
    a = ap.parse_args()
    out = Path(a.out or f"out/{Path(a.text).stem}")
    (out / "chunks").mkdir(parents=True, exist_ok=True)
    (out / "tokens").mkdir(exist_ok=True)

    # 1-2. read and clean
    paras = read_text(a.text)
    ed = Counter()
    for p in paras:
        if p["type"] != "heading":
            p["text"], _ = clean_editorial(p["text"], ed)

    # 3. quotations
    from corpus import quotations as q
    refs = Path(a.refs)
    print("indexing references (about a minute) ...", flush=True)
    quran, hadith, (proverbs, _) = q.build_quran(refs), q.build_hadith(refs), q.build_proverbs(refs)
    qs = Counter()
    for p in paras:
        if p["type"] == "prose":
            p["text"] = q.process_paragraph(p["text"], quran, hadith, proverbs, lambda h: True, qs)
    (out / "authorial.txt").write_text("\n".join(p["text"] for p in paras if p["type"] == "prose"),
                                       encoding="utf-8")

    # 4. layer 1 per chunk
    from restorer import Restorer
    from stylometry.features import chunk_features
    chunks = list(chunk_paragraphs(paras, a.chunk))
    if not chunks:   # short text: one chunk of whatever there is
        text = "\n".join(p["text"] for p in paras if p["type"] == "prose")
        chunks = [("", "", len(text.split()), text)]
    R = Restorer()
    feats = []
    for i, (_, _, words, text) in enumerate(chunks):
        (out / "chunks" / f"{i:04d}.txt").write_text(text, encoding="utf-8")
        _, _, res = R.run(text)
        rows = [{k: str(v) for k, v in r.to_dict().items()} for r in res]
        with open(out / "tokens" / f"{i:04d}.tsv", "w", encoding="utf-8") as f:
            cols = ["raw", "restored", "status", "lemma", "pos", "root", "pattern", "diac"]
            f.write("\t".join(cols) + "\n")
            for r in rows:
                f.write("\t".join(r[c].replace("\t", " ") for c in cols) + "\n")
        feats.append(chunk_features([r for r in rows if r["status"] != "not_word"], text))
        print(f"analysed chunk {i + 1}/{len(chunks)}", flush=True)

    # 5. compare with the benchmark
    from stylometry.features import load_sample
    from stylometry.methods import Families, zscore, cosine_delta, impostors_score, contributions, zeta
    from stylometry.verify import SPEC, profiles
    metas, bfeats = load_sample(a.data)
    known = [i for i, m in enumerate(metas) if m["status"] in ("secure", "single")]
    allf = bfeats + feats
    allm = metas + [{"author": "QUESTIONED", "book": "QUESTIONED", "status": "questioned"}] * len(feats)
    fams = Families(allf)
    X, names = fams.select(SPEC, known)
    Z = zscore(X, known)
    te = list(range(len(bfeats), len(allf)))
    prof = profiles(Z, allm, known)
    x = Z[te].mean(0)
    ranking = sorted((float(cosine_delta(x[None], v[None])[0, 0]), k) for k, v in prof.items())

    words_total = qs["words_total"]
    lines = [f"# Analysis of `{Path(a.text).name}`", "",
             f"- prose words: {words_total:,}; the author's own after removing quotations "
             f"and isnads: {qs['words_author']:,} ({qs['words_author'] / max(words_total, 1):.1%})",
             f"- chunks analysed: {len(chunks)} (verdicts on fewer than 3 chunks are weak)",
             f"- editorial items removed: {sum(v for k, v in ed.items() if 'hint' not in k and k != 'restored')}",
             f"- quotations: isnad {qs['I_isnad_words']} words; Qur'an {qs['Q_block_words']} reproduced / "
             f"{qs['Q_embedded_n']} embedded; hadith {qs['H_block_words']} / {qs['H_embedded_n']}; "
             f"proverbs {qs['M_block_words']} / {qs['M_embedded_n']}", "",
             "## Closest benchmark authors (Cosine Delta, lower = closer)", "",
             "| rank | author | distance |", "|---|---|---|"]
    lines += [f"| {r} | {k} | {d:.3f} |" for r, (d, k) in enumerate(ranking[:8], 1)]

    # the text's own feature profile, against the whole benchmark corpus
    from stylometry.profile import FAMILIES, LABEL, feature_matrix, profile_group, top_lines
    with open(out / "features.tsv", "w", encoding="utf-8") as f:
        f.write("family\tfeature\traw\tper_1000_text\tper_1000_corpus\tz\tused_in_chunks\n")
        prof_lines = ["", "## Most distinctive features of this text", "",
                      "Compared with all benchmark chunks. z = how far the text's average is "
                      "from the corpus average, in units of the usual chunk-to-chunk spread "
                      "(|z| above 0.5 is a clear tendency, above 1 strong). Full table: features.tsv.", ""]
        for fam, n in FAMILIES.items():
            Xf, nf = feature_matrix(fams, fam, n, known)
            rows = profile_group(Xf, nf, fam, te, known)
            for r in sorted(rows, key=lambda r: -r["z"]):
                f.write(f"{fam}\t{r['feature']}\t{r['raw']}\t{r['rate']:.3f}\t"
                        f"{r['rest_rate']:.3f}\t{r['z']:.3f}\t{r['used_in']:.3f}\n")
            prof_lines += top_lines(rows, 8, LABEL[fam])
    lines += prof_lines

    if a.candidate:
        if a.candidate not in prof:
            raise SystemExit(f"unknown candidate {a.candidate}; see corpus/books.tsv")
        pool = {k: v for k, v in prof.items() if k != a.candidate}
        s = impostors_score(Z[te], prof[a.candidate], pool, iters=200)
        cal = Path("reports/layer3/verification.json")
        thr = json.load(open(cal))["threshold"] if cal.exists() else None
        verdict = ("" if thr is None else
                   f" (calibrated threshold {thr:.2f}: "
                   f"{'consistent with' if s >= thr else 'not supported for'} {a.candidate})")
        lines += ["", f"## Verification against {a.candidate}", "",
                  f"Impostors score: **{s:.2f}**{verdict}", "",
                  "Features that most separate the text from the candidate "
                  "(z-scores; positive = used more than the corpus average):", "",
                  "| feature | this text | candidate |", "|---|---|---|"]
        lines += [f"| {n} | {zx:+.2f} | {zc:+.2f} |" for n, zx, zc in contributions(x, prof[a.candidate], names)]
        cand = [bfeats[i] for i in known if metas[i]["author"] == a.candidate]
        hi, lo = zeta(feats, cand, family="lemma")
        lines += ["", "Zeta (share of samples using a lemma): in this text but not the candidate: "
                  + "، ".join(w for _, w in hi[:12]),
                  "", "the candidate's habits absent here: " + "، ".join(w for _, w in lo[:12])]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
