"""Run layer 1 over a sample of authorial chunks and store the token tiers.

usage: python -m stylometry.analyze_chunks DATA_DIR [--per-book 12] [--worker k/n]

Picks up to N chunks per book, evenly spaced through the book (all chunks of
disputed works), and writes DATA_DIR/analyzed/<chunk_id>.tsv with one row per
token: raw, restored, status, lemma, pos, root, pattern, diac.
Already-analyzed chunks are skipped, so the run can be resumed.
"""
import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

COLS = ["raw", "restored", "status", "lemma", "pos", "root", "pattern", "diac"]


def sample_chunks(manifest, per_book):
    by = defaultdict(list)
    for r in manifest:
        by[r["book"]].append(r)
    out = []
    for book, rows in by.items():
        if rows[0]["status"] == "disputed" or len(rows) <= per_book:
            out += rows
        else:
            step = len(rows) / per_book
            out += [rows[int(i * step + step / 2)] for i in range(per_book)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--per-book", type=int, default=12)
    ap.add_argument("--worker", default="0/1",
                    help="k/n: process every n-th chunk starting at k (parallel runs)")
    a = ap.parse_args()
    root = Path(a.data_dir)
    manifest = list(csv.DictReader(open(root / "manifest_authorial.tsv", encoding="utf-8"),
                                   delimiter="\t"))
    todo = sample_chunks(manifest, a.per_book)
    outdir = root / "analyzed"
    outdir.mkdir(exist_ok=True)
    with open(root / "sample.tsv", "w", encoding="utf-8") as f:
        f.write("\t".join(manifest[0].keys()) + "\n")
        for r in todo:
            f.write("\t".join(r.values()) + "\n")
    todo = [r for r in todo if not (outdir / f"{r['chunk_id']}.tsv").exists()]
    # round-robin over books, so partial results already cover every author
    by = defaultdict(list)
    for r in todo:
        by[r["book"]].append(r)
    todo = [r for i in range(max(map(len, by.values()), default=0))
            for rows in by.values() if i < len(rows) for r in [rows[i]]]
    k, n = map(int, a.worker.split("/"))
    todo = todo[k::n]
    print(f"{len(todo)} chunks to analyze", flush=True)
    if not todo:
        return
    from restorer import Restorer
    R = Restorer()
    t0, words = time.time(), 0
    for i, r in enumerate(todo, 1):
        text = (root / r["path"]).read_text(encoding="utf-8")
        _, _, res = R.run(text)
        tmp = outdir / f"{r['chunk_id']}.tsv.part"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("\t".join(COLS) + "\n")
            for t in res:
                d = t.to_dict()
                f.write("\t".join(str(d[c]).replace("\t", " ") for c in COLS) + "\n")
        tmp.rename(outdir / f"{r['chunk_id']}.tsv")
        words += sum(1 for t in res if t.status != "not_word")
        if i % 10 == 0:
            rate = words / (time.time() - t0)
            print(f"{i}/{len(todo)} chunks, {rate:.0f} words/s, "
                  f"eta {(len(todo) - i) * 2000 / rate / 60:.0f} min", flush=True)


if __name__ == "__main__":
    sys.exit(main())
