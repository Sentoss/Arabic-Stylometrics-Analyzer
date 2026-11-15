"""Build the benchmark corpus.

usage: python -m corpus.build DATA_DIR [--chunk 2000]

DATA_DIR/raw/<versionUri>.mARkdown      downloaded files
DATA_DIR/clean/<book>.jsonl             labelled paragraphs
DATA_DIR/chunks/<book>/<n>.txt          prose chunks of ~N Arabic words
DATA_DIR/manifest.tsv                   one row per chunk
DATA_DIR/books.tsv                      one row per book with stats

Chunks contain prose paragraphs only (poetry and headings excluded) and
never cross a book boundary. They are cut at paragraph ends, so sizes
vary slightly around N. A trailing remainder under N/2 is dropped.
"""
import argparse
import json
import re
from pathlib import Path

from corpus.benchmark import BOOKS
from corpus.openiti import load_metadata, primary_version, download, parse_markdown

AR_WORD = re.compile(r"[ء-ي]+")


SENT_END = re.compile(r"(?<=[.؟!؛:])\s+")


def split_long(paras, limit):
    """Some files have few paragraph breaks (al-Basa'ir is nearly one
    paragraph). Split any prose paragraph over `limit` words at sentence
    punctuation, or at the word limit if there is none."""
    for p in paras:
        words = p["text"].split()
        if p["type"] != "prose" or len(words) <= limit:
            yield p
            continue
        piece = []
        for sent in SENT_END.split(p["text"]):
            piece.append(sent)
            if len(" ".join(piece).split()) >= limit:
                yield {**p, "text": " ".join(piece)}
                piece = []
        rest = " ".join(piece).split()
        for i in range(0, len(rest), limit):
            yield {**p, "text": " ".join(rest[i:i + limit])}


def chunk_paragraphs(paras, size):
    chunk, n, first_page = [], 0, None
    for p in split_long(paras, size // 4):
        if p["type"] != "prose":
            continue
        w = len(AR_WORD.findall(p["text"]))
        if not w:
            continue
        if first_page is None:
            first_page = p["page"]
        chunk.append(p["text"])
        n += w
        if n >= size:
            yield first_page, p["page"], n, "\n".join(chunk)
            chunk, n, first_page = [], 0, None
    if n >= size // 2:
        yield first_page, p["page"], n, "\n".join(chunk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--chunk", type=int, default=2000)
    a = ap.parse_args()
    root = Path(a.data_dir)
    for d in ("raw", "clean", "chunks"):
        (root / d).mkdir(parents=True, exist_ok=True)

    meta = load_metadata()
    books_out = ["book\tauthor\tdeath_ah\tgenre\tstatus\tversion\twords_prose\twords_poetry"
                 "\tverse_marked\tchunks\turl"]
    manifest = ["chunk_id\tbook\tauthor\tdeath_ah\tgenre\tstatus\tpage_from\tpage_to\twords\tpath"]
    for book, (genre, status) in BOOKS.items():
        r = primary_version(meta, book)
        if r is None:
            print("MISSING", book)
            continue
        raw_path = root / "raw" / (r["versionUri"] + ".mARkdown")
        try:
            download(r["url"], raw_path)
        except Exception as e:
            print("FAILED", book, e)
            continue
        paras = parse_markdown(raw_path.read_text(encoding="utf-8"))
        with open(root / "clean" / f"{book}.jsonl", "w", encoding="utf-8") as f:
            for p in paras:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        author = book.split(".")[0]
        death = int(author[:4])
        wp = sum(len(AR_WORD.findall(p["text"])) for p in paras if p["type"] == "prose")
        wv = sum(len(AR_WORD.findall(p["text"])) for p in paras if p["type"] == "poetry")
        cdir = root / "chunks" / book
        cdir.mkdir(exist_ok=True)
        n = 0
        for i, (p0, p1, words, text) in enumerate(chunk_paragraphs(paras, a.chunk)):
            path = cdir / f"{i:04d}.txt"
            path.write_text(text, encoding="utf-8")
            manifest.append(f"{book}.{i:04d}\t{book}\t{author}\t{death}\t{genre}\t{status}"
                            f"\t{p0}\t{p1}\t{words}\t{path.relative_to(root)}")
            n += 1
        # Under 1% poetry in an adab/history/criticism book almost always means
        # the edition's verses are unmarked and sit inside the prose chunks.
        marked = "yes" if wv >= 0.01 * (wp + wv) or genre in ("law", "ethics", "zuhd") else "no"
        books_out.append(f"{book}\t{author}\t{death}\t{genre}\t{status}\t{r['versionUri']}"
                         f"\t{wp}\t{wv}\t{marked}\t{n}\t{r['url']}")
        print(f"{book:50} prose {wp:>8,}  poetry {wv:>7,}  chunks {n:>4}")
    (root / "books.tsv").write_text("\n".join(books_out) + "\n", encoding="utf-8")
    (root / "manifest.tsv").write_text("\n".join(manifest) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
