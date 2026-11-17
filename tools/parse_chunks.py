"""Parse corpus chunks with CamelParser (CATiB dependencies).

usage: python parse_chunks.py CORPUS_DIR SAMPLE.tsv OUT_DIR [--worker k/n] [--max-chunks N]

Each chunk is cut into segments of at most 40 words: the chunk's own lines
first (the authorial chunks carry no punctuation), then, inside long lines,
before a connective (فـ، ثم، و) where possible. One .conllx file per chunk;
finished chunks are skipped, so the run can be resumed.
"""
import argparse
import csv
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/claude/arabic_stylometrics")

from camel_tools.utils.charmap import CharMapper
from pandas import read_csv

from src.conll_output import text_tuples_to_string
from src.data_preparation import get_file_type_params, parse_text
from src.initialize_disambiguator.disambiguator_interface import get_disambiguator

ROOT = Path(__file__).parent
CONN = re.compile(r"^(ف|ثم$|و)")


def segment(text, maxw=40):
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^ء-يٱ\s]", " ", text)
    out = []
    for line in text.split("\n"):
        w = line.split()
        while len(w) > maxw:
            cut = max((i for i in range(maxw // 2, maxw) if CONN.match(w[i]) and len(w[i]) > 2),
                      default=maxw)
            out.append(" ".join(w[:cut]))
            w = w[cut:]
        if len(w) >= 2:
            out.append(" ".join(w))
    return out


class PerSentence:
    """CAMeL's batch call fails on mixed lengths; disambiguate one by one."""

    def __init__(self, base):
        self.base = base

    def disambiguate_sentences(self, token_lines):
        return [self.base.disambiguate(s) for s in token_lines]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("sample")
    ap.add_argument("out")
    ap.add_argument("--worker", default="0/1")
    ap.add_argument("--max-chunks", type=int, default=0)
    a = ap.parse_args()
    root, out = Path(a.corpus), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(a.sample, encoding="utf-8"), delimiter="\t"))
    k, n = map(int, a.worker.split("/"))
    todo = [r for r in rows[k::n] if not (out / f"{r['chunk_id']}.conllx").exists()]
    print(f"{len(todo)} chunks to parse", flush=True)
    if a.max_chunks:
        todo = todo[:a.max_chunks]
    if not todo:
        return
    base = get_disambiguator("bert", "r13")
    from layer2.supplement import ClassicalAnalyzer
    base._analyzer = ClassicalAnalyzer(analyzer=base._analyzer, keep_backoff=True, mode="text")
    dis = PerSentence(base)
    clitic = read_csv(ROOT / "data/clitic_feats.csv").astype(str).astype(object)
    arclean = CharMapper.builtin_mapper("arclean")
    t0, words = time.time(), 0
    for i, r in enumerate(todo, 1):
        lines = segment((root / r["path"] if not Path(r["path"]).is_absolute() else Path(r["path"])).read_text(encoding="utf-8"))
        params = get_file_type_params(lines, "text", None, ROOT / "models/CAMeLBERT-CATiB-biaffine.model",
                                      arclean, dis, clitic, "catib6", "r13")
        parsed = parse_text("text", params)
        tmp = out / f"{r['chunk_id']}.conllx.part{k}"
        tmp.write_text("\n".join(text_tuples_to_string(parsed, "text", sentences=lines)) + "\n",
                       encoding="utf-8")
        tmp.rename(out / f"{r['chunk_id']}.conllx")
        words += sum(len(l.split()) for l in lines)
        print(f"{i}/{len(todo)} chunks, {words / (time.time() - t0):.0f} words/s", flush=True)


if __name__ == "__main__":
    main()
