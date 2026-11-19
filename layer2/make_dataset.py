"""Turn the layer 2 labels into a training set for the morphology model (2a).

Input: labels_{train,val,test}.tsv from build_labels.py.
Output (OUT_DIR):
  {train,val,test}.jsonl.gz  one segment per line:
      {"w": [bare words], "t": [ [[allowed value ids] per feature] | null ]}
      (features in the order of FEATS; value ids index labels.json)
      A word's allowed values are the union over its surviving readings, so a
      tie is "any of these is right" (partial-label training). Words without a
      label (no_match, oov) get null and are ignored by the loss, but stay in
      the input as context.
  labels.json  the value vocabulary of every feature.

The input words are bare (no diacritics): the model must learn to read
unvowelled Classical text, as it will meet it in the corpus.

usage: python layer2/make_dataset.py LABELS_DIR OUT_DIR [--max-words 100]
"""
import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

FEATS = ["pos", "prc3", "prc2", "prc1", "prc0", "per", "asp", "vox", "mod", "gen", "num", "stt", "cas", "enc0"]
LABELED = {"unique", "case_tie", "tie"}


def parse_option(opt):
    lex, pos, feats = opt.split("~", 2)
    d = {"pos": pos}
    for kv in feats.split("|"):
        k, v = kv.split(":", 1)
        if k != "pos":
            d[k] = v
    return d


def sentences(path):
    cur, sid = [], None
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if cur:
                    yield cur
                cur = []
                continue
            r = line.split("\t")
            if len(r) < 12:
                r += [""] * (12 - len(r))
            cur.append(r)
    if cur:
        yield cur


def first_pass(paths):
    vocab = defaultdict(set)
    for path in paths:
        for sent in sentences(path):
            for r in sent:
                if r[4] in LABELED and r[11]:
                    for opt in r[11].split(";"):
                        for k, v in parse_option(opt).items():
                            if k in FEATS:
                                vocab[k].add(v)
    return {k: sorted(vocab[k]) for k in FEATS}


def convert(path, out, vocab, max_words):
    idx = {k: {v: i for i, v in enumerate(vocab[k])} for k in FEATS}
    n_seg = n_w = n_lab = 0
    with gzip.open(out, "wt", encoding="utf-8") as o:
        for sent in sentences(path):
            words, targets = [], []
            for r in sent:
                bare, status, options = r[3], r[4], r[11]
                words.append(bare)
                if status in LABELED and options:
                    allowed = defaultdict(set)
                    for opt in options.split(";"):
                        for k, v in parse_option(opt).items():
                            if k in FEATS:
                                if v in idx[k]:
                                    allowed[k].add(idx[k][v])
                    targets.append([sorted(allowed[k]) for k in FEATS])
                    n_lab += 1
                else:
                    targets.append(None)
            for s in range(0, len(words), max_words):
                o.write(json.dumps({"w": words[s:s + max_words], "t": targets[s:s + max_words]},
                                   ensure_ascii=False, separators=(",", ":")) + "\n")
                n_seg += 1
            n_w += len(words)
    return n_seg, n_w, n_lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--max-words", type=int, default=100)
    ap.add_argument("--vocab", default="", help="reuse this labels.json (keeps value ids, e.g. to continue training)")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {s: Path(a.labels_dir) / f"labels_{s}.tsv" for s in ("train", "val", "test")}
    vocab = json.load(open(a.vocab, encoding="utf-8"))["values"] if a.vocab else first_pass(paths.values())
    for split, path in paths.items():
        n_seg, n_w, n_lab = convert(path, out / f"{split}.jsonl.gz", vocab, a.max_words)
        print(f"{split}: {n_seg} segments, {n_w} words, {n_lab} labelled ({n_lab / n_w:.1%})")
    json.dump({"features": FEATS, "values": vocab}, open(out / "labels.json", "w"), ensure_ascii=False, indent=1)
    print({k: len(vocab[k]) for k in FEATS})


if __name__ == "__main__":
    main()
