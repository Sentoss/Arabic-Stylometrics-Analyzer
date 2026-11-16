"""Turn analysed chunks into feature counts.

Each chunk becomes a dict of counters, one per feature family:
  surface   restored word forms (diacritics removed)
  lemma     lemmas (restored form when the word is unknown to the analyzer)
  function  closed-class lemmas (prepositions, particles, pronouns, conjunctions,
            ...) plus clitics read from the analysed pattern: PRC:و, PRC:ف,
            PRC:ب, PRC:ال ..., ENC:ه, ENC:هم ...
  pos       part-of-speech 1-, 2- and 3-grams
  pattern   morphological pattern (wazn) without clitics, e.g. 1َ2َّ3َ, 1ا2ِ3
  char      character 3- and 4-grams within words of the restored text
  quoting   placeholders per chunk: <Q> <H> <M> <ISNAD> and *-BLOCK

Relative frequencies (per word, or per token for n-grams) are computed by
the methods, so the raw counts here stay reusable.
"""
import csv
import re
from collections import Counter
from pathlib import Path

MARKS = re.compile(r"[ً-ْٰـ]")
CLOSED_POS = {"prep", "conj", "conj_sub", "part", "part_neg", "part_verb", "part_det",
              "part_focus", "part_fut", "part_interrog", "part_restrict", "part_voc",
              "part_emphatic", "pron", "pron_dem", "pron_rel", "pron_interrog",
              "pron_exclam", "verb_pseudo", "adv_rel", "adv_interrog", "abbrev"}
PLACEHOLDERS = re.compile(r"<(Q|H|M|ISNAD|Q-BLOCK|H-BLOCK|M-BLOCK)>")
AR_WORD = re.compile(r"[ء-ي]+")


def bare(s):
    return MARKS.sub("", s)


def clitics(pattern):
    """'وَال1ا2ِ3ِ' -> ['PRC:و', 'PRC:ال']; '1َيْ3َهُ' -> ['ENC:ه']."""
    if not pattern or not re.search(r"\d", pattern):
        return []
    first = re.search(r"\d", pattern).start()
    last = max(m.start() for m in re.finditer(r"\d", pattern))
    pre, post = bare(pattern[:first]), bare(pattern[last + 1:])
    out = []
    # proclitics in order: conjunction, preposition/future, article
    for c in ("و", "ف"):
        if pre.startswith(c):
            out.append("PRC:" + c)
            pre = pre[1:]
    for c in ("ب", "ل", "ك", "س"):
        if pre.startswith(c) and not pre.startswith("ال"):
            out.append("PRC:" + c)
            pre = pre[1:]
    if pre.startswith("ال") or pre.startswith("ل") and out and out[-1] == "PRC:ل":
        out.append("PRC:ال")
    # enclitic pronouns. Final ي is left out: in patterns it is usually the
    # nisba ending (ال1َ2ْ3ِيّ) or a weak root letter, not the 1sg suffix.
    for e in ("هما", "هم", "هن", "كما", "كم", "كن", "نا", "ها", "ه", "ك"):
        if post.endswith(e):
            out.append("ENC:" + e)
            break
    return out


def core_pattern(pattern):
    """Strip clitics: keep the stretch from the first to the last root slot."""
    if not pattern or not re.search(r"\d", pattern):
        return ""
    first = re.search(r"\d", pattern).start()
    last = max(m.start() for m in re.finditer(r"\d", pattern))
    core = pattern[first:last + 1]
    # keep a following long vowel that belongs to the stem (1ا2ِي, 1َ2ا), but
    # not the case vowel: case endings come from syntax and are model guesses
    tail = re.match(r"ّ?(?:[َ-ِ]?[اىوي])?", pattern[last + 1:])
    return core + (tail.group(0) if tail else "")


def read_tokens(path):
    with open(path, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f, delimiter="\t") if r["status"] != "not_word"]


def chunk_features(tokens, chunk_text=""):
    f = {k: Counter() for k in ("surface", "lemma", "function", "pos", "pattern", "char", "quoting")}
    pos_seq = []
    for t in tokens:
        surf = bare(t["restored"])
        if not AR_WORD.fullmatch(surf or ""):
            continue
        f["surface"][surf] += 1
        lem = bare(t["lemma"]) if t["lemma"] else surf
        f["lemma"][lem] += 1
        pos = t["pos"] or "UNK"
        pos_seq.append(pos)
        if pos in CLOSED_POS:
            f["function"]["W:" + lem] += 1
        for c in clitics(t["pattern"]):
            f["function"][c] += 1
        cp = core_pattern(t["pattern"])
        if cp and pos not in CLOSED_POS:
            f["pattern"][pos[:4] + ":" + cp] += 1
        padded = f" {surf} "
        for n in (3, 4):
            for i in range(len(padded) - n + 1):
                f["char"][padded[i:i + n]] += 1
    for n in (1, 2, 3):
        for i in range(len(pos_seq) - n + 1):
            f["pos"][" ".join(pos_seq[i:i + n])] += 1
    for m in PLACEHOLDERS.finditer(chunk_text):
        f["quoting"][m.group(1)] += 1
    f["quoting"]["_words"] = sum(f["surface"].values())
    return f


def load_sample(data_dir, analyzed="analyzed", manifest="sample.tsv"):
    """Return (meta rows, feature dicts) for every analysed chunk."""
    root = Path(data_dir)
    rows = list(csv.DictReader(open(root / manifest, encoding="utf-8"), delimiter="\t"))
    metas, feats = [], []
    for r in rows:
        p = root / analyzed / f"{r['chunk_id']}.tsv"
        if not p.exists():
            continue
        text = (root / r["path"]).read_text(encoding="utf-8") if r.get("path") else ""
        metas.append(r)
        feats.append(chunk_features(read_tokens(p), text))
    return metas, feats
