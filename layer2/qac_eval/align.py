"""Align QAC words (Uthmani, by QAC's word numbering) with the Wikisource
imla'i text, verse by verse, and attach gold features to each aligned word.

usage: python align.py QAC_MORPH.txt verses.json OUT.jsonl
Output: one record per verse: {"loc": "s:v", "words": [bare imla'i tokens],
        "gold": [features or null per token], "align": [[qac word numbers]...]}
"""
import json
import re
import sys
from collections import defaultdict

BW = {"'": "ء", ">": "أ", "<": "إ", "&": "ؤ", "}": "ئ", "{": "ٱ", "A": "ا", "b": "ب",
      "p": "ة", "t": "ت", "v": "ث", "j": "ج", "H": "ح", "x": "خ", "d": "د", "*": "ذ",
      "r": "ر", "z": "ز", "s": "س", "$": "ش", "S": "ص", "D": "ض", "T": "ط", "Z": "ظ",
      "E": "ع", "g": "غ", "f": "ف", "q": "ق", "k": "ك", "l": "ل", "m": "م", "n": "ن",
      "h": "ه", "w": "و", "Y": "ى", "y": "ي"}
DIAC = re.compile("[ً-ٰٟۖ-ۭـ]")


def skel(ar):
    """Consonant skeleton for matching Uthmani against imla'i spelling."""
    ar = DIAC.sub("", ar)
    out = []
    for ch in ar:
        if ch in "اأإآٱءٰ":
            continue
        out.append({"ى": "ي", "ة": "ت", "ؤ": "و", "ئ": "ي"}.get(ch, ch))
    return "".join(out)


def bw2ar(s):
    return "".join(BW.get(c, "") for c in s)


def lev(a, b):
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cost(a, b):
    return lev(a, b) / max(len(a), len(b), 1)


OPS = [(1, 1, 0.0), (1, 2, 0.1), (1, 3, 0.1), (2, 1, 0.1), (1, 0, 1.0), (0, 1, 1.0)]


def align(q, w):
    """q, w: lists of skeletons. Returns list of (q_idx_list, w_idx_list, cost)."""
    n, m = len(q), len(w)
    INF = float("inf")
    best = [[INF] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    best[0][0] = 0
    for i in range(n + 1):
        for j in range(m + 1):
            if best[i][j] == INF:
                continue
            for a, b, pen in OPS:
                if i + a > n or j + b > m:
                    continue
                c = pen if (a == 0 or b == 0) else cost("".join(q[i:i + a]), "".join(w[j:j + b])) + pen
                if best[i][j] + c < best[i + a][j + b]:
                    best[i + a][j + b] = best[i][j] + c
                    back[i + a][j + b] = (i, j, c)
    path, i, j = [], n, m
    while (i, j) != (0, 0):
        pi, pj, c = back[i][j]
        path.append((list(range(pi, i)), list(range(pj, j)), c))
        i, j = pi, pj
    return path[::-1]


# ---------------------------------------------------------------- gold features
PGN = re.compile(r"^([123])?([MF])?([SDP])$")


def pgn(tok):
    m = PGN.match(tok)
    return m.groups() if m else None


def gold(segs):
    """CAMeL-style gold features from the QAC segments of one word."""
    stem = next((s for s in segs if s["kind"] == "STEM"), None)
    if stem is None:
        return None
    f = stem["feats"]
    tag = stem["tag"]
    g = {"qac_tag": tag}
    if tag == "V":
        g["cls"] = "verb"
    elif tag in ("N", "ADJ", "PN"):
        g["cls"] = "nominal"
    elif tag in ("PRON", "DEM", "REL"):
        g["cls"] = "pronominal"
    elif tag in ("T", "LOC"):
        g["cls"] = None            # adverbs: schemes differ, not scored
    else:
        g["cls"] = "particle"
    g["prop"] = tag == "PN" if g["cls"] == "nominal" else None
    if tag == "V":
        g["asp"] = {"PERF": "p", "IMPF": "i", "IMPV": "c"}.get(next((x for x in f if x in ("PERF", "IMPF", "IMPV")), ""))
        g["vox"] = "p" if "PASS" in f else "a"
        if g["asp"] == "i":
            mood = next((x[5:] for x in f if x.startswith("MOOD:")), "IND")
            g["mod"] = {"IND": "i", "SUBJ": "s", "JUS": "j"}[mood]
        p = next((pgn(x) for x in f if pgn(x)), None)
        if p:
            g["per"], gen, num = p
            g["num"] = num.lower()
            if gen:
                g["gen"] = gen.lower()
    if g["cls"] == "nominal":
        cas = next((x for x in f if x in ("NOM", "ACC", "GEN")), None)
        if cas:
            g["cas"] = {"NOM": "n", "ACC": "a", "GEN": "g"}[cas]
    # proclitics
    pre = [s["feats"][1] for s in segs if s["kind"] == "PREFIX"]
    g["conj"] = any(p.split(":")[0] in ("w", "f") and not p.startswith("w:P") for p in pre)
    g["det"] = "Al+" in pre
    g["prep_clitic"] = any(p in ("bi+", "l:P+", "ka+", "w:P+", "ta+") for p in pre)
    g["q"] = any(p.startswith("A:") for p in pre)
    # object / possessive pronoun suffix (verb subject suffixes skipped)
    sufs = [s["feats"][1][5:] for s in segs if s["kind"] == "SUFFIX" and s["feats"][1].startswith("PRON:")]
    if tag == "V" and sufs:
        vp = "".join(x or "" for x in pgn(next((x for x in f if pgn(x)), "3MS")) or ())
        no_subj_suffix = vp in ("3MS", "3FS") or (g.get("asp") == "i" and vp in ("1S", "1P", "2MS")) \
            or (g.get("asp") == "c" and vp == "2MS")
        if not no_subj_suffix:
            sufs = sufs[1:]
    g["enc"] = sufs[-1].lower() if sufs else "0"
    # بِهِ، لَهُمْ: QAC makes the pronoun the stem and the preposition a prefix;
    # CAMeL makes the preposition the stem and the pronoun an enclitic.
    if tag == "PRON" and g["prep_clitic"]:
        p = next((x for x in f if pgn(x)), None)
        g.update(cls="particle", prep_clitic=False, enc=(p or "0").lower(), qac_tag="P+PRON")
    return g


def load_qac(path):
    words = defaultdict(list)
    for line in open(path, encoding="utf-8"):
        m = re.match(r"\((\d+):(\d+):(\d+):(\d+)\)\t(\S*)\t(\S+)\t(\S+)", line)
        if not m:
            continue
        feats = m.group(7).split("|")
        words[tuple(int(x) for x in m.group(1, 2, 3))].append(
            {"form": m.group(5), "tag": m.group(6), "kind": feats[0], "feats": feats})
    verses = defaultdict(dict)
    for (s, v, w), segs in words.items():
        verses[(s, v)][w] = segs
    return verses


def main(qac_path, verses_path, out_path):
    ws = json.load(open(verses_path, encoding="utf-8"))
    qac = load_qac(qac_path)
    stats = defaultdict(int)
    with open(out_path, "w", encoding="utf-8") as out:
        for (s, v), qw in sorted(qac.items()):
            text = ws[f"{s}:{v}"]
            toks = text.split()
            if v == 1 and s not in (1, 9) and toks[:4] and skel(" ".join(toks[:4])) == skel("بسم الله الرحمن الرحيم"):
                toks = toks[4:]
            nums = sorted(qw)
            qsk = [skel(bw2ar("".join(x["form"] for x in qw[n]))) for n in nums]
            wsk = [skel(t) for t in toks]
            path = align(qsk, wsk)
            gold_tok = [None] * len(toks)
            amap = [[] for _ in toks]
            for qi, wj, c in path:
                for j in wj:
                    amap[j] = [nums[i] for i in qi]
                kind = f"{len(qi)}:{len(wj)}"
                stats[kind] += 1
                if len(qi) == 1 and len(wj) == 1:
                    if c <= 0.5:
                        gold_tok[wj[0]] = gold(qw[nums[qi[0]]])
                        stats["scored"] += 1
                    else:
                        stats["1:1_poor_match"] += 1
            out.write(json.dumps({"loc": f"{s}:{v}", "words": [DIAC.sub("", t) for t in toks],
                                  "diac": toks, "gold": gold_tok, "align": amap}, ensure_ascii=False) + "\n")
    print(dict(stats))


if __name__ == "__main__":
    main(*sys.argv[1:4])
