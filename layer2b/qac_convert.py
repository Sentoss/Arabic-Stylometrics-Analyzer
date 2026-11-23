"""Convert the Quranic Arabic Corpus treebank (Kais Dukes, University of Leeds;
GPL with terms of use: credit corpus.quran.com) into per-word naḥw records.

Input: `syntax.txt` and `morphology.txt` from github.com/kaisdukes/quranic-corpus-api
(src/main/resources/data), i.e. the verified treebank of sūras 1–9 and 59–114.
The data are used to train models and are not redistributed here.

syntax.txt holds one graph per verse or verse part:

    -- words
    n1 = reference(2:15:2)          a word of another verse, repeated as a head
    n2, n3, n4 = word(2:15:4)        one node per segment: و + يمد + هم
    n5 = PRON(هُوَ)                  an omitted word (ضمير مستتر, فعل محذوف…)
    -- phrases
    n11 = PP(n6 - n7)                a phrase or clause over nodes n6..n7
    -- edges
    conj(n3 - n1)                    relation(dependent - head)

Output (JSONL, one record per word):
    loc, text, segments [{tag, form, node}], edges of each segment
    function     the naḥw function of the word's stem, in QAC's Arabic terms
    head         {"word": "c:v:w"} | {"omitted": TAG, "text": …} | None
    pp_head      for a preposition (or a word starting with one): what the
                 phrase attaches to (المتعلَّق به)
    clause       relations of clauses the word heads (حال، صلة، خبر…)
    hidden       the omitted pronoun subject of a verb (ضمير مستتر), if any

usage: python layer2b/qac_convert.py QAC_DIR OUT.jsonl.gz
"""
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REL_AR = json.load(open(Path(__file__).parent / "resources" / "qac_relations.json", encoding="utf-8"))
NODE = re.compile(r"^(n\d+(?:,\s*n\d+)*)\s*=\s*(\w+)\((.*)\)\s*$")
EDGE = re.compile(r"^(\w+)\((n\d+)\s*-\s*(n\d+)\)\s*$")
LOC = re.compile(r"^(\d+):(\d+):(\d+)$")


def load_morphology(qac_dir, verse_words=None):
    """Segments of every word from QAC's morphology file, in the order the
    treebank numbers them. The article ال (DET) gets no node of its own."""
    out = defaultdict(list)
    for line in open(Path(qac_dir) / "quranic-corpus-morphology-0.4.txt", encoding="utf-8"):
        m = re.match(r"\((\d+):(\d+):(\d+):\d+\)\t(\S*)\t(\S+)\t(\S+)", line)
        if not m or m.group(5) == "DET":
            continue
        feats = m.group(6).split("|")
        out[tuple(int(x) for x in m.group(1, 2, 3))].append(
            {"tag": m.group(5), "kind": feats[0].lower(), "form": m.group(4),
             "lemma": next((f[4:] for f in feats if f.startswith("LEM:")), None),
             "aspect": next((f for f in feats if f in ("PERF", "IMPF", "IMPV")), None)})
    return out


def verse_word_counts(qac_dir):
    """(chapter, verse) → number of words, from QAC's own morphology file
    (quranic-corpus-morphology-0.4.txt, which numbers every segment)."""
    c = defaultdict(int)
    for line in open(Path(qac_dir) / "quranic-corpus-morphology-0.4.txt", encoding="utf-8"):
        m = re.match(r"\((\d+):(\d+):(\d+):\d+\)", line)
        if m:
            k = (int(m.group(1)), int(m.group(2)))
            c[k] = max(c[k], int(m.group(3)))
    counts = sorted(c.items())
    return counts, dict(counts)


def parse_graphs(path):
    for block in open(path, encoding="utf-8").read().split("\ngo\n"):
        nodes, phrases, edges = {}, {}, []
        for line in block.splitlines():
            line = line.strip()
            m = NODE.match(line)
            if m:
                names = [x.strip() for x in m.group(1).split(",")]
                kind, arg = m.group(2), m.group(3)
                if kind in ("word", "reference"):
                    loc = tuple(int(x) for x in arg.split(":"))
                    for i, nm in enumerate(names):
                        nodes[nm] = {"type": kind, "loc": loc, "seg": i, "nseg": len(names)}
                elif re.match(r"n\d+\s*-\s*n\d+", arg):
                    a, b = [int(x.strip()[1:]) for x in arg.split("-")]
                    phrases[names[0]] = {"type": kind, "span": (a, b)}
                else:
                    nodes[names[0]] = {"type": "omitted", "tag": kind, "text": "" if arg == "*" else arg}
                continue
            m = EDGE.match(line)
            if m:
                edges.append((m.group(1), m.group(2), m.group(3)))
        if nodes or edges:
            yield nodes, phrases, edges


def phrase_head(phrases, nodes, name):
    """The word that heads a phrase or clause: the verb of VS, the preposition of
    PP, otherwise its first word."""
    a, b = phrases[name]["span"]
    span = [f"n{i}" for i in range(a, b + 1) if f"n{i}" in nodes and nodes[f"n{i}"]["type"] != "omitted"]
    if not span:
        return None
    return span[0]


def describe(nm, nodes, phrases):
    if nm in phrases:
        h = phrase_head(phrases, nodes, nm)
        d = describe(h, nodes, phrases) if h else {"phrase": phrases[nm]["type"]}
        d["phrase"] = phrases[nm]["type"]
        return d
    n = nodes.get(nm)
    if n is None:
        return None
    if n["type"] == "omitted":
        return {"omitted": n["tag"], "text": n["text"]}
    return {"word": "%d:%d:%d" % n["loc"], "seg": n["seg"]}


KADA = {"kaAda", "EasaY", ">awoshaka", "karaba", "Tafiqa", ">axa*a", "jaEala", "{xolawolaq"}


def nasikh_name(rel, head, words):
    """QAC's subjx/predx cover كان، إنّ، لا and ما alike; name them by the governing word."""
    kind = "اسم" if rel == "subjx" else "خبر"
    tag = lemma = None
    key = tuple(map(int, head["word"].split(":"))) if isinstance(head, dict) and "word" in head else None
    if key in words:
        segs = words[key]["segments"]
        if head["seg"] < len(segs):
            tag, lemma = segs[head["seg"]]["tag"], segs[head["seg"]].get("lemma")
    elif isinstance(head, dict) and head.get("omitted"):
        tag = head["omitted"]
    if tag == "ACC":
        return kind + " إنّ"
    if tag == "NEG":
        return kind + (" ما" if lemma == "maA" else " لا")
    if tag == "V" and lemma in KADA:
        return kind + " كاد"
    if tag == "V":
        return kind + " كان"
    return None


def main():
    qac_dir, out = sys.argv[1], sys.argv[2]
    morph = load_morphology(qac_dir)
    words = {}
    stats = Counter()
    for nodes, phrases, edges in parse_graphs(Path(qac_dir) / "syntax.txt"):
        for nm, n in nodes.items():
            if n["type"] == "word" and n["loc"] not in words:
                segs = morph.get(n["loc"])
                words[n["loc"]] = {"loc": "%d:%d:%d" % n["loc"],
                                   "segments": [dict(s, edges=[]) for s in segs] if segs else [],
                                   "nodes": n["nseg"]}
                if segs and len(segs) != n["nseg"]:
                    stats["segment_count_mismatch"] += 1
        for rel, dep, head in edges:
            h = describe(head, nodes, phrases)
            if rel == "pred":                        # the head of a خبر is its مبتدأ
                hn = nodes.get(head) if head not in phrases else nodes.get(phrase_head(phrases, nodes, head))
                if hn and hn["type"] == "word":
                    words[hn["loc"]].setdefault("mubtada_of", describe(dep, nodes, phrases))
            item = {"rel": rel, "ar": REL_AR.get(rel, [rel])[0], "head": h}
            if dep in phrases:
                hd = phrase_head(phrases, nodes, dep)
                d = nodes.get(hd)
                if d and d["type"] == "word":
                    w = words[d["loc"]]
                    w.setdefault("clause", []).append(dict(item, phrase=phrases[dep]["type"]))
                    if phrases[dep]["type"] == "PP" and rel == "link":
                        w["pp_head"] = h
                continue
            d = nodes.get(dep)
            if not d:
                stats["edge_without_node"] += 1
                continue
            if d["type"] == "omitted":
                # an omitted dependent: a hidden pronoun subject (ضمير مستتر) of a verb,
                # or an omitted word in some other role
                hn = nodes.get(head)
                if hn and hn["type"] == "word" and rel in ("subj", "subjx", "pass") and d["tag"] == "PRON":
                    words[hn["loc"]]["hidden"] = {"rel": rel, "text": d["text"]}
                    stats["hidden_subject"] += 1
                else:
                    stats["omitted_dependent"] += 1
                continue
            if d["type"] == "reference":
                stats["reference_dependent"] += 1
                continue
            w = words[d["loc"]]
            if d["seg"] < len(w["segments"]):
                w["segments"][d["seg"]]["edges"].append(item)
    with gzip.open(out, "wt", encoding="utf-8") as o:
        for loc in sorted(words):
            w = words[loc]
            stem = next((s for s in w["segments"] if s["kind"] == "stem"), None)
            if stem and stem["edges"]:
                e = stem["edges"][0]
                w["function"], w["rel"], w["head"] = e["ar"], e["rel"], e["head"]
                if e["rel"] == "pred":
                    w["function"] = "خبر"
                elif e["rel"] in ("subjx", "predx"):
                    w["function"] = nasikh_name(e["rel"], e["head"], words)
                elif e["rel"] == "link" and stem["tag"] in ("T", "LOC"):
                    w["function"] = "ظرف زمان" if stem["tag"] == "T" else "ظرف مكان"
            elif w.get("pp_head") is not None:
                w["function"], w["rel"] = "جار ومجرور", "link"
            elif "mubtada_of" in w:
                w["function"], w["rel"] = "مبتدأ", "pred_head"
            elif stem and stem["tag"] == "V":
                w["function"] = {"PERF": "فعل ماض", "IMPF": "فعل مضارع", "IMPV": "فعل أمر"}.get(stem["aspect"], "فعل")
                w["rel"] = "verb"
            if w.get("clause"):
                c0 = w["clause"][0]
                w["clause_function"] = ("خبر" if c0["rel"] == "pred" else
                                        nasikh_name(c0["rel"], c0["head"], words) if c0["rel"] in ("subjx", "predx")
                                        else c0["ar"])
            if "function" in w:
                stats["with_function"] += 1
            stats["words"] += 1
            o.write(json.dumps(w, ensure_ascii=False) + "\n")
    print(dict(stats))


if __name__ == "__main__":
    main()
