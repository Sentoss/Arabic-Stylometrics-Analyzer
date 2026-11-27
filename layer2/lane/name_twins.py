"""Proper-noun twins for Lane words that our texts use as names.

Lane defines مُصْعَب as "an untractable stallion camel" and عَلَس as a kind of
wheat; in akhbār and isnāds they are almost always names (مصعب بن الزبير).
For each Lane stem seen at least twice in a name position in the corpus (after
بن، ابن، أبو، أبي، أبا، أم، بنت، عبد, or before بن), add a noun_prop entry with
the same vowels, so the model can choose between the word and the name from
context. Masculine stems get CAMeL's N0 class (like زَيْد، خالِد), stems with ة
its Nap class (like فاطِمَة); features are copied from those entries.

usage: python layer2/lane/name_twins.py CORPUS_DIR(with manifest_authorial.tsv) OUT_NAMES.tsv
"""
import csv
import math
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from probe_labels import MARK  # noqa: E402

RES = HERE.parent / "resources"
DB = Path.home() / ".camel_tools/data/morphology_db/calima-msa-r13/morphology.db"
WORD = re.compile(r"[ء-ي]+")
BEFORE = {"بن", "ابن", "أبو", "أبي", "أبا", "ابو", "ابي", "ابا", "أم", "ام", "بنت", "عبد",
          "وابن", "وأبو", "وأبي", "لابن", "لأبي", "بابن", "بأبي"}
AFTER = {"بن", "ابن"}
NOT_NAMES = {"هذا", "هذه", "ذلك"}              # function words that slipped into the count


def nb(w):
    return w.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")


def template(lex):
    for line in open(DB, encoding="utf-8"):
        if f" lex:{lex} " in line and "pos:noun_prop" in line:
            p = line.rstrip("\n").split("\t")
            return p[1], dict(x.split(":", 1) for x in p[2].split(" ") if ":" in x)
    raise SystemExit(f"no template {lex}")


def main(corpus, out_path, min_count=2):
    stems = {}
    for line in open(RES / "lane_stems.tsv", encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        f = dict(x.split(":", 1) for x in p[2].split(" ") if ":" in x)
        if f.get("num") == "p" or f.get("pos") not in ("noun", "adj"):
            continue
        fem = f.get("lex", "").split("_")[0].endswith("ة")
        stems.setdefault(nb(p[0]) + ("ة" if fem else ""), (p[0], f))
    root = Path(corpus)
    name, total, tokens = Counter(), Counter(), 0
    for r in csv.DictReader(open(root / "manifest_authorial.tsv", encoding="utf-8"), delimiter="\t"):
        toks = WORD.findall(MARK.sub("", (root / r["path"]).read_text(encoding="utf-8")))
        tokens += len(toks)
        for i, t in enumerate(toks):
            x = nb(t[2:] if t.startswith("ال") else t)
            if x not in stems:
                continue
            total[x] += 1
            if (i > 0 and toks[i - 1] in BEFORE) or (i + 1 < len(toks) and toks[i + 1] in AFTER):
                name[x] += 1
    from camel_tools.morphology.analyzer import Analyzer
    from camel_tools.morphology.database import MorphologyDB
    base = Analyzer(MorphologyDB.builtin_db("calima-msa-r13", "a"))
    masc_cat, masc = template("زَيْد_1")
    fem_cat, fem = template("فاطِمَة_1")
    rows = []
    for x, n in name.most_common():
        if n < min_count or x in NOT_NAMES:
            continue
        if any(a["pos"] == "noun_prop" and a.get("source") != "backoff" for a in base.analyze(x)):     # the dictionary already has the name
            continue
        bare, f0 = stems[x]
        diac = f0["diac"]
        is_fem = f0.get("lex", "").split("_")[0].endswith("ة")
        cat, f = (fem_cat, dict(fem)) if is_fem else (masc_cat, dict(masc))
        logp = round(math.log10(n / tokens), 4)
        lexname = f0["lex"].split("_")[0]
        f.update({"diac": diac, "lex": lexname + "_lanename", "bw": diac + "/NOUN_PROP",
                  "gloss": "name~" + f0.get("gloss", ""), "root": f0.get("root", ""),
                  "pattern": f0.get("pattern", ""), "source": "lane", "lex_logprob": logp, "pos_lex_logprob": logp,
                  "caphi": "_", "d1tok": diac, "d2tok": diac, "d3tok": diac, "atbtok": diac, "bwtok": diac,
                  "d1seg": diac, "d2seg": diac, "d3seg": diac, "atbseg": diac})
        rows.append((bare, cat, " ".join(f"{a}:{b}" for a, b in f.items()), n, total[x]))
    with open(out_path, "w", encoding="utf-8") as fo:
        for r in rows:
            fo.write("\t".join(str(v) for v in r) + "\n")
    print(len(rows), "name twins;", sum(r[3] for r in rows), "name-position occurrences of", sum(r[4] for r in rows))


if __name__ == "__main__":
    main(*sys.argv[1:3])
