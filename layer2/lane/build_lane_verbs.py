"""Add Lane's verbs to CAMeL's database by analogy with existing verbs.

CAMeL stores a verb as several stems (perfect, imperfect, imperative, their
passives; weak roots have several shapes: قال/قُل/قُول/قِيل), each with its own
inflection class. A Lane verb copies the full stem set of an existing verb
with the same lemma pattern, filling in its own radicals: صالَ (ص و ل, aor. u)
copies قالَ-u, so it gets صُل، صال، صُول، صِيل with قالَ's classes.

Lemma pattern = the perfect with strong radicals numbered (كَتَب → 1َ2َ3,
قال → 1ا3, رَمَى → 1َ2َى); for Form I also the imperfect vowel: from Lane
("aor."), else from Arramooz's conjugation class (باب: 1 a-u, 2 a-i, 3 a-a,
4 i-a, 5 u-u, 6 i-i), else the verb is skipped. Derived forms (II-X) take
their pattern from the vowelled headword; unvowelled headwords of sound roots
get the form's standard vowelling.

The same add/skip rules as for nouns (build_lane_stems.check), applied to the
perfect: known -> skip; blocked (a proclitic + the verb has the vowels of an
existing word) -> skip; competing -> only if Tashkeela attests the perfect.

usage: python layer2/lane/build_lane_verbs.py lane_cov.jsonl TASHKEELA_FORMS.tsv ARRAMOOZ_VERBS.csv OUT.tsv
"""
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from probe_labels import MARK  # noqa: E402
from build_lane_stems import DB, Base, check, lane_norm, letters, load_forms  # noqa: E402

BAB = {"1": "u", "2": "i", "3": "a", "4": "a", "5": "u", "6": "i"}         # imperfect vowel
BAB_PV = {"1": "a", "2": "a", "3": "a", "4": "i", "5": "u", "6": "i"}      # perfect vowel
WEAK = set("وياىءأإآؤئ#")
HAMZA = "ءأإآؤئ"
# standard perfects of the derived forms for sound roots (radicals 1 2 3)
FORM_TEMPLATES = {2: "1َ2َّ3", 3: "1ا2َ3", 4: "أَ1ْ2َ3", 5: "تَ1َ2َّ3", 6: "تَ1ا2َ3",
                  7: "اِنْ1َ2َ3", 8: "اِ1ْتَ2َ3", 10: "اِسْتَ1ْ2َ3"}


def radicals(root):
    out = []
    for c in root:
        if c in ".":
            continue
        out.append("ء" if c in HAMZA else ("ي" if c == "ى" else c))
    return out


def canon(w):
    """One order for the marks on a letter: shadda first (عَلَّم = ل + ّ + َ);
    the connecting alif written plain (the database writes ٱِجْتَمَع)."""
    w = w.replace("ٱ", "ا")
    out, marks = [], []
    for ch in w:
        if MARK.match(ch):
            marks.append(ch)
        else:
            out += sorted(marks, key=lambda m: m != "ّ") + [ch]
            marks = []
    return "".join(out + sorted(marks, key=lambda m: m != "ّ"))


def lemma_pattern(verb, rad):
    """Number the strong radicals of a vowelled perfect by their position."""
    chars, pos = list(canon(verb)), 0
    for i, r in enumerate(rad):
        if r in WEAK:
            continue
        j = pos
        while j < len(chars) and chars[j] != r:
            j += 1
        if j == len(chars):
            if i > 0 and rad[i - 1] == r:          # doubled: covered by the shadda
                continue
            return None
        chars[j] = str(i + 1)
        pos = j + 1
    p = "".join(chars)
    return p if any(c.isdigit() for c in p) else None


def verb_norm(w):
    """Lane's spelling of verbs → the database's: connecting alif اِ (not إِ / ٱ),
    hamza on an initial alif with fatha (اَبَّرَ → أَبَّرَ), no sukun on alif."""
    w = w.replace("ٱ", "ا").replace("اْ", "ا")
    if w.startswith("إِ"):
        w = "اِ" + w[2:]
    if w.startswith("اَ"):
        w = "أَ" + w[2:]
    return w


def fill(pattern, rad):
    return "".join(rad[int(c) - 1] if c.isdigit() and int(c) <= len(rad) else c for c in pattern)


def load_templates():
    """lemma pattern (+ '-vowel' for Form I) -> list of (lex, [stem lines as (cat, feats)])."""
    by_lex = defaultdict(list)
    in_stems = False
    for line in open(DB, encoding="utf-8"):
        if line.startswith("###STEMS###"):
            in_stems = True
            continue
        if line.startswith("###TABLE"):
            break
        if not in_stems:
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) < 3 or "pos:verb" not in p[2]:
            continue
        f = dict(x.split(":", 1) for x in p[2].split(" ") if ":" in x)
        if not f.get("lex", "").endswith("_1"):
            continue
        by_lex[f["lex"]].append((p[1], f))
    idx = defaultdict(list)
    for lex, lines in by_lex.items():
        base = lex[:-2]
        verb, _, vowel = base.partition("-")
        rad = [c for c in lines[0][1].get("root", "").split(".")]
        pat = lemma_pattern(verb, rad)
        if pat:
            idx[pat + ("-" + vowel if vowel else "")].append((lex, lines))
    for k in idx:                                   # most complete stem set first
        idx[k].sort(key=lambda x: -len(x[1]))
    return idx


def arramooz(path):
    cls = defaultdict(set)
    for row in csv.reader(open(path, encoding="utf-8"), delimiter="\t"):
        if len(row) < 3 or row[0].startswith("#"):
            continue
        bare = MARK.sub("", row[0]).replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
        cls[(bare, row[1])].add(row[2].strip())
    return cls


def main(cov_path, forms_path, arr_path, out_path):
    idx = load_templates()
    forms = load_forms(forms_path)
    total = sum(forms.values())
    arr = arramooz(arr_path)
    base = Base()
    stats, out, seen = Counter(), [], set()
    for line in open(cov_path, encoding="utf-8"):
        r = json_loads(line)
        if r["kind"] != "verb":
            continue
        form = r.get("form")
        rad = radicals("".join(c for c in r["root"] if not MARK.match(c)))
        if len(rad) != 3 or form not in (1, 2, 3, 4, 5, 6, 7, 8, 10):
            stats["skip: not a triliteral form I-X"] += 1
            continue
        w = verb_norm(lane_norm(r["w"]).split()[0])
        sound = not any(x in WEAK for x in rad) and rad[1] != rad[2]
        if form == 8 and rad[0] in "دذزصضطظث":       # اِدَّكَرَ، اِصْطَبَرَ: the ت assimilates
            sound = False
        if form != 1 and sound:
            # derived form of a sound root: its vowelling is fixed by the form
            # (Lane often leaves these unvowelled or half-vowelled: أبّد، تأبّد)
            w = fill(FORM_TEMPLATES[form], rad) + "َ"
        elif len(MARK.findall(w)) < len(letters(w)) - 2 or w.endswith(("تُ", "ٌ", "ٍ", "ً")):
            stats["skip: unvowelled or not a 3ms perfect"] += 1
            continue
        perfect = w[:-1] if w.endswith("َ") else w   # lemma: no final fatha (كَتَب)
        vowel = ""
        if form == 1:
            aor = r.get("aor")
            if aor in ("a", "u", "i"):
                vowel = aor
            else:
                bare = "".join(letters(perfect)).replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
                c = arr.get((bare, "".join(rad).replace("ء", "ء")), set())
                if len(c) == 1:
                    vowel = BAB[c.pop()]
                    stats["aorist from Arramooz"] += 1
                else:
                    stats["skip: no aorist vowel"] += 1
                    continue
        pat = lemma_pattern(perfect, rad)
        key = (pat or "") + ("-" + vowel if vowel else "")
        tmpl = idx.get(key)
        if not tmpl:
            stats["skip: no template verb"] += 1
            continue
        status, n_att, why = check(perfect + "َ", base, forms)
        if status not in ("new", "shared"):
            stats["skip: " + status] += 1
            continue
        lex = perfect + ("-" + vowel if vowel else "") + "_lane"
        if lex in seen:
            stats["skip: duplicate"] += 1
            continue
        seen.add(lex)
        t_lex, lines = tmpl[0]
        logp = round(math.log10((n_att + 0.5) / total), 4)
        for cat, f0 in lines:
            diac = fill(f0["pattern"], rad)
            f = dict(f0)
            f.update({"diac": diac, "lex": lex, "bw": diac + "/" + f0.get("bw", "/").split("/")[-1],
                      "gloss": f"Lane:{r.get('entry') or ''}~like:{t_lex}", "root": ".".join(rad),
                      "source": "lane", "lex_logprob": logp, "pos_lex_logprob": logp, "caphi": "_",
                      "d1tok": diac, "d2tok": diac, "d3tok": diac, "atbtok": diac, "bwtok": diac,
                      "d1seg": diac, "d2seg": diac, "d3seg": diac, "atbseg": diac})
            bare = "".join(letters(diac))
            out.append((bare, cat, " ".join(f"{a}:{b}" for a, b in f.items()), status, n_att, f"verb{form}"))
        stats[f"added: form {form}, {status}"] += 1
    with open(out_path, "w", encoding="utf-8") as fo:
        for row in out:
            fo.write("\t".join(str(x) for x in row) + "\n")
    for k, v in sorted(stats.items()):
        print(f"  {k:34} {v}")
    print(len(out), "stem lines")


def json_loads(s):
    import json
    return json.loads(s)


if __name__ == "__main__":
    main(*sys.argv[1:5])
