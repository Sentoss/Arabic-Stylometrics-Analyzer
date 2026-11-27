"""Turn Lane's nouns, adjectives, plurals, maṣdars and feminines into stem
entries for CAMeL's morphology database (calima-msa-r13 format).

CAMeL's database stores *stems* with an inflection class (category): the
class says which prefixes and suffixes the stem takes (ة, duals, sound plurals,
diptote endings…). A Lane word gets the class most common among the database's
existing stems with the same pattern (وزن), the same feminine/plural status,
and noun/adjective part of speech: كَتِيبَة gets what other فَعِيلَة nouns have.

Which words are added (agreed rule, see check()): words whose bare form is
new to the dictionary, also with a proclitic attached; words that compete with
existing readings only if the full Tashkeela attests their vowelling often
enough; never words that, with a proclitic, have the same vowels as an existing
word (كَ + ثِير = كَثِير). The Tashkeela frequency sets each reading's prior
(lex_logprob), so rare readings stay unlikely.

usage: python layer2/lane/build_lane_stems.py lane_cov.jsonl TASHKEELA_FORMS.tsv OUT_STEMS.tsv [OUT_RARE.tsv]

OUT_RARE (trial): the rare competing words too ("shared_rare": Tashkeela does
not attest them often enough), with a very low prior (RARE_LOGP), so they win
only where the context clearly calls for their features.
"""
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from diac_freq import key  # noqa: E402
from probe_labels import MARK  # noqa: E402

DB = Path.home() / ".camel_tools/data/morphology_db/calima-msa-r13/morphology.db"
WEAK = set("ويىاءأإآؤئ")
RARE_LOGP = -9.0
PROCLITICS = ["", "وَ", "فَ", "بِ", "لِ", "كَ", "ال", "وَال", "فَال", "بِال", "لِل", "كَال"]


def lane_norm(w):
    """Lane's typography → ours: final ى after kasra / with shadda is ي; آء → اء."""
    w = w.replace("آء", "اء").replace("ٱ", "ا")
    w = re.sub("ِى", "ِي", w)
    w = re.sub("ىّ", "يّ", w)
    w = re.sub("ِىٌّ|ِىٌ", lambda m: m.group(0).replace("ى", "ي"), w)
    return w


def letters(s):
    return [c for c in s if not MARK.match(c)]


def pattern_of(stem, root):
    """DB-style pattern of a vowelled stem: each *strong* radical replaced by its
    position number (1-4); weak and hamza radicals stay as letters (قَوْل →
    1َوْ3, رَأْس → 1َأْ3); a doubled radical written once with shadda keeps the
    shadda (مَدّ → 1َ2ّ). None if a strong radical cannot be found in order."""
    rad = [c for c in root if c not in "#."]
    chars = list(stem)
    pos = 0
    for i, r in enumerate(rad):
        if r in WEAK:
            continue
        j = pos
        while j < len(chars) and chars[j] != r:
            j += 1
        if j == len(chars):
            if i > 0 and rad[i - 1] == r:       # doubled radical: already covered by the shadda
                continue
            return None
        chars[j] = str(i + 1)
        pos = j + 1
    pat = "".join(chars)
    return pat if any(ch.isdigit() for ch in pat) else None


def stem_of(w):
    """Lane headword → DB-style stem: no case ending (tanwin / final vowel),
    feminine ة and the fatha before it removed, no fatha before ا."""
    w = re.sub("ً?ا?ً$", lambda m: m.group(0).replace("ً", ""), w) if w.endswith(("ًا", "اً")) else w
    while w and w[-1] in "ًٌٍَُِ":
        w = w[:-1]
    fem = w.endswith("ة")
    if fem:
        w = w[:-1]
        if w and w[-1] == "َ":
            w = w[:-1]
    w = w.replace("َا", "ا")
    return w, fem


def load_db_classes():
    """(pattern, fem, plural, pos) -> Counter(category), and an example feature line per key."""
    cls, example = defaultdict(Counter), {}
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
        if len(p) < 3:
            continue
        f = dict(x.split(":", 1) for x in p[2].split(" ") if ":" in x)
        if f.get("pos") not in ("noun", "adj"):
            continue
        fem = f.get("lex", "").split("_")[0].endswith("ة")
        plural = f.get("num") == "p"
        k = (f.get("pattern"), fem, plural)
        cls[k][p[1]] += 1
        example.setdefault((k, p[1]), f)
    return cls, example


def load_forms(path):
    c = {}
    for line in open(path, encoding="utf-8"):
        k, n = line.rstrip("\n").split("\t")
        c[k] = int(n)
    return c


BARE_PRE = ["و", "ف", "ب", "ل", "ك", "ال", "وال", "فال", "بال", "لل", "كال"]
BARE_SUF = ["ة", "ه", "ها", "هم", "ي", "ك", "ات", "ين", "ون", "ان", "ا", "تي", "ته"]
VOC_PRE = {"و": "وَ", "ف": "فَ", "ب": "بِ", "ل": "لِ", "ك": "كَ", "ال": "ال", "وال": "وَال",
           "فال": "فَال", "بال": "بِال", "لل": "لِل", "كال": "كَال"}


class Base:
    """The current dictionary (CAMeL + Classical supplement), cached."""
    def __init__(self):
        from supplement import ClassicalAnalyzer, VARIANTS
        self.an = ClassicalAnalyzer(passives=True, variants=VARIANTS)
        self.cache = {}

    def keys(self, bare):
        if bare not in self.cache:
            self.cache[bare] = {key(a["diac"]) for a in self.an.analyze(bare) if a.get("source") != "backoff"}
        return self.cache[bare]


def check(word, base, forms):
    """Decide whether a Lane word may be added. Returns (status, attested count, why).

    known    the dictionary already has this vowelling          -> skip
    new      no reading for the bare form or any clitic + form   -> add
    shared   competes with existing readings                     -> add only if Tashkeela
             attests this vowelling often enough (>= 3 times and >= 2% of the
             competing readings' count)
    blocked  clitic + Lane word has the same vowels as an existing word
             (كَ + ثِير = كَثِير، فَ + سَخ = فَسَخَ): it would add a false
             segmentation that vowels cannot rule out                  -> skip
    """
    k = key(word)
    bare = "".join(letters(word)).replace("ٱ", "ا")
    own = base.keys(bare)
    if k in own:
        return "known", 0, ""
    n_att = forms.get(k, 0)
    competitors = set(own)
    for p in BARE_PRE:
        ks = base.keys(p + bare)
        if not ks:
            continue
        if key(VOC_PRE[p] + word) in ks:
            return "blocked", n_att, p
        competitors |= ks
    stem_bare = bare[:-1] if bare.endswith("ة") else bare
    for suf in BARE_SUF:                      # شَهّاد + ة = شهادة: inflected forms compete too
        competitors |= base.keys(stem_bare + suf)
    if not competitors:
        return "new", n_att, ""
    comp = sum(forms.get(c, 0) for c in competitors)
    if n_att >= 3 and n_att >= 0.02 * comp:
        return "shared", n_att, f"{n_att}/{comp}"
    return "shared_rare", n_att, f"{n_att}/{comp}"


def main(cov_path, forms_path, out_path, rare_path=None):
    cls, example = load_db_classes()
    forms = load_forms(forms_path)
    base = Base()
    total = sum(forms.values())
    stats, out = Counter(), []
    seen = set()
    for line in open(cov_path, encoding="utf-8"):
        r = json.loads(line)
        if r["kind"] not in ("noun", "plural", "masdar", "fem"):
            continue
        if r["status"].endswith("_unvowelled"):
            stats["skip: unvowelled"] += 1
            continue
        w = lane_norm(r["w"])
        k = key(w)
        status, n_att, why = check(w, base, forms)
        if status not in ("new", "shared") and not (rare_path and status == "shared_rare"):
            stats["skip: " + status] += 1
            continue
        stem, fem = stem_of(w)
        plural = r["kind"] == "plural"
        root = "".join(c for c in r["root"] if not MARK.match(c))
        pat = pattern_of(stem, root)
        if not pat:
            stats["skip: pattern not found"] += 1
            continue
        cands = cls.get((pat, fem, plural))
        if not cands:
            stats["skip: no class for pattern"] += 1
            continue
        cat = cands.most_common(1)[0][0]
        if (stem, cat) in seen:
            stats["skip: duplicate"] += 1
            continue
        seen.add((stem, cat))
        f = dict(example[((pat, fem, plural), cat)])
        lex = stem + ("َة" if fem else "") + "_lane"
        logp = round(math.log10((n_att + 0.5) / total), 4)
        if status == "shared_rare":
            logp = min(logp, RARE_LOGP)
        f.update({"diac": stem, "lex": lex, "bw": stem + "/" + f.get("pos", "noun").upper(),
                  "gloss": "Lane:" + (r.get("entry") or ""), "root": ".".join(letters(root)),
                  "pattern": pat, "source": "lane", "lex_logprob": logp, "pos_lex_logprob": logp,
                  "caphi": "_", "d1tok": stem, "d2tok": stem, "d3tok": stem, "atbtok": stem, "bwtok": stem,
                  "d1seg": stem, "d2seg": stem, "d3seg": stem, "atbseg": stem})
        bare_stem = "".join(letters(stem)).replace("ٱ", "ا")
        out.append((bare_stem, cat, " ".join(f"{a}:{b}" for a, b in f.items()), status, n_att, r["kind"]))
        stats[f"added: {status}" + (" (attested)" if n_att else "")] += 1
    with open(out_path, "w", encoding="utf-8") as fo:
        for row in out:
            if row[3] != "shared_rare":
                fo.write("\t".join(str(x) for x in row) + "\n")
    if rare_path:
        with open(rare_path, "w", encoding="utf-8") as fo:
            for row in out:
                if row[3] == "shared_rare":
                    fo.write("\t".join(str(x) for x in row) + "\n")
    for k_, v in sorted(stats.items()):
        print(f"  {k_:34} {v}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
