"""Classical supplement to CAMeL's MSA analyzer.

Two kinds of extra readings are added to what the analyzer returns:

1. Passives, by rule, for every active perfect or imperfect verb reading
   (the MSA database has them only for some verbs):
     perfect:   ذَكَرَ → ذُكِرَ   عَلَّمَ → عُلِّمَ   قاتَلَ → قُوتِلَ   أَكْرَمَ → أُكْرِمَ
                اِسْتَخْرَجَ → اُسْتُخْرِجَ   قالَ → قِيلَ   رَمَى → رُمِيَ   مَدَّ → مُدَّ
     imperfect: يَذْكُرُ → يُذْكَرُ   يُعَلِّمُ → يُعَلَّمُ   يَقُولُ → يُقالُ   يَرْمِي → يُرْمَى

2. Classical variants harvested from vowelled text (harvest_variants.py),
   stored in resources/classical_variants.tsv: the same lemma with another
   vowelling of its stem (ثُلْث → ثُلُث, جُمْعَة → جُمُعَة, يُوسِف → يُوسُف,
   عَجُزَ → عَجَزَ), and Form II verbs missing from the MSA database, derived
   from the Form I entry by doubling the middle radical (حَدَثَ → حَدَّثَ, so
   حَدَّثَنا gets a reading, with its own lemma).

Supplemented readings carry source="classical_passive" or "classical_variant".
"""
import csv
import re
from pathlib import Path

from camel_tools.morphology.analyzer import Analyzer
from camel_tools.morphology.database import MorphologyDB

FATHA, DAMMA, KASRA, SUKUN, SHADDA = "َ", "ُ", "ِ", "ْ", "ّ"
VOWELS = FATHA + DAMMA + KASRA
MARKS = re.compile("[ً-ْٰ]")
UNIT = re.compile("([^ً-ْٰ])([ً-ْٰ]*)")
VARIANTS = Path(__file__).parent / "resources" / "classical_variants.tsv"


def units(s):
    return [[c, m] for c, m in UNIT.findall(s)]


def join(us):
    return "".join(c + m for c, m in us)


def set_vowel(marks, v):
    keep = "".join(ch for ch in marks if ch not in VOWELS and ch != SUKUN)
    return keep + v


def has_vowel(marks):
    return any(ch in VOWELS for ch in marks)


def passive_perfect(stem):
    u = units(stem)
    if len(u) < 2:
        return None
    last = len(u) - 1
    # geminate Form I: مَدّ → مُدّ
    if len(u) == 2 and SHADDA in u[1][1]:
        u[0][1] = set_vowel(u[0][1], DAMMA)
        return join(u)
    # defective: final ى/ا (3ms) or ي/و (other persons) → ي after kasra
    if u[last][0] in "ىا" and last >= 1 and not u[last][1]:
        u[last] = ["ي", ""]
        u[last - 1][1] = set_vowel(u[last - 1][1], KASRA)
        tail_fatha = True
    else:
        tail_fatha = False
    if u[last][0] in "وي" and SUKUN in u[last][1] or (u[last][0] == "و" and not u[last][1]):
        u[last][0] = "ي"
    # hollow: long ا right before the last radical → ِي
    hollow = last >= 2 and u[last - 1][0] == "ا" and not u[last - 1][1]
    for i in range(last):
        c, m = u[i]
        if c == "ا" and not m:
            continue
        if not m and i + 1 < len(u) and u[i + 1][0] == "ا" and not u[i + 1][1]:
            m = FATHA      # CAMeL leaves the vowel before a long ā unwritten
        if tail_fatha and i == last - 1:
            continue
        if hollow and i == last - 2:
            u[i][1] = set_vowel(m, KASRA)
        elif i == last - 1 and c not in "اوي" or (i == last - 1 and m):
            u[i][1] = set_vowel(m, KASRA) if has_vowel(m) else m
        elif has_vowel(m):
            u[i][1] = set_vowel(m, DAMMA)
    for i in range(last):
        # long ā after the first radical (Forms III, VI): قاتَل → قُوتِل
        if u[i][0] == "ا" and not u[i][1] and i >= 1 and not (hollow and i == last - 1):
            u[i][0] = "و"
    if hollow:
        u[last - 1][0] = "ي"
    out = join(u)
    if tail_fatha:
        out += FATHA if not out.endswith(FATHA) else ""
    return out


def passive_imperfect(stem):
    u = units(stem)
    if len(u) < 2:
        return None
    last = len(u) - 1
    # defective: final long ي/و → ى after fatha (يَرْمِي → يُرْمَى)
    if u[last][0] in "يو" and not u[last][1] and has_vowel(u[last - 1][1]):
        u[last - 1][1] = set_vowel(u[last - 1][1], FATHA)
        u[last][0] = "ى"
        return join(u)
    # hollow: long و/ي before the last radical → ا (يَقُول → يُقال)
    if last >= 2 and u[last - 1][0] in "وي" and not u[last - 1][1] and has_vowel(u[last - 2][1]):
        u[last - 2][1] = set_vowel(u[last - 2][1], "")   # CAMeL: no fatha before long ā
        u[last - 1][0] = "ا"
        return join(u)
    # sound: the vowel before the last radical becomes fatha
    for i in range(last - 1, -1, -1):
        if has_vowel(u[i][1]):
            u[i][1] = set_vowel(u[i][1], FATHA)
            break
    return join(u)


def stem_of(a):
    """(index, stem text, tag) of the stem morpheme in the bw segmentation."""
    parts = [p.rsplit("/", 1) for p in a.get("bw", "").split("+") if "/" in p]
    for i, (txt, tag) in enumerate(parts):
        if re.fullmatch(r"(PV|IV|CV)(_PASS)?", tag) or tag.startswith(("NOUN", "ADJ")):
            return i, txt, tag, parts
    return None


def make_passive(a):
    if a.get("pos") != "verb" or a.get("vox") != "a" or a.get("asp") not in ("p", "i"):
        return None
    if "dobj" in str(a.get("enc0", "")):
        return None
    st = stem_of(a)
    if not st:
        return None
    i, stem, tag, parts = st
    new = passive_perfect(stem) if a["asp"] == "p" else passive_imperfect(stem)
    if not new or stem not in a["diac"]:
        return None
    if new == stem and a["asp"] == "p":
        return None       # imperfect: the prefix vowel alone can make the passive (يَجْعَل → يُجْعَل)
    diac = a["diac"].replace(stem, new, 1)
    if a["asp"] == "i" and i > 0:
        pre = parts[i - 1][0]
        if pre and pre in diac:
            diac = diac.replace(pre, set_prefix(pre), 1)
    if diac == a["diac"]:
        return None
    b = dict(a)
    b.update(diac=diac, vox="p", source="classical_passive",
             bw=a["bw"].replace(f"{stem}/{tag}", f"{new}/{tag}_PASS", 1),
             pos_lex_logprob=a.get("pos_lex_logprob", -99) - 1.0)
    return b


def set_prefix(pre):
    u = units(pre)
    if u:
        u[-1][1] = set_vowel(u[-1][1], DAMMA)
    return join(u)


class ClassicalAnalyzer:
    """CAMeL's analyzer plus Classical passives and harvested variants."""

    def __init__(self, analyzer=None, variants=VARIANTS, passives=True):
        self.an = analyzer or Analyzer(MorphologyDB.builtin_db())
        self.passives = passives
        self.variants = {}
        if variants and Path(variants).exists():
            for r in csv.DictReader(open(variants, encoding="utf-8"), delimiter="\t"):
                self.variants.setdefault((r["lex"], r["pos"], r["stem"]), []).append(r)

    def analyze(self, word):
        base = [a for a in self.an.analyze(word) if a.get("source") != "backoff"]
        out = list(base)
        seen = {(a["diac"], a["lex"], a["pos"]) for a in base}

        def add(b):
            k = (b["diac"], b["lex"], b["pos"])
            if k not in seen:
                seen.add(k)
                out.append(b)

        if self.variants:
            for a in base:
                st = stem_of(a)
                if not st:
                    continue
                for v in self.variants.get((a["lex"], a["pos"], st[1]), []):
                    if st[1] not in a["diac"]:
                        continue
                    b = dict(a)
                    b.update(diac=a["diac"].replace(st[1], v["new_stem"], 1),
                             lex=v["new_lex"] or a["lex"], source="classical_variant",
                             pos_lex_logprob=a.get("pos_lex_logprob", -99) - 0.5)
                    add(b)
        if self.passives:
            for a in list(out):
                p = make_passive(a)
                if p:
                    add(p)
        return out
