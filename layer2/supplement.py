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

3. Imperatives, derived from the 2nd-person jussive (the MSA database has
   almost none): تَكْتُبْ → اُكْتُبْ, تَعْلَمْ → اِعْلَمْ, تُكْرِمْ → أَكْرِمْ,
   تُعَلِّمْ → عَلِّمْ, تَسْتَغْفِرْ → اِسْتَغْفِرْ, تَقُلْ → قُلْ, تَرْمِ → اِرْمِ,
   تَمُدَّ → مُدَّ, تَكْتُبُوا → اُكْتُبُوا; also after وَ/فَ (وَاعْلَمْ, فَانْظُرْ),
   and the irregular خُذْ, كُلْ, مُرْ.

4. The interrogative hamza on any word (the database allows it only on a few,
   e.g. أَكانَ, أَلَيْسَ): أَرَأَيْتَ, أَتَعْرِفُ, أَهٰذا, أَفَلا, أَوَلَمْ.

5. Fixes: the lightened كَأَنْ (lemma كَأَنَّ), and the lemma of the verb رَأَى,
   which the database gives as راوَنْد ("rhubarb").

Supplemented readings carry source="classical_passive", "classical_variant",
"classical_imperative", "classical_question" or "classical_manual".
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


def bare_letters(s):
    return MARKS.sub("", s).replace("ٱ", "ا")


def make_imperative(a, kind, final_vowel=None):
    """Imperative from a 2nd-person jussive/imperfect reading (prefix تَ / تُ removed).
    final_vowel: for defective verbs, the vowel before the dropped long vowel
    (يَرْمِي → kasra, يَدْعُو → damma, يَرْضَى → fatha)."""
    d = a["diac"]
    m = re.match("ت([\u064e\u064f])", d)
    if not m:
        return None
    damma_prefix = m.group(1) == DAMMA
    body = d[2:]
    if kind == "hamza":            # خُذْ، كُلْ، مُرْ
        if a.get("lex") not in ("أَخَذ", "أَكَل", "أَمَر") or not body.startswith("أ"):
            return None
        body = re.sub("^أ\u0652?", "", body)
    u = units(body)
    if not u:
        return None
    two_ms = a.get("num") == "s" and a.get("gen") == "m"
    if two_ms:
        last = len(u) - 1
        if u[last][0] in "يوى" and not u[last][1] and last >= 1:
            u = u[:-1]                                  # تَرْمِي → (اِ)رْمِ، تُعَنِّي → عَنِّ
            u[-1][1] = set_vowel(u[-1][1], final_vowel or KASRA) if not has_vowel(u[-1][1]) or final_vowel \
                else u[-1][1]
        elif last >= 1 and u[last - 1][0] in "اوي" and not u[last - 1][1] and not has_vowel(u[last][1]):
            del u[last - 1]                             # تَقُولُ → قُلْ
            u[-1][1] = SUKUN
        elif SHADDA in u[last][1]:
            u[last][1] = SHADDA + FATHA                 # تَمُدُّ → مُدَّ
        elif not has_vowel(u[last][1]) and SUKUN not in u[last][1]:
            if final_vowel:                             # shortened defective stem تَرْم
                u[last][1] = final_vowel
            else:
                u[last][1] = SUKUN
    lex = a.get("lex", "")
    if kind == "hamza":
        head = ""
    elif damma_prefix and lex.startswith(("أ", "آ")):
        head = "أَ"                                     # Form IV: تُكْرِمْ → أَكْرِمْ، تُقِلْ → أَقِلْ
    elif SUKUN in u[0][1] or (kind == "alif" and SHADDA in u[0][1]):
        # consonant cluster (اُكْتُبْ، اِذْهَبْ، اُدْعُ) or doubled first letter
        # of Form VIII from a weak root (تَتَّقِي → اِتَّقِ)
        first = next((mm for _, mm in u[1:] if has_vowel(mm)), "") if SUKUN in u[0][1] else ""
        head = "اُ" if DAMMA in first else "اِ"
    else:
        head = ""                                       # عَلِّمْ، قاتِلْ، تَعَلَّمْ، قُلْ
    return head + join(u)


class ClassicalAnalyzer:
    """CAMeL's analyzer plus Classical passives and harvested variants."""

    def __init__(self, analyzer=None, variants=VARIANTS, passives=True, keep_backoff=False,
                 cache_size=200000, mode="labels"):
        """keep_backoff: also return the analyzer's backoff guesses (proper
        nouns), as layer 1 needs; the label builder leaves them out.
        mode="labels": every supplement reading (the printed vowels decide).
        mode="text": for unvowelled text, where only a context model decides:
        no interrogative readings, and imperatives only for words with no other
        reading (the MSA model cannot be trusted with them: it read the name
        أهرمن as أَ + هَرَّمَ). Revisit once the Classical model is trained."""
        self.mode = mode
        self.an = analyzer or Analyzer(MorphologyDB.builtin_db())
        self.passives = passives
        self.keep_backoff = keep_backoff
        self._cache, self._cache_size = {}, cache_size
        self.variants = {}
        if variants and Path(variants).exists():
            for r in csv.DictReader(open(variants, encoding="utf-8"), delimiter="\t"):
                self.variants.setdefault((r["lex"], r["pos"], r["stem"]), []).append(r)

    def _base(self, word):
        return [a for a in self.an.analyze(word) if a.get("source") != "backoff"]

    def imperatives(self, word):
        """Imperatives of `word` derived from 2nd-person jussive readings."""
        out = []
        if len(word) < 2:
            return out          # single letters are abbreviations (ه، ع، ش), not قِ / عِ
        procl = [("", {}, word)]
        if len(word) > 2 and word[0] in "وف":
            procl.append(("وَ" if word[0] == "و" else "فَ",
                          {"prc2": "wa_conj" if word[0] == "و" else "fa_conj"}, word[1:]))
        for pdiac, pfeat, rest in procl:
            cands = []
            stems = [rest[1:]] if rest[:1] in "اأإ" else [rest]
            kind0 = "alif" if rest[:1] in "اأإ" else "none"
            for st in stems:
                cands.append((kind0, "ت" + st))
                for lv in "يوى":                             # defective: تَرْمِي، تَدْعُو، تَرْضَى
                    cands.append((kind0, "ت" + st + lv))
                if len(st) == 2:                              # hollow: تَقُولُ → قُلْ
                    for lv in "وي":
                        cands.append((kind0, "ت" + st[0] + lv + st[1]))
            if rest[:1] in "خكم" and len(rest) >= 2:
                cands.append(("hamza", "تأ" + rest))           # خُذْ، كُلْ، مُرْ
            for kind, imperf in cands:
                for a in self._base(imperf):
                    if a.get("pos") != "verb" or a.get("asp") != "i" or a.get("per") != "2" \
                            or a.get("vox") != "a" or any(str(a.get(f, "0")) not in ("0", "na")
                                                          for f in ("prc0", "prc1", "prc2", "prc3")):
                        continue
                    fv = None
                    if imperf.endswith(("ي", "و", "ى")) and a.get("num") == "s" and a.get("gen") == "m":
                        dd = a["diac"]
                        mm = re.search("([\u064e\u064f\u0650])[يوى]$", dd)
                        fv = mm.group(1) if mm else (FATHA if dd.endswith("ى") else None)
                    elif str(a.get("root", "")).endswith("#") and a.get("num") == "s" and a.get("gen") == "m":
                        for lv in "يوى":                         # shortened stem: get the vowel
                            for f in self._base(imperf + lv):
                                if f.get("lex") == a.get("lex") and f.get("per") == "2":
                                    mm = re.search("([\u064e\u064f\u0650])[يوى]$", f["diac"])
                                    fv = fv or (mm.group(1) if mm else (FATHA if lv == "ى" else None))
                    two_fp = a.get("gen") == "f" and a.get("num") == "p"
                    two_ms = a.get("gen") == "m" and a.get("num") == "s"
                    if not (two_ms or two_fp) and bare_letters(a["diac"]).endswith("ن"):
                        continue        # indicative تُوَرِّدانِ / تَكْتُبُونَ / تَكْتُبِينَ: no imperative
                    imp = make_imperative(a, kind, fv)
                    if not imp or bare_letters(imp).lstrip("اأإ") != bare_letters(rest).lstrip("اأإ") \
                            or (kind == "alif") != (bare_letters(imp)[:1] in "اأإ"):
                        continue
                    b = dict(a)
                    b.update(pfeat)
                    b.update(diac=pdiac + imp, asp="c", mod="i", source="classical_imperative",
                             bw=re.sub(r"^[^+]*/IV2\w*\+", "", a["bw"]).replace("/IV", "/CV", 1),
                             pos_lex_logprob=a.get("pos_lex_logprob", -99) - 1.0)
                    out.append(b)
        return out

    def question(self, word):
        """Readings with the interrogative hamza: أَ + the rest of the word."""
        if len(word) < 3 or word[0] != "أ":
            return []
        out = []
        for a in self._core(word[1:], question=False):
            if str(a.get("prc3", "0")) not in ("0", "na") or bare_letters(a["diac"]) != word[1:]:
                continue
            # verbs, pronouns, demonstratives, particles: أَرَأَيْتَ، أَهُوَ، أَهٰذا، أَلَيْسَ.
            # Not bare nouns/adjectives: there the hamza is usually an elative the
            # dictionary lacks (أَعَمُّ، أَوْرَعُ), not a question.
            if not str(a.get("pos", "")).startswith(("verb", "pron", "part", "conj", "adv")):
                continue
            b = dict(a)
            b.update(diac="أَ" + a["diac"], prc3=">a_ques", source="classical_question",
                     bw="أَ/INTERROG_PART+" + a.get("bw", ""),
                     pos_lex_logprob=a.get("pos_lex_logprob", -99) - 1.5)
            out.append(b)
        return out

    def __getattr__(self, name):          # all_feats(), tok_feats() … of the wrapped analyzer
        return getattr(self.__dict__["an"], name)

    def analyze(self, word):
        hit = self._cache.get(word)
        if hit is None:
            hit = self._core(word, question=True)
            if self.keep_backoff:
                hit = hit + [a for a in self.an.analyze(word) if a.get("source") == "backoff"]
            if len(self._cache) >= self._cache_size:
                self._cache.clear()
            self._cache[word] = hit
        return [dict(a) for a in hit]

    def _core(self, word, question=True):
        base = self._base(word)
        for a in base:     # database error: the verb رَأَى carries the lemma راوَنْد
            if a.get("lex") == "راوَنْد" and a.get("pos") == "verb":
                a["lex"] = "رَأَى"
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
        if self.mode == "labels" or not out:
            for b in self.imperatives(word):
                if b.get("lex") == "راوَنْد":
                    b["lex"] = "رَأَى"
                add(b)
        # the lightened كَأَنْ (كَأَنْ لَمْ تَغْنَ بِالأَمْسِ): same particle, lemma كَأَنَّ
        for a in base:
            if bare_letters(a.get("lex", "")) == "كأن" and str(a.get("enc0", "0")) in ("0", "na") \
                    and re.search("كَأَن[\u064e\u0651]+$", a["diac"]):
                b = dict(a)
                b.update(diac=re.sub("كَأَن[\u064e\u0651]+$", "كَأَنْ", a["diac"]), source="classical_manual",
                         pos_lex_logprob=a.get("pos_lex_logprob", -99) - 0.5)
                add(b)
        if question and self.mode == "labels":
            for b in self.question(word):
                add(b)
        return out
