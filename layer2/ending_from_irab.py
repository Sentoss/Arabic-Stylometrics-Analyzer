"""The word's ending from layer 2b: 2a chooses the word, 2b's `end` head says which
final vowel it is printed with (learned by prediction from vowelled prose, Tashkeela).

For each word:
  1. 2a's reading (chosen ignoring case and mood) is kept;
  2. if 2a's form already ends in the predicted vowel, nothing changes;
  3. else, if the dictionary has a form of the same reading with that ending, it is used;
  4. else the predicted vowel is written on 2a's form (not on a final long vowel, no
     sukun on a doubled letter, tanwīn al-fatḥ as ـاً except after ة/ء).
Nothing is invented: the word is 2a's, the vowel is the model's prediction.

    from layer2.ending_from_irab import IrabEndings
    dis = IrabEndings(ClassicalDisambiguator(MORPH_DIR), IrabTagger(IRAB_DIR))
    Restorer(disambiguator=dis)        # or Restorer(irab_model=IRAB_DIR)

Each result keeps a record in dis.log: (word, predicted ending, "kept"/"dictionary"/"written").
Replaces case_from_irab.py (2b's case head picked endings from the dictionary only and,
trained on the Qur'an alone, did not carry over to prose).
"""
import re

from layer2.case_from_irab import SAME

MARK = {"ضمة": "ُ", "فتحة": "َ", "كسرة": "ِ", "تنوين ضم": "ٌ",
        "تنوين فتح": "ً", "تنوين كسر": "ٍ", "سكون": "ْ"}
NAME = {v: k for k, v in MARK.items()}
FINAL_MARKS = re.compile("[ً-ِْ]+$")
VOWELS = re.compile("[ً-ْٰـ]")


def ending(word):
    """The final short vowel of a vowelled word (as in layer2b/make_dataset.py), or None."""
    w = word.replace("ـ", "")
    units = re.findall("([ء-ي])([ً-ْٰ]*)", w)
    if not units:
        return None
    marks = lambda m: [NAME[c] for c in m if c in NAME]
    last, lm = units[-1]
    prev = marks(units[-2][1]) if len(units) > 1 else []
    lm = marks(lm)
    if last in "اى":
        if "تنوين فتح" in lm or "تنوين فتح" in prev:
            return "تنوين فتح"
        if lm:
            return lm[-1]
        return "none" if prev else None
    if not lm:
        if (last == "و" and "ضمة" in prev) or (last == "ي" and "كسرة" in prev):
            return "none"
        return None
    return lm[-1]


def write_end(diac, end):
    """2a's form with the predicted final vowel, or None where the last letter takes none."""
    base = FINAL_MARKS.sub("", diac)
    if not base or end not in MARK or (base[-1] in "اوىي" and not base.endswith("ءا")):
        return None
    if end == "سكون" and base.endswith("ّ"):
        return None
    if end == "تنوين فتح" and base[-1] not in "ةء":
        return base + "اً"
    return base + MARK[end]


class IrabEndings:
    """A drop-in disambiguator: 2a's choice with the ending 2b predicts."""

    def __init__(self, disambiguator, tagger):
        self.d, self.t = disambiguator, tagger
        self.d.ignore = ("cas", "mod")      # the ending comes from 2b
        self.d.top = max(getattr(self.d, "top", 0), 100000)
        self.log = []

    def __getattr__(self, k):              # _analyzer, freq … as the wrapped disambiguator
        return getattr(self.d, k)

    def disambiguate(self, sentence):
        from camel_tools.disambig.common import DisambiguatedWord
        res = self.d.disambiguate(sentence)
        tags = self.t.tag([VOWELS.sub("", w) for w in sentence])
        out = []
        for dw, x in zip(res, tags):
            end = x.get("end", "none") if "role" in x else None
            how = "kept"
            if dw.analyses and end and end != "none":
                first = dw.analyses[0]
                top = first.analysis
                if ending(top.get("diac", "")) != end:
                    same = [s for s in dw.analyses[1:]
                            if all(s.analysis.get(f) == top.get(f) for f in SAME)
                            and ending(s.analysis.get("diac", "")) == end]
                    if same:
                        # 2a's score, so callers that re-sort by score keep it first
                        pick = same[0]._replace(score=first.score)
                        dw = DisambiguatedWord(dw.word, [pick] + [s for s in dw.analyses if s is not same[0]])
                        how = "dictionary"
                    else:
                        new = write_end(top.get("diac", ""), end)
                        if new:
                            a = dict(top, diac=new)
                            dw = DisambiguatedWord(dw.word, [first._replace(analysis=a, diac=new)] + list(dw.analyses[1:]))
                            how = "written"
            self.log.append((dw.word, end, how))
            out.append(dw)
        return out
