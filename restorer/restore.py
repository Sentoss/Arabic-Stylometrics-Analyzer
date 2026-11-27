"""Layer 1c: restore standard spelling without erasing real distinctions.

For each word:
  1. The analyzer proposes readings (it is lenient: على -> عَلَى and عَلِيّ).
  2. A reading is *compatible* only if it differs from the written word
     by substitutions the source's profile allows (e.g. final ى->ي only in
     sources that print ي as ى). In a carefully spelled source only exact
     matches survive, so على stays على.
  3. Among compatible readings, the context model (CAMeL's BERT
     disambiguator) picks one; editor-supplied diacritics, if any, act as
     a constraint.
  4. The chosen reading's letters are written back onto the original word
     (its diacritics are kept), giving the restored form.
"""
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from .text import bare, split_marks, SHADDA
from .markup import clean, tokenize, Token
from .profile import detect
from .prior import FormPrior

HAMZA_ALIFS = "أإآ"

# (written, standard, final_only) -> convention that licenses it
SUBSTITUTIONS = [
    ("ا", HAMZA_ALIFS, False, "hamza_dropped_on_alif"),
    ("ى", "ي", True, "final_ya_as_alif_maqsura"),
    ("ي", "ى", True, "alif_maqsura_as_final_ya"),
    ("ه", "ة", True, "ha_for_ta_marbuta"),
    ("ي", "ئ", False, "ya_for_hamza_seat"),
]


def compat(written: str, standard: str, profile):
    """Return (licensed?, list of conventions used)."""
    if len(written) != len(standard):
        return False, []
    used = []
    last = len(written) - 1
    for i, (w, s) in enumerate(zip(written, standard)):
        if w == s:
            continue
        for src, tgt, final_only, conv in SUBSTITUTIONS:
            if w == src and s in tgt and (i == last or not final_only) and profile.on(conv):
                used.append(conv)
                break
        else:
            return False, []
    return True, used


def diac_mismatches(written_word: str, analysis_diac: str) -> int:
    """How many editor-supplied marks the reading contradicts (case ending ignored)."""
    w, a = split_marks(written_word), split_marks(analysis_diac)
    if len(w) != len(a):
        return 0
    bad = 0
    for i, ((_, wm), (_, am)) in enumerate(zip(w, a)):
        if i == len(w) - 1:
            wm = wm & {SHADDA}           # only shadda is reliable on the last letter
        bad += len(wm - am - {"ْ"})  # editors add sukun the DB often omits
    return bad


# Old prints seat a final hamza under the alif to show the genitive
# (المبتدإ، الخطإ، المتوضإ). Analyse it as أ, but keep the writing.
FINAL_HAMZA_BELOW = re.compile(r"([\u0621-\u064A][\u064B-\u0652]*)إ([\u064B-\u0652]*)$")


def for_analysis(word: str) -> str:
    return FINAL_HAMZA_BELOW.sub(r"\1أ\2", word)


# Short words that a missing space most often glues to the word before
GLUED = {"من", "ما", "في", "عن", "لا", "لم", "لن", "قد", "أن", "إن", "أو", "ثم", "هو", "هي",
         "كل", "على", "إلى", "مع", "بن", "بعد", "قبل", "الله"}


def apply_letters(written_word: str, standard: str) -> str:
    """Write the standard letters onto the written word, keeping its marks."""
    out, i = [], 0
    for ch in written_word:
        if ch == "ـ":
            continue
        if bare(ch) == "":           # a mark
            out.append(ch)
        else:
            out.append(standard[i] if i < len(standard) else ch)
            i += 1
    return "".join(out)


@dataclass
class Result:
    raw: str
    restored: str
    changed: bool
    status: str          # exact | restored | ambiguous | oov | no_compatible | not_word
    confidence: str      # certain | high | low | ''
    conventions: list
    alternatives: list   # other restored forms that were possible
    lemma: str = ""
    pos: str = ""
    diac: str = ""
    root: str = ""
    pattern: str = ""
    quran: str = ""
    start: int = 0
    end: int = 0
    sent: int = 0

    def to_dict(self):
        return asdict(self)


DEFAULT_PRIOR = Path(__file__).resolve().parent.parent / "resources" / "form_freq_tashkeela.tsv"


import os
DEFAULT_MORPH = Path(os.environ.get("MORPH_MODEL", Path(__file__).resolve().parent.parent / "models" / "morph_model"))


class Restorer:
    """window: forms whose context score is within `window` of the best are
    treated as a tie and decided by Classical frequency (if a prior is given)."""

    def __init__(self, disambiguator=None, prior="default", window=0.15, classical=True,
                 morph_model="default", irab_model=None):
        """classical: give the disambiguator the layer 2 Classical supplement
        (passives, imperatives, interrogative أَ, Classical vowelings, Form II
        verbs such as حَدَّثَ, the lightened كَأَنْ, the lemma of رَأَى).
        irab_model: a layer 2b model folder with the `end` head; the word's final
        vowel is then the one 2b predicts (layer2/ending_from_irab.py)."""
        if morph_model == "default":
            morph_model = DEFAULT_MORPH if (DEFAULT_MORPH / "model.pt").exists() else None
        if disambiguator is None and morph_model and classical:
            # layer 2a: the Classical tagger (fine-tuned CAMeLBERT-CA)
            from layer2.disambiguator import ClassicalDisambiguator
            disambiguator = ClassicalDisambiguator(morph_model)
        if disambiguator is None:
            from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
            # MSA model only: dialect models are deliberately not used.
            disambiguator = BERTUnfactoredDisambiguator.pretrained(
                "msa", top=1000, use_gpu=False)
        if classical:
            from layer2.supplement import ClassicalAnalyzer
            if not isinstance(disambiguator._analyzer, ClassicalAnalyzer):
                disambiguator._analyzer = ClassicalAnalyzer(analyzer=disambiguator._analyzer,
                                                            keep_backoff=True, mode="text")
        if irab_model:
            from layer2.ending_from_irab import IrabEndings
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "layer2b"))
            from irab import IrabTagger
            disambiguator = IrabEndings(disambiguator, IrabTagger(irab_model))
        self.dis = disambiguator
        if prior == "default":
            prior = FormPrior.load(DEFAULT_PRIOR) if DEFAULT_PRIOR.exists() else None
        self.prior = prior
        self.window = window

    def run(self, raw_text: str, overrides=None):
        text = clean(raw_text)
        tokens = self._split_glued(tokenize(text))
        profile = detect(tokens, overrides)
        results = self._restore(tokens, profile)
        return text, profile, results

    # -- internals -------------------------------------------------------
    def _lexical(self, word):
        an = self.dis._analyzer
        return [a for a in an.analyze(word) if a.get("source") != "backoff"]

    def _split_glued(self, tokens):
        """A word the analyzer does not know, which is two known words with the
        space between them lost (بآخرمن = بآخر من), is split in two. One part must
        be a common short word (من، ما، في …) and the other a known word.
        Vowelled words are left alone: the editor wrote them deliberately."""
        out = []
        for t in tokens:
            w = bare(t.text)
            if t.kind != "ar" or len(w) < 5 or t.text != w or self._lexical(for_analysis(w)):
                out.append(t)
                continue
            best = None
            for i in range(2, len(w) - 1):
                left, right = w[:i], w[i:]
                if right in GLUED and len(left) >= 3 and self._lexical(left) and self._lexical(right):
                    if best is None or len(right) > len(w) - best[1]:
                        best = (0, i)
            if best is None:
                out.append(t)
                continue
            i, k, cut = best[1], 0, len(t.text)
            for j, ch in enumerate(t.text):               # position after the i-th letter, marks included
                if bare(ch):
                    k += 1
                    if k == i:
                        cut = j + 1
                        while cut < len(t.text) and not bare(t.text[cut]) and t.text[cut] != "ـ":
                            cut += 1
                        break
            a = Token(t.text[:cut], t.start, t.start + cut, t.kind, t.sent, t.quran, list(t.notes) + ["missing_space"])
            b = Token(t.text[cut:], t.start + cut, t.end, t.kind, t.sent, t.quran, list(t.notes) + ["missing_space"])
            out += [a, b]
        return out

    def _restore(self, tokens, profile):
        sents = {}
        for t in tokens:
            sents.setdefault(t.sent, []).append(t)
        order = sorted(sents)
        inputs = [[for_analysis(t.text.replace("ـ", "")) for t in sents[k]] for k in order]
        # one call per sentence: camel-tools 1.6 batching breaks on mixed lengths
        disamb = [self.dis.disambiguate(s) for s in inputs]
        results = []
        for k, dws in zip(order, disamb):
            for tok, dw in zip(sents[k], dws):
                results.append(self._decide(tok, dw, profile))
        return results

    def _decide(self, tok, dw, profile):
        base = dict(raw=tok.text, quran=tok.quran, start=tok.start, end=tok.end, sent=tok.sent)
        if tok.kind != "ar":
            return Result(restored=tok.text, changed=False, status="not_word",
                          confidence="", conventions=[], alternatives=[], **base)
        written = for_analysis(bare(tok.text))
        hamza_below = written != bare(tok.text)
        scored = list(dw.analyses)
        lexical = [s for s in scored if s.analysis.get("source") != "backoff"]
        if not lexical:
            return self._oov(tok, written, profile, base)

        cands = []
        for s in lexical:
            std = bare(s.analysis["diac"])
            ok, used = compat(written, std, profile)
            if ok:
                mm = diac_mismatches(tok.text, s.analysis["diac"])
                cands.append((s.score - 0.5 * mm, len(used), std, used, s.analysis))
        if not cands:
            return self._no_compatible(tok, written, profile, lexical, base)

        # best reading per restored form (context score; ties -> fewer substitutions)
        cands.sort(key=lambda c: (-c[0], c[1]))
        best_of = {}
        for c in cands:
            best_of.setdefault(c[2], c)

        # A proper-noun guess for the word *as written* (analyzer backoff) wins
        # only if context prefers it outright: keeps واسط from becoming وأسط.
        # It also needs corpus evidence: the written form occurs in clean
        # Classical text and the dictionary alternatives do not.
        backoff = [s for s in scored if s.analysis.get("source") == "backoff"]
        if backoff and written not in best_of and backoff[0].score > cands[0][0] \
                and self.prior is not None and self.prior(written) > 0 \
                and all(self.prior(f) == 0 for f in best_of):
            b = backoff[0]
            best_of[written] = (b.score, 0, written, [], b.analysis)

        forms = sorted(best_of, key=lambda f: (-best_of[f][0], best_of[f][1]))
        top = best_of[forms[0]][0]
        tied = [f for f in forms if best_of[f][0] >= top - self.window]
        if len(forms) == 1:
            chosen, conf = forms[0], "certain"
        elif len(tied) == 1:
            chosen, conf = forms[0], "high"
        else:
            # Classical frequency decides a tie only when it is decisive
            # (>= 5 occurrences and >= 5x the runner-up); otherwise keep the
            # context model's order and flag the word.
            chosen, conf = forms[0], "low"
            if self.prior is not None:
                by_freq = sorted(tied, key=lambda f: -self.prior(f))
                f1, f2 = self.prior(by_freq[0]), self.prior(by_freq[1])
                if f1 >= 5 and f1 >= 5 * max(f2, 1):
                    chosen, conf = by_freq[0], "high"

        best = best_of[chosen]
        restored = apply_letters(tok.text, best[2])
        if hamza_below:
            restored = re.sub(r"أ([\u064B-\u0652]*)$", r"إ\1", restored)
        changed = bare(restored) != bare(tok.text)
        a = best[4]
        status = "restored" if changed else "exact"
        if a.get("source") == "backoff":
            status = "kept_as_name"
        if conf == "low":
            status = "ambiguous"
        convs = set(best[3]) | (set(tok.notes) & {"missing_space"}) | ({"final_hamza_below"} if hamza_below else set())
        if "missing_space" in tok.notes and status == "exact":
            status, changed = "restored", True
        return Result(restored=restored, changed=changed, status=status, confidence=conf,
                      conventions=sorted(convs),
                      alternatives=[f for f in forms if f != chosen][:3],
                      lemma=a.get("lex", ""), pos=a.get("pos", ""), diac=a.get("diac", ""),
                      root=a.get("root", ""), pattern=a.get("pattern", ""), **base)

    def _no_compatible(self, tok, written, profile, lexical, base):
        """The analyzer knows the word only in a form this source's conventions
        do not license (e.g. الأصفهانىّ known only as الإصفهاني). Apply only
        what the writing itself proves, and flag for review.

        One exception: a word opening (after و/ف) with a bare alif that every
        reading spells with a hamza (واعرضه read only as وَأَعْرَضَهُ, الق only
        as أَلْقَ) is an imperative with hamzat al-wasl (وَاعْرِضْهُ، اِلْقَ)
        when the verb has one: the writing itself then fits."""
        imp = self._imperative(tok, written)
        if imp is not None:
            return Result(restored=tok.text.replace("ـ", ""), changed=False, status="exact",
                          confidence="high", conventions=["imperative_wasl"], alternatives=[],
                          lemma=imp.get("lex", ""), pos=imp.get("pos", ""), diac=imp.get("diac", ""),
                          root=imp.get("root", ""), pattern=imp.get("pattern", ""), **base)
        r = self._oov(tok, written, profile, base)
        r.status = "no_compatible"
        r.confidence = "low" if not r.changed else r.confidence
        r.alternatives = sorted({bare(s.analysis["diac"]) for s in lexical})[:5]
        return r

    def _imperative(self, tok, written):
        rest = written[1:] if len(written) > 2 and written[0] in "وف" else written
        an = self.dis._analyzer
        if not rest.startswith("ا") or not hasattr(an, "imperatives"):
            return None
        k = len(written) - len(rest)
        for a in an.imperatives(written):
            d = bare(a["diac"]).replace("ٱ", "ا")
            if d == written and d[k] == "ا":
                return a
        return None

    def _oov(self, tok, written, profile, base):
        """Unknown to the analyzer: only change what is certain from the
        writing itself (ى cannot carry shadda, so ىّ must be يّ)."""
        restored = tok.text.replace("ـ", "")
        conv = []
        letters = split_marks(restored)
        if letters and letters[-1][0] == "ى" and SHADDA in letters[-1][1] \
                and profile.on("final_ya_as_alif_maqsura"):
            restored = apply_letters(restored, written[:-1] + "ي")
            conv.append("final_ya_as_alif_maqsura")
        changed = bare(restored) != written
        return Result(restored=restored, changed=changed, status="oov",
                      confidence="certain" if changed else "", conventions=conv,
                      alternatives=[], **base)
