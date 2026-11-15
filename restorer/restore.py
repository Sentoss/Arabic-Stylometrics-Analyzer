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
from dataclasses import dataclass, asdict
from pathlib import Path

from .text import bare, split_marks, SHADDA
from .markup import clean, tokenize
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


class Restorer:
    """window: forms whose context score is within `window` of the best are
    treated as a tie and decided by Classical frequency (if a prior is given)."""

    def __init__(self, disambiguator=None, prior="default", window=0.15):
        if disambiguator is None:
            from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
            # MSA model only: dialect models are deliberately not used.
            disambiguator = BERTUnfactoredDisambiguator.pretrained(
                "msa", top=1000, use_gpu=False)
        self.dis = disambiguator
        if prior == "default":
            prior = FormPrior.load(DEFAULT_PRIOR) if DEFAULT_PRIOR.exists() else None
        self.prior = prior
        self.window = window

    def run(self, raw_text: str, overrides=None):
        text = clean(raw_text)
        tokens = tokenize(text)
        profile = detect(tokens, overrides)
        results = self._restore(tokens, profile)
        return text, profile, results

    # -- internals -------------------------------------------------------
    def _restore(self, tokens, profile):
        sents = {}
        for t in tokens:
            sents.setdefault(t.sent, []).append(t)
        order = sorted(sents)
        inputs = [[t.text.replace("ـ", "") for t in sents[k]] for k in order]
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
        written = bare(tok.text)
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
        changed = bare(restored) != written
        a = best[4]
        status = "restored" if changed else "exact"
        if a.get("source") == "backoff":
            status = "kept_as_name"
        if conf == "low":
            status = "ambiguous"
        return Result(restored=restored, changed=changed, status=status, confidence=conf,
                      conventions=sorted(set(best[3])),
                      alternatives=[f for f in forms if f != chosen][:3],
                      lemma=a.get("lex", ""), pos=a.get("pos", ""), diac=a.get("diac", ""),
                      root=a.get("root", ""), pattern=a.get("pattern", ""), **base)

    def _no_compatible(self, tok, written, profile, lexical, base):
        """The analyzer knows the word only in a form this source's conventions
        do not license (e.g. الأصفهانىّ known only as الإصفهاني). Apply only
        what the writing itself proves, and flag for review."""
        r = self._oov(tok, written, profile, base)
        r.status = "no_compatible"
        r.confidence = "low" if not r.changed else r.confidence
        r.alternatives = sorted({bare(s.analysis["diac"]) for s in lexical})[:5]
        return r

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
