"""Case and mood from layer 2b, written by layer 2a (the agreed division of labour).

Layer 2a chooses each word's reading ignoring case and mood (ClassicalDisambiguator
with ignore = ("cas", "mod")). Layer 2b's cm head then says which case or mood the
word carries (رفع، نصب، جر، مضارع مرفوع/منصوب/مجزوم, or none). This module keeps
2a's reading but swaps in the analysis of the same reading that has 2b's case or
mood, so the ending is written the way 2a's dictionary writes it for that word
(diptotes, duals and sound plurals, مقصور/منقوص, the five verbs …).

"The same reading" = an analysis that differs from 2a's choice only in case, mood
and the vowelling. If 2a has no such analysis with 2b's case (or 2b says none,
or 2b's answer is unsure), 2a's own choice stands.

    from layer2.case_from_irab import apply_case
    words = disambiguator.disambiguate(sentence)        # DisambiguatedWords
    cms = [r.get("cm") for r in irab_tagger.tag(bare_words)]
    chosen = apply_case(words, cms)                     # one analysis (dict) per word
"""

CAS = {"رفع": "n", "نصب": "a", "جر": "g"}
MOD = {"مضارع مرفوع": "i", "مضارع منصوب": "s", "مضارع مجزوم": "j"}
SAME = ("lex", "pos", "prc3", "prc2", "prc1", "prc0", "per", "asp", "vox", "gen", "num", "enc0")


def wanted(cm):
    """2b's label -> (feature, value) that 2a's analysis should have, or None."""
    if cm in CAS:
        return "cas", CAS[cm]
    if cm in MOD:
        return "mod", MOD[cm]
    return None


def pick(analyses, cm):
    """analyses: 2a's ScoredAnalysis list, best first. -> (analysis dict, changed?)."""
    if not analyses:
        return None, False
    top = analyses[0].analysis
    w = wanted(cm)
    if w is None:
        return top, False
    f, v = w
    if top.get(f) == v:
        return top, False
    for sa in analyses[1:]:
        a = sa.analysis
        if a.get(f) == v and all(a.get(k) == top.get(k) for k in SAME):
            return a, True
    return top, False


def apply_case(words, cms):
    """words: 2a DisambiguatedWords; cms: 2b cm labels (None where 2b has none)."""
    return [pick(dw.analyses, cm)[0] for dw, cm in zip(words, cms)]
