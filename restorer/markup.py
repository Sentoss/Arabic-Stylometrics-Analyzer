"""Layer 1a: strip source markup, tokenize, and flag embedded Qur'an quotations.

Shamela/OpenITI pages carry apparatus that is not the author's text:
footnote markers [١] (٢) «١٥٩», page markers [ص ٥٩], editorial metre
labels [من الرجز], honorific ligatures, verse-hemistich separators.
We remove the apparatus but keep every token's character offsets into
the *cleaned* text so the three output tiers stay aligned.
"""
import re
from dataclasses import dataclass, field

AR_DIGITS = "٠١٢٣٤٥٦٧٨٩0123456789"

# Editorial apparatus patterns (removed before analysis).
_APPARATUS = [
    re.compile(r"\[\s*ص\s*[%s]+\s*\]" % AR_DIGITS),          # [ص ٥٩] page marker
    re.compile(r"\[\s*[%s]+\s*\]" % AR_DIGITS),               # [١] footnote ref
    re.compile(r"\(\s*[%s]+\s*\)" % AR_DIGITS),               # (١) footnote ref
    re.compile(r"«\s*[%s]+\s*»\s*-?" % AR_DIGITS),            # «١٥٩» - item number
    re.compile(r"^\s*[%s]+\s*-\s*" % AR_DIGITS, re.M),        # ١٧٥٧٦ - line number
    re.compile(r"\[\s*(?:من\s+)?(?:ال)?(?:رجز|طويل|بسيط|كامل|وافر|خفيف|"
               r"سريع|منسرح|متقارب|متدارك|مديد|هزج|رمل|مضارع|مقتضب|مجتث)"
               r"[^\]]*\]"),                                   # [من الرجز] metre label
]

# Honorific ligatures (U+FD40–FDFF block, e.g. ﷺ ﵀ ﵇): keep as symbols.
_LIGATURE = re.compile(r"[﴾-﷿ﹰ-﻿]")

AR_LETTERS = "ء-غف-يٱٲ-ۓ"
AR_MARKS = "ً-ْٰٓ-ٕ"       # harakat, tanwin, shadda, sukun, dagger alif, madda/hamza marks
TATWEEL = "ـ"

_TOKEN = re.compile(
    r"(?P<ar>[%s][%s%s%s]*)|(?P<num>[%s]+)|(?P<other>\S)"
    % (AR_LETTERS, AR_LETTERS, AR_MARKS, TATWEEL, AR_DIGITS)
)

# Uthmani-only marks: if present the text needs conversion, not restoration.
UTHMANI_MARKS = re.compile(r"[ٱۖ-ࣰۭ-ࣳ]")


@dataclass
class Token:
    text: str                 # as it appears in the cleaned text (with diacritics)
    start: int
    end: int
    kind: str                 # 'ar', 'num', 'other'
    sent: int = 0
    quran: str = ""           # '', 'bracketed' (inside ﴿﴾) or 'probable' (after تعالى/قوله)
    notes: list = field(default_factory=list)


def clean(raw: str) -> str:
    text = raw
    for pat in _APPARATUS:
        text = pat.sub(" ", text)
    text = text.replace("…", " ... ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


_SENT_END = set(".!?؟:؛\n")


def tokenize(text: str, max_len: int = 60):
    """Tokenize cleaned text; assign sentence ids (split on punctuation,
    newlines, hemistich separators, and every max_len words for BERT)."""
    tokens, sent, n_in_sent = [], 0, 0
    for m in _TOKEN.finditer(text):
        kind = m.lastgroup
        tok = Token(m.group(0), m.start(), m.end(), kind, sent)
        tokens.append(tok)
        if kind == "ar":
            n_in_sent += 1
        # newline between this token and the next counts as a sentence end
        nxt = text[m.end(): m.end() + 2]
        if (kind == "other" and tok.text in _SENT_END) or "\n" in nxt or n_in_sent >= max_len:
            sent += 1
            n_in_sent = 0
    _flag_quran(text, tokens)
    return tokens


_PROBABLE_Q = re.compile(r"(?:تعالى|تعالي|عز وجل|سبحانه|قوله)\s*[:：]?\s*$")


def _flag_quran(text, tokens):
    # Spans inside ornate brackets ﴿ ... ﴾ are certainly Qur'an.
    spans = [(m.start(), m.end(), "bracketed") for m in re.finditer(r"﴿[^﴾]*﴾", text)]
    # Parenthesised spans right after "قال تعالى:" etc. are probably Qur'an.
    for m in re.finditer(r"\(([^()]{3,400})\)", text):
        before = text[max(0, m.start() - 25): m.start()]
        if _PROBABLE_Q.search(before):
            spans.append((m.start(), m.end(), "probable"))
    for t in tokens:
        for s, e, kind in spans:
            if s <= t.start and t.end <= e:
                t.quran = kind
                break
