"""Remove what modern editors added to Classical texts, keep what they only marked.

Classical manuscripts have almost no punctuation; brackets and quotation
marks in printed editions come from 20th-century editors, each with their own
system (see the README). This module:

REMOVES (not the author's words):
  - metre labels           [من الخفيف] [وافر] "من الطويل" [البحر الكامل]
  - Qur'an references      [النساء: 163] (البقرة: 255) [ص: 22]
  - bare numbers / sigla   {145} [12] «خ» (ط)
  - editor's glosses       (الجناجن: عظام الصدر)   -- headword: explanation
  - item numbering         "16 - حدثنا ..." at paragraph start, stray # and *

KEEPS the words but rewrites the marking:
  - { ... } and ' ... '    -> ﴿ ... ﴾   Qur'an hint (layer 1 flags these spans)
  - « » " " ( )            -> « ... »   quotation hint (proverb, hadith, speech,
                                       name or title; decided later by
                                       reference matching)
  - [ ... ]                -> ...       editor's restored words; brackets dropped

Every removal is counted so its effect can be audited per book.
"""
import re
from collections import Counter

AR = "ء-يً-ْٰ"
DIG = "0-9٠-٩"
METRES = ("طويل|مديد|بسيط|وافر|كامل|هزج|رجز|رمل|سريع|منسرح|خفيف|مضارع|"
          "مقتضب|مجتث|متقارب|متدارك|مشطور|منهوك")
METRE_LABEL = re.compile(
    r"[\[\(\"«]\s*(?:من\s+)?(?:البحر\s+)?(?:ال)?(?:%s)(?:\s+[%s]+){0,2}\s*[\]\)\"»]" % (METRES, AR))
QURAN_REF = re.compile(
    r"[\[\(\{]\s*(?:سورة\s+)?[%s]+(?:\s+[%s]+){0,2}\s*:\s*[%s]+(?:\s*[-–،,]\s*[%s]+)*\s*[\]\)\}]"
    % (AR, AR, DIG, DIG))
NUMBERS = re.compile(r"[\[\(\{«]\s*[%s\s\-–/]+\s*[\]\)\}»]" % DIG)
SIGLA = re.compile(r"[\(«]\s*[ء-ي]\s*[\)»]")   # «خ» (ط); not [و], a restored word
SPEECH = {"قال", "قالت", "فقال", "فقالت", "يقول", "تقول", "قلت", "فقلت", "قالوا",
          "فقالوا", "أي", "يعني", "تعني", "يريد", "وقال", "قيل", "وقيل"}
GLOSS = re.compile(r"\(\s*([%s]+(?:\s+[%s]+)?)\s*:\s*([^()]{1,60})\)" % (AR, AR))
NUMBERING = re.compile(r"^\s*[#*]?\s*[%s]+\s*[-–]\s*" % DIG)
STRAY = re.compile(r"(?<!\S)[#*]+(?!\S)")

CURLY = re.compile(r"\{([^{}]{2,600})\}")
SINGLE = re.compile(r"(?<![%s])'([^']{2,600})'(?![%s])" % (AR, AR))
GUILLEMET = re.compile(r"«([^«»]{1,600})»")
DOUBLE = re.compile(r"\"([^\"]{1,600})\"")
PAREN = re.compile(r"\(([^()]{1,600})\)")
SQUARE = re.compile(r"\[([^\[\]]{1,200})\]")
SPACES = re.compile(r"[ \t]+")


def _gloss(m, stats):
    head, expl = m.group(1), m.group(2)
    if head.split()[0] in SPEECH or len(expl.split()) > 8:
        return m.group(0)                     # the author's own voice, keep
    stats["gloss"] += 1
    return " "


def clean_editorial(text, stats=None):
    stats = stats if stats is not None else Counter()

    def sub(pat, name, s, repl=" "):
        new, n = pat.subn(repl, s)
        stats[name] += n
        return new

    t = sub(NUMBERING, "numbering", text, "")
    t = sub(METRE_LABEL, "metre_label", t)
    t = sub(QURAN_REF, "quran_ref", t)
    t = sub(NUMBERS, "number", t)
    t = sub(SIGLA, "siglum", t)
    t = GLOSS.sub(lambda m: _gloss(m, stats), t)
    # Qur'an hints first, so the generic quote rules don't swallow them
    t = sub(CURLY, "quran_hint", t, r"﴿\1﴾")
    t = sub(SINGLE, "quran_or_quote_hint", t, r"﴿\1﴾")
    t = sub(DOUBLE, "quote_hint", t, r"«\1»")
    t = sub(PAREN, "quote_hint", t, r"«\1»")
    t = sub(SQUARE, "restored", t, r"\1")
    t = sub(STRAY, "stray_symbol", t)
    t = SPACES.sub(" ", t).strip()
    stats["quote_hint"] += len(GUILLEMET.findall(text))  # already in « »
    return t, stats
