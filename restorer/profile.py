"""Layer 1b: detect a source's printing conventions.

Each convention is measured on diagnostic words whose standard spelling
is unambiguous. A convention is switched on only if the source actually
uses it, so carefully spelled sources pass through untouched.
"""
from collections import Counter
from dataclasses import dataclass, field, asdict

from .markup import UTHMANI_MARKS
from .text import bare

# convention -> list of (standard form, non-standard variants)
DIAGNOSTICS = {
    # Egyptian print convention: final ي written ى
    "final_ya_as_alif_maqsura": [
        ("في", ["فى"]), ("الذي", ["الذى"]), ("التي", ["التى"]),
        ("أبي", ["أبى", "ابى"]), ("بني", ["بنى"]), ("النبي", ["النبى"]),
        ("هي", ["هى"]), ("فهي", ["فهى"]), ("يعني", ["يعنى"]),
        ("الراضي", ["الراضى"]), ("القاضي", ["القاضى"]), ("وفي", ["وفى"]),
        ("الأصفهاني", ["الأصفهانى", "الاصفهانى"]), ("نفسي", ["نفسى"]),
    ],
    # The reverse: final ى written ي
    "alif_maqsura_as_final_ya": [
        ("حتى", ["حتي"]), ("متى", ["متي"]), ("موسى", ["موسي"]),
        ("عيسى", ["عيسي"]), ("مصطفى", ["مصطفي"]), ("يحيى", ["يحيي"]),
        ("مستشفى", ["مستشفي"]), ("أولى", ["أولي"]),
    ],
    # Hamza on alif left out
    "hamza_dropped_on_alif": [
        ("إلى", ["الى"]), ("إذا", ["اذا"]), ("إنما", ["انما"]),
        ("أيضا", ["ايضا"]), ("أبو", ["ابو"]), ("أحمد", ["احمد"]),
        ("الأمير", ["الامير"]), ("أمير", ["امير"]), ("أن", ["ان"]),
        ("إن", ["ان"]), ("أو", ["او"]), ("إليه", ["اليه"]),
        ("الإسلام", ["الاسلام"]), ("أصحاب", ["اصحاب"]), ("الأرض", ["الارض"]),
    ],
    # Final ة written ه
    "ha_for_ta_marbuta": [
        ("الصلاة", ["الصلاه"]), ("الزكاة", ["الزكاه"]), ("القضاة", ["القضاه"]),
        ("المدينة", ["المدينه"]), ("الجنة", ["الجنه"]), ("الدولة", ["الدوله"]),
        ("الخلافة", ["الخلافه"]), ("قصيدة", ["قصيده"]), ("طويلة", ["طويله"]),
        ("سنة", ["سنه"]), ("الشرطة", ["الشرطه"]), ("زيادة", ["زياده"]),
    ],
    # Hamza seated on ya left out (قايل for قائل)
    "ya_for_hamza_seat": [
        ("قائل", ["قايل"]), ("مسائل", ["مسايل"]), ("رسائل", ["رسايل"]),
        ("فوائد", ["فوايد"]), ("عجائب", ["عجايب"]), ("سائر", ["ساير"]),
        ("جائز", ["جايز"]), ("الطائف", ["الطايف"]),
    ],
}

THRESHOLD = 0.15     # share of non-standard spellings that switches a convention on
MIN_EVIDENCE = 4     # diagnostic hits needed before we trust the ratio


@dataclass
class SourceProfile:
    conventions: dict = field(default_factory=dict)   # name -> bool
    evidence: dict = field(default_factory=dict)      # name -> (nonstd, total, ratio)
    diacritized_ratio: float = 0.0
    uthmani: bool = False
    notes: list = field(default_factory=list)

    def on(self, name):
        return self.conventions.get(name, False)

    def to_dict(self):
        return asdict(self)


def detect(tokens, overrides=None) -> SourceProfile:
    words = [t.text for t in tokens if t.kind == "ar"]
    counts = Counter(bare(w) for w in words)
    prof = SourceProfile()
    for name, pairs in DIAGNOSTICS.items():
        std = sum(counts[s] for s, _ in pairs)
        # a variant shared by two standard forms (ان) is counted once
        variants = {v for _, vs in pairs for v in vs}
        nonstd = sum(counts[v] for v in variants)
        total = std + nonstd
        ratio = nonstd / total if total else 0.0
        prof.evidence[name] = (nonstd, total, round(ratio, 3))
        if total >= MIN_EVIDENCE:
            prof.conventions[name] = ratio >= THRESHOLD
        else:
            prof.conventions[name] = nonstd >= 2   # sparse evidence: need repeated sightings
            if nonstd:
                prof.notes.append(f"{name}: little evidence ({nonstd}/{total})")
    marked = sum(1 for w in words if w != bare(w))
    prof.diacritized_ratio = round(marked / len(words), 3) if words else 0.0
    prof.uthmani = any(UTHMANI_MARKS.search(w) for w in words)
    if prof.uthmani:
        prof.notes.append("Uthmani marks found: convert to imla'i first; results unreliable")
    for k, v in (overrides or {}).items():
        prof.conventions[k] = v
        prof.notes.append(f"{k} forced to {v}")
    return prof
