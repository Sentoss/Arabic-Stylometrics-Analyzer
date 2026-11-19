"""Fill gaps in الجدول's analyses from the book itself (internal use only).

Two passes over the converted records, in the book's order:

1. References. الجدول often writes "(قال) مثل مكر", "(لا) مثل الأولى",
   "(عليهم) مثل منهم ..." or ends a gloss with "ومثله" for the next word. The
   function is copied from the entry referred to: the latest earlier entry
   for that word (or for the same word, for الأول/الأولى/السابق/نظيرها).
2. Repeated phrases. Words الجدول skips (often with "...") get the function
   of the same word in the same context (previous, same and next word, fully
   vocalised) wherever الجدول did analyse it, if every such place agrees.

Filled words carry "src": "ref" or "phrase"; nothing here is invented.
"""
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import irab_rules as R  # noqa: E402
from jadwal_convert import function_of, DESCRIPTIVE  # noqa: E402

SELF = {"الاول", "الاولي", "الاوله", "السابق", "السابقه", "الاخير", "الاخيره", "الثاني", "الثانيه",
        "نظيرها", "نظيره", "المتقدم", "المتقدمه", "السابقين", "الاوليين", "الاولين"}
PART = re.compile(r"^\(([^()]*)\)\s*(.*)$", re.S)


def first_entry(r):
    m = PART.match(r.get("jadwal", "").split(" | ")[0])
    return (m.group(1), m.group(2)) if m else (None, None)


def fill(rows):
    stats = Counter()
    # ---- 1. references, in reading order
    last = {}                                   # norm(token as written in الجدول) -> (function, MASAQ role)
    for row in rows:
        recs = row["words"]
        carry = None                            # "ومثله": the next analysed word takes this function
        for r in recs:
            if "jadwal" not in r:
                continue
            tok, gloss = first_entry(r)
            if tok is None:
                continue
            g = R.norm(re.sub(r"[().،:؛]", " ", gloss))
            if carry and "function" not in r:
                r["function"], r["src"], r["ref"] = carry, "ref", "ومثله"
                stats["ref_wamithluh"] += 1
            carry = None
            m = re.match(r"(?:و)?مثل(?:ه|ها)?\s+(\S+)(?:\s+(\S+))?", g)
            single = len(R.norm(tok).split()) == 1
            if m and single:
                rest = g[m.end(1):]
                own = function_of(rest) if rest.strip() else None
                key = None
                if m.group(1) in SELF:
                    key = R.norm(tok)
                elif m.group(1) in last:
                    key = m.group(1)
                if "متعلق" in rest[:40] and "محذوف" not in rest[:40]:
                    if r.get("function") != "جار ومجرور":
                        r["function"], r["src"] = "جار ومجرور", "ref"
                        stats["ref_own"] += 1
                elif own and own not in DESCRIPTIVE and not (own == "معطوف" and key in last):
                    if r.get("function") != own:
                        r["function"], r["src"] = own, "ref"
                        stats["ref_own"] += 1
                elif key in last and last[key][1] == r["role"]:
                    if r.get("function") != last[key][0]:
                        r["function"] = last[key][0]
                        r["src"], r["ref"] = "ref", key
                        stats["ref_mithl"] += 1
                else:
                    stats["ref_unresolved"] += 1
            if re.search(r"(?:^|\s)و\s?مثل(?:ه|ها)\s*[.،]?$", g) and r.get("function"):
                carry = r["function"]
            if r.get("function") and single:
                last[R.norm(tok)] = (r["function"], r["role"])
    # ---- 2. repeated phrases: (previous word, word, next word), vocalised
    def ctx(recs, i):
        w = lambda k: recs[k]["word"] if 0 <= k < len(recs) else "#"
        return (w(i - 1), w(i), w(i + 1))
    seen = defaultdict(Counter)
    for row in rows:
        for i, r in enumerate(row["words"]):
            if r.get("function"):
                seen[ctx(row["words"], i)][r["function"]] += 1
    for row in rows:
        for i, r in enumerate(row["words"]):
            if r.get("function") or "covered_by" in r:
                continue
            c = seen.get(ctx(row["words"], i))
            if c and len(c) == 1:
                r["function"], r["src"] = next(iter(c)), "phrase"
                stats["phrase"] += 1
            elif c:
                stats["phrase_conflict"] += 1
    stats["remaining"] = sum(1 for row in rows for r in row["words"]
                             if not r.get("function") and "covered_by" not in r)
    return stats
