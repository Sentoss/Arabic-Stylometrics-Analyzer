"""Convert الجدول في إعراب القرآن into naḥw fields aligned with MASAQ (layer 2b).

For every word of a verse block (MASAQ's words), the output gives:

  role        MASAQ's i'rab role of the word's stem (59 roles; CC BY 4.0)
  jadwal      الجدول's gloss for the word (internal use only)
  function    the naḥw function named first in الجدول's gloss (فاعل، حال، ...)
  amil        العامل, where it follows from the analysis itself: the مضاف of a
              مضاف إليه (the word before it)
  mutaallaq   المتعلَّق به of a جار ومجرور / ظرف: a word index, or
              {"محذوف": خبر|صفة|حال|صلة|?}
  matbu       المتبوع of a تابع: {"type": نعت|عطف|بدل|توكيد, "word": index}
  sahib_hal   صاحب الحال: a word index (the word that is, or contains, it)
  muqaddar    for verbs: {"مستتر": true/false}
  flags       disagreements between الجدول's function and MASAQ's role

الجدول is a modern work: this file and its output are for internal training
and evaluation only and are not redistributed.

usage: python layer2b/jadwal_convert.py PAGES_DIR SURAS_JSON MASAQ.csv OUT.jsonl.gz [--suras 2,12,78]
"""
import argparse
import gzip
import html
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import irab_rules as R  # noqa: E402

FUNCTIONS = [  # longest first; matched on the normalised gloss
    "مفعول مطلق", "مفعول لاجله", "مفعول لأجله", "مفعول معه", "مفعول فيه", "مفعول به ثان", "مفعول به اول",
    "مفعول به", "نائب الفاعل", "نائب فاعل", "فاعل", "مبتدا مؤخر", "مبتدا", "خبر مقدم", "خبر ثان", "خبر ان",
    "خبر كان", "خبر لا", "خبر ما", "خبر", "اسم ان", "اسم كان", "اسم لا", "اسم ما", "مضاف اليه",
    "نعت", "صفه", "معطوف", "بدل", "عطف بيان", "توكيد", "حال", "تمييز", "منادي", "مستثني", "ظرف زمان",
    "ظرف مكان", "جار ومجرور", "حرف جر", "فعل ماض", "فعل مضارع", "مضارع", "فعل امر", "امر", "اسم موصول",
    "موصول", "اسم استفهام", "اسم اشاره", "ضمير", "حرف",
]
ROLE_AR = {
    "AGNT": "فاعل", "PASS_SUBJ": "نائب فاعل", "OBJ": "مفعول به", "SUBJ": "مبتدا", "PRED": "خبر",
    "SUBJ_COP_V": "اسم كان", "V_COP_PRED": "خبر كان", "SUBJ_COP_PART": "اسم ان", "PART_COP_PRED": "خبر ان",
    "GEN_CONS": "مضاف اليه", "ADJ": "نعت", "CONJ_N": "معطوف", "APPOS": "بدل", "CIRCUM": "حال",
    "ACC_SPECIF": "تمييز", "COGN": "مفعول مطلق", "PURP": "مفعول لاجله", "COMIT": "مفعول معه", "VOC": "منادي",
    "EXCP": "مستثني", "ADV_TIME": "ظرف زمان", "ADV_PLCE": "ظرف مكان", "SUBJ_NEG_CAT": "اسم لا",
}
EQUIV = {"صفه": "نعت", "نائب الفاعل": "نائب فاعل", "مبتدا مؤخر": "مبتدا", "خبر مقدم": "خبر",
         "خبر ثان": "خبر", "مفعول لأجله": "مفعول لاجله", "مفعول به ثان": "مفعول به", "مفعول به اول": "مفعول به",
         "عطف بيان": "بدل"}


DESCRIPTIVE = {"ضمير", "اسم موصول", "موصول", "حرف", "اسم اشاره", "اسم استفهام", "حرف جر", "جار ومجرور"}


def function_of(gloss):
    """The first naḥw function in the gloss; word-class terms (ضمير، اسم موصول،
    حرف...) only when no function follows (ضمير ... مبتدأ → مبتدأ)."""
    g = R.norm(gloss.split("،")[0] if "متعلق" not in R.norm(gloss.split("،")[0]) else gloss)
    g = R.norm(gloss)[:160]
    found = []
    for f in FUNCTIONS:
        k = g.find(R.norm(f))
        if k >= 0:
            found.append((k, -len(f), f))
    if not found:
        return None
    found.sort()
    real = [f for _, _, f in found if f not in DESCRIPTIVE]
    return real[0] if real else found[0][2]


HAL = re.compile(r"حال(?:\s+(?!من[\s(])\S+){0,2}\s+من(?=[\s(])\s*(?:فاعل\s+|الضمير\s+في\s+|ضمير\s+)?\(?([^\s(),.]+)")
STEM_ROLE = lambda w: w["role"] or ("VERB" if w["verb"] else "NONE")


def convert_block(nums, body, ws):
    recs = [{"word": w["text"], "verse": w["verse"], "n": w["n"], "role": STEM_ROLE(w)} for w in ws]
    for i, w in enumerate(ws):
        if w["role"] == "GEN_CONS" and i > 0:
            recs[i]["amil"] = i - 1
    cursor, last, pending = 0, None, None
    ents = R.entries(body)
    for ei, (tok, gloss) in enumerate(ents):
        letter = bool(R.LETTER.match(tok.strip()))
        toks = R.norm(tok).split()
        if not toks:
            continue
        LETTER_CH = {"اللام": "ل", "الباء": "ب", "الكاف": "ك", "الواو": "و", "الفاء": "ف", "السين": "س",
                     "التاء": "ت", "الهاء": "ه", "الياء": "ي", "الهمزة": "ا", "النون": "ن", "الميم": "م"}
        if letter and pending is not None and tok.strip() in ("الكاف", "الهاء", "الياء") \
                and ws[pending]["norm"][1:].startswith(LETTER_CH[tok.strip()]):
            toks = [LETTER_CH[tok.strip()]]
            letter = False
        if letter:
            ch = LETTER_CH.get(tok.strip())
            # a prefixed preposition (اللام، الباء، الكاف) opens a word whose pronoun follows: لكم، به
            if ch in ("ل", "ب", "ك") and cursor < len(ws) and ws[cursor]["norm"].startswith(ch) \
                    and "جر" in R.norm(gloss):
                pending = cursor
            continue
        if len(toks) == 1 and toks[0] in R.PRON_TOK | {"هو", "هي", "انت", "انا", "نحن", "كما", "هم", "ه", "ك", "ي"}:
            if pending is not None and ws[pending]["norm"][1:].startswith(toks[0]):
                dep, pending = pending, None            # (اللام) حرف جر و (كم) ضمير ... متعلق ب
                cursor = dep + 1
                last = dep
                r = recs[dep]
                r["jadwal"] = (r.get("jadwal", "") + " | " if r.get("jadwal") else "") + f"({tok}) {gloss.strip()[:200]}"
                r.setdefault("function", "جار ومجرور")
                h = R.HEAD_PATTERNS[0][1].search(gloss)
                if h:
                    if h.group(2) or h.group(4) or "بفعل محذوف" in h.group(0):
                        f2 = "حال" if h.group(4) else {"نعت": "صفة"}.get(h.group(3) or "", h.group(3) or "?")
                        r["mutaallaq"] = {"محذوف": f2}
                    else:
                        k = R.resolve(h.group(1), ws, dep)
                        if k is not None:
                            r["mutaallaq"] = k
                continue
            # an attached pronoun: belongs to the word just analysed (زادهم → هم)
            if last is not None and ws[last]["norm"].endswith(toks[0]):
                recs[last].setdefault("suffix", []).append(f"({tok}) {gloss.strip()[:80]}")
                continue
        dep = None
        for j in range(cursor, min(cursor + 6, len(ws))):
            wn = ws[j]["norm"]
            if toks[0] == wn or (len(toks[0]) >= 3 and toks[0] in wn) or \
                    (len(toks[0]) >= 2 and len(toks) == 1 and wn.endswith(toks[0]) and j <= cursor + 1):
                dep = j
                break
        if dep is None:
            continue
        full = toks[0] == ws[dep]["norm"] or len(toks[0]) >= 3
        cursor = dep + len(toks) if full else dep
        last = dep + len(toks) - 1 if full else dep
        r = recs[dep]
        r["jadwal"] = (r.get("jadwal", "") + " | " if r.get("jadwal") else "") + f"({tok}) {gloss.strip()[:200]}"
        fn = function_of(gloss)
        if fn and "function" not in r:
            r["function"] = fn
        for cat, pat in R.HEAD_PATTERNS:
            h = pat.search(gloss)
            if not h:
                continue
            if cat == "pp":
                if h.group(2) or h.group(4) or "بفعل محذوف" in h.group(0):
                    f2 = "حال" if h.group(4) else {"نعت": "صفة"}.get(h.group(3) or "", h.group(3) or "?")
                    r["mutaallaq"] = {"محذوف": f2}
                else:
                    k = R.resolve(h.group(1), ws, dep)
                    if k is not None:
                        r["mutaallaq"] = k
            else:
                k = R.resolve(h.group(1), ws, dep)
                if k is not None and k != dep:
                    r["matbu"] = {"type": {"atf": "عطف", "naat": "نعت", "badal": "بدل"}[cat], "word": k}
            break
        if re.search(r"ل?فعل محذوف", gloss) and "amil" not in r and "mutaallaq" not in r:
            r["amil"] = {"محذوف": "فعل"}              # مفعول به لفعل محذوف أي جعل السماء
        hm = HAL.search(gloss)
        if hm and "mutaallaq" not in r:
            k = R.resolve(hm.group(1), ws, dep)
            if k is not None:
                r["sahib_hal"] = k
        if re.search(r"توكيد\s+ل", gloss) and "matbu" not in r:
            m2 = re.search(r"توكيد\s+ل‍?\s*\(?([^\s(),.]+)", gloss)
            k = R.resolve(m2.group(1), ws, dep) if m2 else None
            if k is not None and k != dep:
                r["matbu"] = {"type": "توكيد", "word": k}
        if ws[dep]["verb"] and re.match(r"\s*(فعل|مضارع|ماض|أمر)", gloss):
            ext, explicit = gloss, False
            for t2, g2 in ents[ei + 1:ei + 4]:
                if R.LETTER.match(t2.strip()) or R.norm(t2) in R.PRON_TOK:
                    ext += " " + g2
                    if re.match(r"\s*(ضمير\s+)?(متّصل\s+)?(مبنيّ\s+\S+\s+)?(في محلّ رفع\s+)?(فاعل|نائب فاعل|اسم كان)", g2):
                        explicit = True
                else:
                    break
            if re.search(r"والواو فاعل|و\s*\(?الواو\)?\s+فاعل|ضمير فاعل|والتاء فاعل|ونا فاعل|والألف فاعل|والنون فاعل|ضمير في محلّ رفع فاعل", ext):
                explicit = True
            r["muqaddar"] = {"مستتر": bool(R.HIDDEN.search(ext[:300])) and not explicit}
        ra = ROLE_AR.get(r["role"])
        fa = EQUIV.get(r.get("function"), r.get("function"))
        if ra and fa and R.norm(fa) != R.norm(ra) and fa not in ("جار ومجرور", "حرف جر", "ضمير", "حرف"):
            r.setdefault("flags", []).append(f"الجدول: {fa} / MASAQ: {ra}")
    return recs


def sura_pages(suras_json):
    toc = sorted(json.load(open(suras_json, encoding="utf-8")))
    names = [n for _, n in toc]
    return toc, names


def page_text(pages_dir, p):
    fn = Path(pages_dir) / f"p{p}.html"
    if not fn.exists():
        return ""
    s = fn.read_text(encoding="utf-8", errors="replace")
    m = re.search(r'<div class="nass[^"]*"[^>]*>(.*?)</div>\s*</div>', s, re.S)
    t = m.group(1) if m else ""
    t = re.sub(r'<p class="hamesh">.*?</p>', "", t, flags=re.S)    # footnotes (alternative analyses)
    t = re.sub(r"<br\s*/?>|</p>", "\n", t)
    return html.unescape(re.sub(r"<[^>]+>", "", t))


BLOCK = re.compile(r"\{([^{}]+?)\}\s*\n?\s*الإعراب\s*:(.*?)(?=\{[^{}]+?\}\s*\n?\s*الإعراب\s*:|\Z)", re.S)


def blocks_with_header(text):
    for m in BLOCK.finditer(text):
        nums = [int(x.translate(R.AR_DIG)) for x in re.findall(r"\(([٠-٩]+)\)", m.group(1))]
        body = m.group(2)
        e = R.SECTION_END.search(body)
        body = body[:e.start()] if e else body
        # drop the clause analyses (جملة: «...» لا محلّ لها ...) but keep what follows them:
        # later verses of the block are glossed after them ("٥ - (الواو) ...")
        body = "\n".join(l for l in body.split("\n") if not re.match(r"\s*و?جملة\s*:", l))
        yield nums, R.norm(re.sub(r"\([٠-٩]+\)", " ", m.group(1))).split(), body


def matches(header, masaq, sura, verse):
    ws = [w["norm"] for w in masaq.get((str(sura), verse), [])][:4]
    if not ws or not header:
        return False
    hits = sum(1 for a, b in zip(header[:4], ws) if a == b or a in b or b in a)
    return hits >= min(3, len(ws))


def iter_suras(pages_dir, suras_json, masaq, last_page=5940):
    """Yield (sura number, [(verse numbers, body)]). The sūra of each block is
    found by matching the verse text in its header against MASAQ."""
    toc = sorted(json.load(open(suras_json, encoding="utf-8")))
    first = toc[0][0]
    text = "\n".join(re.sub(r"تحميل الصفحة التالية.*", "", page_text(pages_dir, p), flags=re.S)
                     for p in range(first, last_page))
    sura, cur = 1, []
    for nums, header, body in blocks_with_header(text):
        if not nums:
            continue
        if not matches(header, masaq, sura, nums[0]):
            for cand in range(sura + 1, min(sura + 4, 115)):
                if matches(header, masaq, cand, nums[0]):
                    if cur:
                        yield sura, cur
                    sura, cur = cand, []
                    break
            else:
                continue              # unmatched block: skip rather than misalign
        cur.append((nums, body))
    if cur:
        yield sura, cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pages_dir")
    ap.add_argument("suras_json")
    ap.add_argument("masaq")
    ap.add_argument("out")
    ap.add_argument("--suras", default="")
    a = ap.parse_args()
    masaq = R.load_masaq(a.masaq)
    want = {int(x) for x in a.suras.split(",") if x}
    stats = Counter()
    with gzip.open(a.out, "wt", encoding="utf-8") as o:
        for sura, blocks in iter_suras(a.pages_dir, a.suras_json, masaq):
            if want and sura not in want:
                continue
            for nums, body in blocks:
                ws = [w for v in nums for w in masaq.get((str(sura), v), [])]
                if not ws:
                    stats["blocks_without_masaq"] += 1
                    continue
                recs = convert_block(nums, body, ws)
                stats["words"] += len(recs)
                stats["with_jadwal"] += sum("jadwal" in r for r in recs)
                for k in ("function", "mutaallaq", "matbu", "sahib_hal", "muqaddar", "amil", "flags"):
                    stats[k] += sum(k in r for r in recs)
                o.write(json.dumps({"sura": sura, "verses": nums, "words": recs}, ensure_ascii=False) + "\n")
            stats["suras"] += 1
    print(dict(stats))


if __name__ == "__main__":
    main()
