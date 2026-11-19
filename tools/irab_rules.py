"""Rule-based heads on MASAQ, checked against الجدول في إعراب القرآن.

MASAQ (v5, CC BY 4.0) gives each word segment its i'rab role but no head.
Rules supply the head; الجدول, where it names the head explicitly, is the
yardstick. Categories checked:

  pp      جار ومجرور: the word it attaches to, or محذوف (خبر/صفة/حال/صلة)
  atf     معطوف على X
  naat    نعت لـ X
  badal   بدل من X
  hidden  whether a verb has a hidden subject (ضمير مستتر)

usage: python irab_rules.py   (writes /home/claude/syntax/irab_eval.json)
"""
import csv
import json
import re
from collections import Counter, defaultdict

SURAS = {"baqara": "2", "yusuf": "12", "naba": "78"}
AR_DIG = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
DIAC = re.compile("[ً-ْٰـۡ۟ۥۦ۪-ۭٓ-ٟ]")


def norm(s):
    s = DIAC.sub("", s or "")
    s = re.sub("[إأآٱ]", "ا", s).replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    return re.sub("[^ء-ي ]", "", s).strip()


# ---------------------------------------------------------------- MASAQ
VERB_TAGS = {"PV", "IV", "CV", "PV_PASS", "IV_PASS"}
DERIV = {"NOUN_ACTIVE_PART", "NOUN_PASSIVE_PART", "GERUND", "ADJ_QUALIT", "ADJ_INTENS", "ADJ_COMP"}
NOMINAL = {"NOUN_CONCRETE", "NOUN_ABSTRACT", "GERUND", "NOUN_PROP", "NOUN_ACTIVE_PART", "NOUN_PASSIVE_PART",
           "ADJ_QUALIT", "ADJ_INTENS", "ADJ_COMP", "NOUN_NUM", "NOUN_TIME_PLACE", "NOUN_PROP_FOREIGN",
           "REL_PRON", "DEM_PRON", "DEM_PRON_MS", "PRON"}
PRED_FUN = {"PRED": "خبر", "PART_COP_PRED": "خبر", "V_COP_PRED": "خبر", "NEG_CAT_PRED": "خبر",
            "CIRCUM": "حال", "ADJ": "صفة"}


def load_masaq(path="/home/claude/data_ext/masaq_v5/MASAQ.csv"):
    words = defaultdict(list)          # (sura, verse) -> list of word dicts
    cur = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        key = (r["Sura_No"], int(r["Verse_No"]))
        wn = int(r["Word_No"])
        if cur.get(key) != wn:
            words[key].append({"verse": key[1], "n": wn, "text": r["Word"], "norm": norm(r["Word"]), "segs": []})
            cur[key] = wn
        words[key][-1]["segs"].append(r)
    for ws in words.values():
        for w in ws:
            segs = w["segs"]
            stems = [s for s in segs if s["Morph_Type"] == "Stem"] or segs
            st = stems[0]
            w["tag"], w["role"], w["case"] = st["Morph_Tag"], st["Syntactic_Role"], st["Case_Mood"]
            w["verb"] = st["Morph_Tag"] in VERB_TAGS
            w["passive"] = st["Morph_Tag"] in ("PV_PASS", "IV_PASS")
            w["deriv"] = st["Morph_Tag"] in DERIV
            w["nominal"] = st["Morph_Tag"] in NOMINAL
            w["conj"] = any(s["Morph_Tag"] == "CONJ" and s["Morph_Type"] == "Prefix" for s in segs) \
                or (st["Morph_Tag"] == "CONJ")
            pre = [s for s in segs if s["Morph_Tag"] == "PREP"]
            w["prep"] = bool(pre)
            w["pfun"] = pre[0]["Phrasal_Function"] if pre else ""
            w["agnt_in"] = any(
                s is not st and (s["Syntactic_Role"] in ("AGNT", "PASS_SUBJ", "SUBJ_COP_V")
                                 or re.match(r"(PV|IV|CV)SUFF_SUBJ:(?!3MS|3FS)", s["Morph_Tag"]))
                for s in segs)
            w["defn"] = any(s["Morph_Tag"] == "DET" for s in segs) or st["Morph_Tag"].startswith(
                ("NOUN_PROP", "PRON", "DEM", "REL")) or any(s["Morph_Tag"].startswith("POSS_PRON") for s in segs)
    return words


# ---------------------------------------------------------------- rules
def is_head(w):
    return w["verb"] or w["deriv"]


KANA = {"كان", "كون", "يكون", "تكون", "نكون", "اكون", "كن", "صار", "يصير", "اصبح", "يصبح", "ظل", "بات", "ليس", "مازال", "زال", "يزال", "امسي"}


def is_kana(w):
    return w["role"] in ("IV_COP", "CV_COP") or any(norm(s["Segmented_Word"]) in KANA and s["Morph_Type"] == "Stem" for s in w["segs"])


def rule_pp(ws, i):
    w = ws[i]
    if w["pfun"] in PRED_FUN:
        return ("محذوف", PRED_FUN[w["pfun"]])
    if i > 0 and ws[i - 1]["tag"] == "REL_PRON":
        return ("محذوف", "صلة")
    start = i
    while start > 0 and not ws[start]["conj"] and ws[start - 1]["tag"] != "REL_PRON":
        start -= 1
    before = ws[start:i]
    SUBJS = ("SUBJ", "SUBJ_COP_PART", "SUBJ_COP_V", "SUBJ_NEG_CAT")
    PREDS = ("PRED", "PART_COP_PRED", "V_COP_PRED")
    PREDS_ALL = ("PRED", "PART_COP_PRED", "V_COP_PRED", "NEG_CAT_PRED")
    # a derived predicate later in the clause governs the PP: وَهُوَ بِكُلِّ شَيْءٍ عَلِيمٌ، إِنَّ اللَّهَ عَلَى كُلِّ شَيْءٍ قَدِيرٌ
    for j in range(i + 1, min(i + 5, len(ws))):
        if ws[j]["conj"] or ws[j]["verb"]:
            break
        if ws[j]["role"] in PREDS_ALL and ws[j]["deriv"]:
            return ("word", j)
    # كان and its sisters with no expressed predicate: the PP is the خبر — وَكَانَ مِنَ الْكَافِرِينَ
    kv = next((j for j in range(i - 1, max(start, i - 4) - 1, -1) if ws[j]["verb"]), None)
    if kv is not None and is_kana(ws[kv]) and not any(w2["role"] in PREDS_ALL for w2 in ws[kv + 1:i]):
        return ("محذوف", "خبر")
    # fronted khabar: [إنّ/لا/و] PP (PP) SUBJ  — وَلَهُمْ عَذَابٌ، إِنَّ فِي ذَلِكَ لَآيَةً
    if not any(w2["verb"] for w2 in before):
        for j in range(i + 1, min(i + 4, len(ws))):
            if ws[j]["role"] in SUBJS:
                return ("محذوف", "خبر")
            if not (ws[j]["prep"] or ws[j]["role"] in ("PREP_OBJ", "GEN_CONS")):
                break
    # delayed khabar: SUBJ ... PP with no verb or predicate in between — أُولَئِكَ عَلَى هُدًى
    k = next((j for j in range(len(before) - 1, -1, -1) if before[j]["role"] in SUBJS), None)
    if k is not None and not any(w2["verb"] or w2["role"] in PREDS for w2 in before[k + 1:]) \
            and not (ws[i - 1]["verb"]):
        return ("محذوف", "خبر")
    # after an indefinite noun: صفة — هُدًى مِنْ رَبِّهِمْ
    p = ws[i - 1] if i > 0 else None
    if p and p["nominal"] and not p["defn"] and not p["verb"] and p["role"] not in ("PREP_OBJ",) and p["tag"] not in DERIV:
        return ("محذوف", "صفة")
    back = next((j for j in range(i - 1, -1, -1) if is_head(ws[j])), None)
    fwd = None
    for j in range(i + 1, len(ws)):
        if ws[j]["conj"] or ws[j]["tag"] == "REL_PRON":
            break
        if is_head(ws[j]):
            fwd = j
            break
    initial = i == 0 or (ws[i - 1]["conj"] and not is_head(ws[i - 1])) or ws[i - 1]["tag"] in ("ANNUL_PART", "NEG_PART") \
        or ws[i - 1]["role"] in ("SUBJ", "SUBJ_COP_PART")
    if back is None or (initial and fwd is not None):
        return ("word", fwd) if fwd is not None else ("محذوف", "?")
    return ("word", back)


def rule_same_case(ws, i, skip_roles, need_def=False):
    w = ws[i]
    for j in range(i - 1, -1, -1):
        c = ws[j]
        if not c["nominal"] or c["role"] in skip_roles:
            continue
        if c["case"] != w["case"]:
            continue
        if need_def and c["defn"] != w["defn"]:
            continue
        return j
    return None


def rule_atf(ws, i):
    return rule_same_case(ws, i, {"ADJ", "APPOS"})


def rule_naat(ws, i):
    j = rule_same_case(ws, i, {"ADJ"}, need_def=True)
    return j if j is not None else rule_same_case(ws, i, {"ADJ"})


def rule_badal(ws, i):
    return rule_same_case(ws, i, set())


def rule_hidden(ws, i):
    w = ws[i]
    if w["agnt_in"]:
        return False
    for j in range(i + 1, len(ws)):
        if ws[j]["verb"]:
            break
        if ws[j]["role"] in ("AGNT", "PASS_SUBJ", "SUBJ_COP_V"):
            return False
    return True


# ---------------------------------------------------------------- الجدول
SECTION_END = re.compile(r"\n\s*(الصرف|البلاغة|الفوائد|\[الفوائد\]|إعراب|المعنى|التفسير)\s*[:\n]")
ENTRY = re.compile(r"\(([^()]{1,40})\)\s*([^()]*)")
LETTER = re.compile(r"^(ال)?(واو|فاء|لام|باء|كاف|همزة|سين|تاء|نون|ياء|هاء|ميم|ألف|الواو|الفاء|اللام|الباء|الكاف|الهمزة|السين|التاء|النون|الياء|الهاء)$")


def blocks(pages):
    t = "\n".join(re.sub(r"تحميل الصفحة التالية.*", "", p["text"], flags=re.S) for p in pages)
    for m in re.finditer(r"\{([^{}]+?)\}\s*\n?\s*الإعراب\s*:(.*?)(?=\{[^{}]+?\}\s*\n?\s*الإعراب\s*:|\Z)", t, re.S):
        nums = [int(x.translate(AR_DIG)) for x in re.findall(r"\(([٠-٩]+)\)", m.group(1))]
        body = m.group(2)
        e = SECTION_END.search(body)
        body = body[:e.start()] if e else body
        body = body.split("جملة:")[0] if "جملة:" in body else body
        yield nums, body


def resolve(x, ws, near):
    """Index of the word in ws best matching the text x (nearest to `near`, preceding preferred)."""
    xn = norm(x)
    if xn in ("لفظ", "لفظ الجلاله", "الجلاله"):
        xn = "الله"                       # نعت للفظ الجلالة
    xn = re.sub(r"^(ب|ل|ك)\s*", "", xn) if len(xn) > 3 else xn
    if not xn:
        return None
    cands = [j for j, w in enumerate(ws) if xn == w["norm"] or (len(xn) >= 3 and xn in w["norm"])
             or (len(w["norm"]) >= 3 and w["norm"] in xn and len(xn) - len(w["norm"]) <= 4)]
    if not cands:
        return None
    pre = [j for j in cands if j < near]
    return max(pre) if pre else min(cands)


HEAD_PATTERNS = [
    ("pp", re.compile(r"متعلّ?ق(?:ان)?\s*(?:ب\u200d?\s*:?\s*\(([^()]+)\)|ب(محذوف)\s*(خبر|صفة|نعت|حال|صلة)?|بفعل محذوف|ب(حال)\s+من)")),
    ("atf", re.compile(r"معطوف(?:\s+ب\S+)?\s+على\s+\(?([^\s(),.]+(?:\s+[^\s(),.]+)?)")),
    ("naat", re.compile(r"(?:نعت|صفة)(?:\s+(?:ثان|ثالث|رابع|أوّل|أول|ثانية|ثالثة)\S*)?(?:\s+(?:مجرور|مرفوع|منصوب)\S*)?\s+ل\u200d?\s*\(?([^\s(),.]+)")),
    ("badal", re.compile(r"بدل\s+(?:من|عن)\s+\(?([^\s(),.]+)")),
]


REF_BEFORE = re.compile(r"(متعلّ?ق(?:ان)?\s*ب\u200d?\s*|(?:معطوف|عطف)(?:\s+ب\S+)?\s+على\s*|(?:نعت|صفة)(?:\s+\S+)?\s+ل\u200d?\s*|بدل\s+(?:من|عن)\s*|حال\s+من\s*|بحال\s+من\s*|توكيد\s+ل\u200d?\s*)$")
HIDDEN = re.compile(r"مستتر|والفاعل\s+(هو|هي|أنت|أنا|نحن)|فاعله\s+(هو|هي|أنت|أنا|نحن)|نائب الفاعل\s+(هو|هي|أنت|أنا|نحن)")


def entries(body):
    """[(token, gloss)], with parenthesised head references folded into the gloss."""
    parts = re.split(r"\(([^()]{1,40})\)", body)
    out = []
    for k in range(1, len(parts), 2):
        tok, after = parts[k], parts[k + 1] if k + 1 < len(parts) else ""
        if out and REF_BEFORE.search(out[-1][1]) and not re.match(r"^[\d٠-٩]+$", tok):
            out[-1] = (out[-1][0], out[-1][1] + "(" + tok + ")" + after)
            continue
        if re.match(r"^[\d٠-٩]+$", tok):
            if out:
                out[-1] = (out[-1][0], out[-1][1] + after)
            continue
        out.append((tok, after))
    return out


PRON_TOK = {"هم", "هما", "هن", "ه", "ها", "ك", "كم", "كما", "كن", "نا", "ي", "ت", "تم", "تما", "تن", "وا", "و", "ن", "ا"}


def gold(nums, body, ws):
    """Gold facts from one block: list of (category, dep index, head)."""
    out, cursor = [], 0
    ents = entries(body)
    for ei, (tok, gloss) in enumerate(ents):
        letter = bool(LETTER.match(tok.strip()))
        toks = norm(tok).split()
        if not toks:
            continue
        dep = None
        if letter:
            dep = min(cursor, len(ws) - 1) if ws else None
        else:
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
        if dep is None:
            continue
        for cat, pat in HEAD_PATTERNS:
            h = pat.search(gloss)
            if not h:
                continue
            if cat == "pp":
                if h.group(2) or h.group(4) or "بفعل محذوف" in h.group(0):
                    fn = "حال" if h.group(4) else {"نعت": "صفة"}.get(h.group(3) or "", h.group(3) or "?")
                    out.append(("pp", dep, ("محذوف", fn)))
                else:
                    k = resolve(h.group(1), ws, dep)
                    if k is not None:
                        out.append(("pp", dep, ("word", k)))
            elif not letter:
                k = resolve(h.group(1), ws, dep)
                if k is not None and k != dep:
                    out.append((cat, dep, k))
            break
        if not letter and ws[dep]["verb"] and re.match(r"\s*(فعل|مضارع|ماض|أمر)", gloss):
            ext = gloss
            explicit = False
            for t2, g2 in ents[ei + 1:ei + 4]:
                if LETTER.match(t2.strip()) or norm(t2) in PRON_TOK:
                    ext += " " + g2
                    if re.match(r"\s*(ضمير\s+)?(متّصل\s+)?(مبنيّ\s+\S+\s+)?(في محلّ رفع\s+)?(فاعل|نائب فاعل|اسم كان)", g2):
                        explicit = True
                else:
                    break
            if re.search(r"والواو فاعل|و\s*\(?الواو\)?\s+فاعل|ضمير فاعل|والتاء فاعل|ونا فاعل|والألف فاعل|والنون فاعل|ضمير في محلّ رفع فاعل", ext):
                explicit = True
            hidden = bool(HIDDEN.search(ext[:300])) and not explicit
            out.append(("hidden", dep, hidden))
    return out


def main():
    masaq = load_masaq()
    jad = json.load(open("/home/claude/data_ext/jadwal/jadwal.json", encoding="utf-8"))
    res = {}
    for name, sura in SURAS.items():
        stats = defaultdict(Counter)
        errors = defaultdict(list)
        for nums, body in blocks(jad[name]):
            ws = [w for v in nums for w in masaq.get((sura, v), [])]
            if not ws:
                continue
            for cat, dep, g in gold(nums, body, ws):
                w = ws[dep]
                if cat == "pp":
                    if not w["prep"]:
                        continue
                    p = rule_pp(ws, dep)
                    ok = p == g or (p[0] == g[0] == "محذوف" and (p[1] == g[1] or g[1] == "?"))
                    sub = "محذوف" if g[0] == "محذوف" else "word"
                    stats["pp:" + sub]["n"] += 1
                    stats["pp:" + sub]["ok"] += ok
                elif cat == "hidden":
                    p = rule_hidden(ws, dep)
                    ok = p == g
                    stats["hidden"]["n"] += 1
                    stats["hidden"]["ok"] += ok
                else:
                    fn = {"atf": rule_atf, "naat": rule_naat, "badal": rule_badal}[cat]
                    p = fn(ws, dep)
                    ok = p == g
                    stats[cat]["n"] += 1
                    stats[cat]["ok"] += ok
                if not ok and len(errors[cat]) < 40:
                    def show(x):
                        if isinstance(x, tuple):
                            return x[1] if x[0] == "محذوف" else ws[x[1]]["text"] if x[1] is not None else "?"
                        if isinstance(x, bool):
                            return "مستتر" if x else "ظاهر"
                        return ws[x]["text"] if x is not None else "?"
                    ctx = " ".join(x["text"] for x in ws[max(0, dep - 6):dep + 3])
                    errors[cat].append({"verse": w["verse"], "word": w["text"], "gold": "محذوف " + g[1] if isinstance(g, tuple) and g[0] == "محذوف" else show(g),
                                        "rule": ("محذوف " + p[1]) if isinstance(p, tuple) and p[0] == "محذوف" else show(p), "context": ctx})
        res[name] = {"stats": {k: dict(v) for k, v in stats.items()}, "errors": dict(errors)}
        print(name, {k: f"{v['ok']}/{v['n']} ({v['ok'] / v['n']:.0%})" for k, v in stats.items() if v["n"]})
    json.dump(res, open("/home/claude/syntax/irab_eval.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
