"""Convert «إعراب القرآن الكريم» (دار الصحابة للتراث، طنطا 2006; Shamela book 86)
into the same naḥw fields as jadwal_convert.py, aligned with MASAQ's words.

The book's layout, per verse:

    ٣٦ - {فَأَزَلَّهُمَا الشَّيْطانُ عَنْها ... (٣٦)}
    الإعراب
    {فَأَزَلَّهُمَا}: (الفاء): استئنافية. (أزلهما): فعل ماض. و (الهاء): مفعول به مقدم.
    {الشَّيْطانُ}: فاعل مؤخر مرفوع. وجملة: {فَأَزَلَّهُمَا}: استئنافية. {عَنْها}: جار ومجرور، متعلق ب‍ (أزلهما).

Each {word}: gloss becomes one entry; a gloss that opens with (part): sub-entries
is split into them (clitics). Clause notes («وجملة: {…}: …») and bracketed
alternatives («[أو …]») are left out of the entries. The entries then go
through jadwal_convert.convert_block and jadwal_refs.fill, like الجدول.

Internal use: the converted file is not redistributed.

usage: python layer2b/sahaba_convert.py PAGES_DIR MASAQ.csv OUT.jsonl.gz [--first 1 --last 2714]
"""
import argparse
import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tools"))
import irab_rules as R  # noqa: E402
import jadwal_convert as J  # noqa: E402
import jadwal_refs  # noqa: E402

AR_NUM = r"[٠-٩]+"
HEADER = re.compile(r"(?:^|\n)[ \t]*(?:(" + AR_NUM + r"(?:\s*[-،,و]\s*" + AR_NUM + r")*)\s*-\s*)?"
                    r"\{([^{}]*\(" + AR_NUM + r"\)[^{}]*)\}\s*\n?\s*الإعراب\s*:?")
SURA_MARK = re.compile(r"\((" + AR_NUM + r")\)\s*سورة\s+\S+")
WORD = re.compile(r"\{([^{}]{1,60})\}")
SUB = re.compile(r"\(([^()]{1,40})\)")
CLAUSE_BEFORE = re.compile(r"(جملة|الجملة|جملتا|جملتي|المؤول|المؤولة)\s*:?\s*(من\s*:?\s*)?$")
REF_WORD = re.compile(r"(مثل|إعراب|نظير|جواب|مستتر|تقديره|أي|من|على|ل\u200d?|ب\u200d?)\s*:?$")


def book_text(pages_dir, first, last):
    out = []
    for p in range(first, last + 1):
        t = J.page_text(pages_dir, p)
        out.append(re.sub(r"تحميل الصفحة التالية.*", "", t, flags=re.S))
    return "\n".join(out)


def blocks(text):
    """(verse numbers, header words, body) for every verse block."""
    ms = list(HEADER.finditer(text))
    marks = [(m.start(), int(m.group(1).translate(R.AR_DIG))) for m in SURA_MARK.finditer(text)
             if 1 <= int(m.group(1).translate(R.AR_DIG)) <= 114]
    for i, m in enumerate(ms):
        body = text[m.end(): ms[i + 1].start() if i + 1 < len(ms) else len(text)]
        nums = [int(x.translate(R.AR_DIG)) for x in re.findall(r"\((" + AR_NUM + r")\)", m.group(2))]
        if not nums and m.group(1):
            nums = [int(x.translate(R.AR_DIG)) for x in re.findall(AR_NUM, m.group(1))]
        header = R.norm(re.sub(r"\(" + AR_NUM + r"\)", " ", m.group(2))).split()
        sura = max((n for pos, n in marks if pos < m.start()), default=None, key=lambda n: n) \
            if False else next((n for pos, n in reversed(marks) if pos < m.start()), None)
        yield nums, header, body, sura


def split_sub(tok, gloss):
    """{وَزَوْجُكَ}: (الواو): عاطفة. (زوجك): معطوف… → [(الواو, عاطفة), (زوجك, معطوف…)].
    A parenthesis opens a sub-entry when a colon follows it or it names a letter;
    otherwise it stays in the gloss (a reference: متعلق ب‍ (كلا), اسم (كان))."""
    lead = not re.match(r"\s*\(", gloss)
    out, pos = ([(tok, "")] if lead else []), 0
    for m in SUB.finditer(gloss):
        name = m.group(1).strip()
        after = gloss[m.end():m.end() + 2]
        before = gloss[max(0, m.start() - 6):m.start()].rstrip()
        boundary = (not gloss[:m.start()].strip() or before.endswith((".", "،", "؛", ":"))
                    or re.search(r"(^|\s)و$", before))
        if after.lstrip().startswith(":") or R.LETTER.match(name) or (boundary and not REF_WORD.search(before)):
            if out:
                out[-1] = (out[-1][0], out[-1][1] + gloss[pos:m.start()])
            out.append((name, ""))
            pos = m.end()
            if after.lstrip().startswith(":"):
                pos = gloss.index(":", m.end()) + 1
    if not out:
        return [(tok, gloss)]
    out[-1] = (out[-1][0], out[-1][1] + gloss[pos:])
    return out


PHRASE_REF = re.compile(r"(سبق|تقدم|تقدّم|مر|مرّ)\s+إعراب|نظير|مثل|تعرب\s+إعراب")
CLAUSE_NOTE = re.compile(r"(?:^|(?<=[\s.،؛]))و?(?:ال)?(?:جملة|جملتا|جملتي)(?=[\s:]).*?(?:(?<!\.)\.(?!\.)\s|$)", re.S)


def entries(body, refs=None):
    """[(token, gloss)] for convert_block. Phrase cross-references
    ({…several words…}: سبق إعراب نظيرها في الآية (٤٥)) go to `refs` as (phrase, gloss)."""
    body = re.sub(r"\[[^\[\]]*\]", " ", body)                    # [أو …]: alternatives
    word_marks = []
    for m in WORD.finditer(body):
        x = m.group(1)
        before = body[max(0, m.start() - 24):m.start()]
        if R.REF_BEFORE.search(before.rstrip() + " ") or R.REF_BEFORE.search(before) \
                or re.search(r"(مثل|إعراب|نظير|مثل إعراب)\s*:?\s*$", before):
            continue                                              # a reference: متعلق ب‍ {اِضْرِبْ}
        if CLAUSE_BEFORE.search(body[max(0, m.start() - 14):m.start()]) or "..." in x or "…" in x:
            continue
        word_marks.append(m)
    out = []
    for i, m in enumerate(word_marks):
        end = word_marks[i + 1].start() if i + 1 < len(word_marks) else len(body)
        gloss = body[m.end():end]
        gloss = CLAUSE_NOTE.sub(" ", gloss)                       # وجملة: (…): استئنافية.
        gloss = WORD.sub(lambda k: "(" + k.group(1) + ")", gloss)
        gloss = gloss.lstrip(" :")
        gloss = re.sub(r"(متعلّ?ق(?:ان)?\s*ب)\u200d?\s*:\s*([^\s،.()]+)", "\\1\u200d (\\2)", gloss)   # متعلق ب‍: تفرق
        if len(m.group(1).split()) > 3 or (len(m.group(1).split()) > 1 and PHRASE_REF.search(gloss[:40])):
            if refs is not None and PHRASE_REF.search(gloss[:60]):
                refs.append((m.group(1), gloss))
            gloss = re.sub(r"^[^()]*", "", gloss)                 # keep any sub-entries that follow
            if not gloss:
                continue
        for tok, gl in split_sub(m.group(1), gloss):
            if len(tok.split()) > 1 and PHRASE_REF.search(gl[:40]):
                if refs is not None:
                    refs.append((tok, gl))         # (من أوتي كتابه) سبق إعرابها في الآية (٧)
                continue
            out.append((tok, gl))
    return out


def apply_refs(row, refs, by_verse, stats):
    """Copy the analysis of a phrase from the verse it refers to (same sūra):
    «سبق إعراب نظيرها في الآية (٤٥)», «… في الآية السابقة», or the nearest earlier
    verse holding the same words. Functions only, for the words of the phrase."""
    ws = row["words"]
    norm = [R.norm(w["word"]) for w in ws]
    for phrase, gloss in refs:
        ph = R.norm(phrase).split()
        start = next((k for k in range(len(ws) - len(ph) + 1)
                      if all(a == b or a in b or b in a for a, b in zip(ph, norm[k:k + len(ph)]))), None)
        if start is None:
            stats["phrase_ref_unaligned"] += 1
            continue
        m = re.search(r"الآية\s*:?\s*\(?(" + AR_NUM + r"|\d+)", gloss)
        earlier = sorted((v for (s, v) in by_verse if s == row["sura"] and v < row["verses"][0]), reverse=True)
        if m:
            cands = [int(m.group(1).translate(R.AR_DIG))] + earlier
        elif "السابق" in gloss:
            cands = [row["verses"][0] - 1] + earlier
        else:
            cands = earlier
        done = False
        cur = norm[start:start + len(ph)]
        same = lambda a, b: a == b or (len(a) > 2 and (a in b or b in a))
        for v in cands:
            src = by_verse.get((row["sura"], v))
            if not src:
                continue
            sn = [R.norm(w["word"]) for w in src]
            # نظيرها: the same construction, not always the same words (أصحاب النار / أصحاب الجنة):
            # the window of the same length sharing most words, at least half
            best, k0 = 0, None
            for k in range(len(src) - len(ph) + 1):
                hits = sum(same(a, b) for a, b in zip(cur, sn[k:k + len(ph)]))
                if hits > best:
                    best, k0 = hits, k
            if k0 is None or best * 2 < len(ph):
                continue
            for j in range(len(ph)):
                a, b = ws[start + j], src[k0 + j]
                if b.get("function") and not a.get("function") and a["role"] == b["role"]:
                    a["function"], a["src"], a["ref"] = b["function"], "ref", f"{row['sura']}:{v}"
                    if "muqaddar" in b:
                        a["muqaddar"] = b["muqaddar"]
                    stats["phrase_ref_words"] += 1
            done = True
            break
        stats["phrase_ref_ok" if done else "phrase_ref_missing"] += 1


def iter_suras(text, masaq):
    sura, cur = 1, []
    for nums, header, body, marked in blocks(text):
        if not nums:
            continue
        if marked and marked > sura and J.matches(header, masaq, marked, nums[0]):
            if cur:
                yield sura, cur
            sura, cur = marked, []
        if not J.matches(header, masaq, sura, nums[0]):
            for cand in range(sura + 1, min(sura + 4, 115)):
                if J.matches(header, masaq, cand, nums[0]):
                    if cur:
                        yield sura, cur
                    sura, cur = cand, []
                    break
            else:
                continue
        cur.append((nums, body))
    if cur:
        yield sura, cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pages_dir")
    ap.add_argument("masaq")
    ap.add_argument("out")
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument("--last", type=int, default=2714)
    a = ap.parse_args()
    masaq = R.load_masaq(a.masaq)
    text = book_text(a.pages_dir, a.first, a.last)
    rows, stats = [], Counter()
    by_verse = {}
    for sura, bl in iter_suras(text, masaq):
        stats["suras"] += 1
        for nums, body in bl:
            ws = [w for v in nums for w in masaq.get((str(sura), v), [])]
            if not ws:
                stats["blocks_without_masaq"] += 1
                continue
            refs = []
            row = {"sura": sura, "verses": nums,
                   "words": J.convert_block(nums, body, ws, ents=entries(body, refs))}
            apply_refs(row, refs, by_verse, stats)
            for v in nums:
                by_verse[(sura, v)] = [w for w in row["words"] if w["verse"] == v]
            rows.append(row)
    stats.update(jadwal_refs.fill(rows))
    with gzip.open(a.out, "wt", encoding="utf-8") as o:
        for row in rows:
            recs = row["words"]
            stats["words"] += len(recs)
            stats["glossed"] += sum("jadwal" in r for r in recs)
            for k in ("function", "mutaallaq", "matbu", "sahib_hal", "muqaddar", "amil", "covered_by", "flags"):
                stats[k] += sum(k in r for r in recs)
            o.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(dict(stats))


if __name__ == "__main__":
    main()
