"""Separate an author's own words from what he quotes.

Three kinds of borrowed text are found:
  Q  Qur'an        - word 4-grams shared with the Qur'an
  H  hadith        - word 5-grams shared with nine hadith collections
                     (isnad and formula words are not indexed, only matn)
  M  proverb       - whole proverbs from three proverb books
  I  isnad         - chains of transmission (حدثنا فلان عن فلان ...)

Each Q/H/M span is then classified:
  block     reproduction: the author hands the floor to the source
            (12+ words, or the span is its whole sentence apart from framing
            words such as قال تعالى / وفي الحديث / وفي المثل)
  embedded  the phrase is woven into the author's own sentence
            (اقتباس, proverbs in passing)

Output text keeps the author's words. Blocks and isnads are replaced by
placeholders (<Q-BLOCK>, <H-BLOCK>, <M-BLOCK>, <ISNAD>) and embedded quotations
by <Q>, <H>, <M>: their words no longer count as the author's, but the fact of
quoting stays visible as a countable feature.

usage: python -m corpus.quotations DATA_DIR REFS_DIR [--chunk 2000]
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from corpus.openiti import parse_markdown

# ---------------------------------------------------------------- normalizing
MARKS = re.compile(r"[ً-ْٰـ]")
TOKEN = re.compile(r"[ء-يً-ْٰـ]+|[.؟?!:؛;\n«»﴿﴾]")
BOUNDARY = set(".؟?!:؛;\n«»﴿﴾")


def norm_word(w):
    w = MARKS.sub("", w)
    w = re.sub("[أإآ]", "ا", w)
    return w.replace("ى", "ي").replace("ة", "ه")


def tokenize(text):
    """[(norm, start, end, is_word)]"""
    out = []
    for m in TOKEN.finditer(text):
        s = m.group(0)
        if s in BOUNDARY:
            out.append((s, m.start(), m.end(), False))
        else:
            out.append((norm_word(s), m.start(), m.end(), True))
    return out


def words_of(text):
    return [norm_word(w) for w in re.findall(r"[ء-يً-ْٰـ]+", text)]


# ---------------------------------------------------------------- references
# words that frame or transmit a report; hadith shingles containing them are
# not indexed (they would match isnads and formulas, not the Prophet's words)
FRAME = {norm_word(w) for w in """حدثنا حدثني حدثناه أخبرنا أخبرني أخبرناه ثنا نا أنا أنبأنا
أنبأني سمعت سمعنا عن بن ابن بنت أبو أبي أبا صلى وسلم رسول النبي قال قالت يقول""".split()}

Q_N, H_N = 4, 5

# Shared formulas of the period (dates, piety, letter openings): a hadith
# shingle made mostly of these is convention, not quotation.
FORMULA = {norm_word(w) for w in """الله بالله لله تعالى عز وجل سبحانه رضي عنه عنها عنهم صلى عليه
وسلم آله رحمه رحمة إله إلا هو حول ولا قوة حمد فحمد وأثنى أثنى بسم الرحمن الرحيم من في سنة
شهر ليلة ليال خلت بقيت يوم المحرم صفر ربيع الأول الآخر جمادى الأولى رجب شعبان رمضان شوال
القعدة الحجة عشرة عشر""".split()}
BASMALA = " ".join(norm_word(w) for w in "بسم الله الرحمن الرحيم".split())


def shingles(ws, n):
    for i in range(len(ws) - n + 1):
        yield i, " ".join(ws[i:i + n])


def load_reference_paras(path):
    return parse_markdown(Path(path).read_text(encoding="utf-8"))


def build_quran(refs):
    s = set()
    for p in load_reference_paras(refs / "raw/0001Quran.Mushaf.mARkdown"):
        if p["type"] == "heading":
            continue
        ws = words_of(re.sub(r"\(\s*\d+\s*\)", " ", p["text"]))
        s.update(hash(g) for _, g in shingles(ws, Q_N) if g != BASMALA)
    return s


HADITH = ["0256Bukhari.Sahih", "0261Muslim.Sahih", "0275AbuDawudSijistani.Sunan",
          "0279Tirmidhi.Sunan", "0303Nasai.SunanSughra", "0273IbnMaja.Sunan",
          "0179MalikIbnAnas.Muwatta", "0241IbnHanbal.Musnad", "0255CabdAllahDarimi.Sunan"]


def build_hadith(refs):
    s = set()
    for b in HADITH:
        for p in load_reference_paras(refs / f"raw/{b}.mARkdown"):
            if p["type"] != "prose":
                continue
            ws = words_of(p["text"])
            for i, g in shingles(ws, H_N):
                win = ws[i:i + H_N]
                if not FRAME.intersection(win) and sum(w in FORMULA for w in win) < 3:
                    s.add(hash(g))
    return s


PROVERB_END = re.compile(r"\s(?:قال|قاله|ويروى|يروى|يقال|يضرب|أي|وهو|وهي|وذلك|يريد|يعني|"
                         r"أصله|وأصله|معناه|ومعناه|المثل|بلد|اسم|موضع|رجل|حكاه)\s|\sهو من\s|[،,.:؛|]")


def build_proverbs(refs):
    """Proverb headwords only; the books' explanations are not indexed."""
    provs = set()
    raw = []
    may = (refs / "raw/0518IbnMuhammadMaydaniNaysaburi.MajmacAmthal.mARkdown").read_text(encoding="utf-8")
    raw += [m.group(1) for m in re.finditer(r"^# \|\s*(.+)$", may, re.M)]
    abu = (refs / "raw/0224QasimIbnSallam.Amthal.mARkdown").read_text(encoding="utf-8")
    raw += [m.group(1) for m in re.finditer(r"^### \|\s*(?!باب)(.+)$", abu, re.M)]
    zam = (refs / "raw/0538JarAllahZamakhshari.MustaqsaFiAmthal.mARkdown").read_text(encoding="utf-8")
    raw += [m.group(1) for m in re.finditer(r"^# [0-9]+\s*[-–]\s*(.+)$", zam, re.M)]
    for r in raw:
        r = re.sub(r"ms\d+|PageV\d+P\d+", " ", r)
        provs.add(PROVERB_END.split(" " + r + " ", 1)[0])   # cut at the first explanatory word
    index = defaultdict(list)
    n = 0
    for p in provs:
        ws = words_of(p)
        if 3 <= len(ws) <= 10 and sum(w in FORMULA for w in ws) * 2 < len(ws):
            index[(ws[0], ws[1])].append(tuple(ws))
            n += 1
    return index, n


# ---------------------------------------------------------------- isnad
TRANSMIT = {norm_word(w) for w in """حدثنا حدثني حدثناه حدثنيه حدثه حدثهم أخبرنا أخبرني أخبرناه
أخبره ثنا نا أنبأنا أنبأني أنبأه سمعت سمعنا""".split()}
NAME_LINK = {"بن", "ابن", "بنت", "ابو", "ابي", "ابا", "عبد", "ام"}
SAYS = {"قال", "قالت", "قالا", "قالوا", "يقول", "فقال"}


def is_transmit(w):
    return w in TRANSMIT or (w[:1] in "وف" and w[1:] in TRANSMIT)


def find_isnads(ws):
    """Spans [i, j) of transmission chains. A chain starts at a transmission
    verb and continues while another transmission word, عن, or a name link
    (بن, أبي, عبد ...) follows within 5 words; it ends before the reporter's
    قال / أن or after a short name tail."""
    spans, i, n = [], 0, len(ws)
    while i < n:
        if not is_transmit(ws[i]):
            i += 1
            continue
        start, last = i, i
        j = i + 1
        while j < n and j - last <= 5:
            if (ws[j] in SAYS and not (j + 1 < n and is_transmit(ws[j + 1]))) \
                    or ws[j] in ("ان", "انه", "انها"):
                break                     # the report itself begins
            if is_transmit(ws[j]) or ws[j] == "عن" or ws[j] in NAME_LINK:
                last = j
            j += 1
        end = last + 1
        while end < n and end - last <= 5 and ws[end] not in SAYS | {"ان", "انه", "انها"}:
            end += 1
        links = sum(1 for w in ws[start:end] if is_transmit(w) or w == "عن")
        has_name = any(w in NAME_LINK for w in ws[start:end])
        if links >= 2 or has_name:        # not "سمعت أعرابيا": a chain, or a named transmitter
            spans.append((start, end))
        i = end
    return spans


# ---------------------------------------------------------------- classify
FRAMING = {norm_word(w) for w in """قال قالت فقال وقال يقول قوله وقوله كقوله تعالى الله عز وجل
سبحانه رسول النبي صلى عليه وسلم الحديث وفي في المثل يقال قيل وقيل كما ومن أمثالهم ذلك
لقوله وذلك فنزلت نزلت الآيه""".split()}


def mark_spans(toks, quran, hadith, proverbs, hadith_df_ok):
    """Return list of (kind, i, j) over word positions, and the word list."""
    ws = [t[0] for t in toks if t[3]]
    lab = [None] * len(ws)
    for a, b in find_isnads(ws):
        for k in range(a, b):
            lab[k] = "I"
    for i, g in shingles(ws, Q_N):
        if hash(g) in quran:
            for k in range(i, i + Q_N):
                if lab[k] is None:
                    lab[k] = "Q"
    for i, g in shingles(ws, H_N):
        h = hash(g)
        if h in hadith and hadith_df_ok(h):
            for k in range(i, i + H_N):
                if lab[k] is None:
                    lab[k] = "H"
    for i in range(len(ws) - 2):
        for prov in proverbs.get((ws[i], ws[i + 1]), ()):
            if tuple(ws[i:i + len(prov)]) == prov:
                for k in range(i, i + len(prov)):
                    if lab[k] is None:
                        lab[k] = "M"
    # merge into spans, bridging single-word gaps of the same kind
    spans, k = [], 0
    while k < len(lab):
        if lab[k] is None:
            k += 1
            continue
        kind, a = lab[k], k
        while k < len(lab) and (lab[k] == kind or
                                (lab[k] is None and k + 1 < len(lab) and lab[k + 1] == kind)):
            k += 1
        spans.append((kind, a, k))
    # drop scraps left where a match overlapped a higher-priority span, and
    # hadith overlaps of a single 5-gram, which can be chance phrasing
    min_len = {"I": 2, "Q": Q_N, "H": H_N + 1, "M": 3}
    spans = [s for s in spans if s[2] - s[1] >= min_len[s[0]]]
    return spans, ws


def classify(span, toks):
    kind, a, b = span
    if kind == "I":
        return "isnad"
    if b - a >= 12:
        return "block"
    # sentence around the span, in word positions
    word_pos = [i for i, t in enumerate(toks) if t[3]]
    ta, tb = word_pos[a], word_pos[b - 1]
    s = ta
    while s > 0 and toks[s - 1][0] not in BOUNDARY:
        s -= 1
    e = tb
    while e + 1 < len(toks) and toks[e + 1][0] not in BOUNDARY:
        e += 1
    outside = [toks[i][0] for i in range(s, e + 1)
               if toks[i][3] and not (ta <= i <= tb) and toks[i][0] not in FRAMING]
    return "block" if len(outside) <= 2 else "embedded"


PLACEHOLDER = {("Q", "block"): "<Q-BLOCK>", ("H", "block"): "<H-BLOCK>", ("M", "block"): "<M-BLOCK>",
               ("Q", "embedded"): "<Q>", ("H", "embedded"): "<H>", ("M", "embedded"): "<M>",
               ("I", "isnad"): "<ISNAD>"}


def process_paragraph(text, quran, hadith, proverbs, hadith_df_ok, stats):
    toks = tokenize(text)
    spans, ws = mark_spans(toks, quran, hadith, proverbs, hadith_df_ok)
    word_pos = [i for i, t in enumerate(toks) if t[3]]
    out, cursor = [], 0
    for sp in spans:
        kind, a, b = sp
        how = classify(sp, toks)
        stats[f"{kind}_{how}_words"] += b - a
        stats[f"{kind}_{how}_n"] += 1
        cs, ce = toks[word_pos[a]][1], toks[word_pos[b - 1]][2]
        out.append(text[cursor:cs])
        out.append(f" {PLACEHOLDER[(kind, how)]} ")
        cursor = ce
    out.append(text[cursor:])
    stats["words_total"] += len(ws)
    stats["words_author"] += len(ws) - sum(b - a for _, a, b in spans)
    return re.sub(r"[ \t]+", " ", "".join(out)).strip()


# ---------------------------------------------------------------- driver
def main():
    from corpus.build import chunk_paragraphs, AR_WORD
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("refs_dir")
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--max-df", type=float, default=0.10,
                    help="hadith shingles found in more than this share of books are "
                         "treated as common phrasing, not quotation")
    a = ap.parse_args()
    root, refs = Path(a.data_dir), Path(a.refs_dir)

    print("indexing references ...")
    quran = build_quran(refs)
    hadith = build_hadith(refs)
    proverbs, nprov = build_proverbs(refs)
    print(f"  Qur'an {len(quran):,} 4-grams, hadith {len(hadith):,} 5-grams, {nprov:,} proverbs")

    books = sorted(p.stem for p in (root / "clean").glob("*.jsonl"))
    paras = {b: [json.loads(l) for l in open(root / "clean" / f"{b}.jsonl", encoding="utf-8")]
             for b in books}

    # document frequency of matched hadith shingles across benchmark books
    df = Counter()
    for b in books:
        seen = set()
        for p in paras[b]:
            if p["type"] == "prose":
                for _, g in shingles(words_of(p["text"]), H_N):
                    h = hash(g)
                    if h in hadith:
                        seen.add(h)
        df.update(seen)
    limit = a.max_df * len(books)
    common = {h for h, c in df.items() if c > limit}
    print(f"  {len(common):,} hadith shingles dropped as common phrasing (in >{limit:.0f} books)")
    ok = lambda h: h not in common

    (root / "authorial").mkdir(exist_ok=True)
    (root / "chunks_authorial").mkdir(exist_ok=True)
    meta = {r.split("\t")[0]: r.rstrip("\n").split("\t")
            for r in open(root / "books.tsv", encoding="utf-8")}
    cols = ["words_total", "words_author", "I_isnad_words", "Q_block_words", "Q_embedded_words",
            "H_block_words", "H_embedded_words", "M_block_words", "M_embedded_words",
            "I_isnad_n", "Q_embedded_n", "H_embedded_n", "M_embedded_n"]
    report = ["book\t" + "\t".join(cols) + "\tchunks"]
    manifest = ["chunk_id\tbook\tauthor\tdeath_ah\tgenre\tstatus\tpage_from\tpage_to\twords\tpath"]
    for b in books:
        stats = Counter()
        out = []
        for p in paras[b]:
            if p["type"] == "prose":
                p = {**p, "text": process_paragraph(p["text"], quran, hadith, proverbs, ok, stats)}
            out.append(p)
        with open(root / "authorial" / f"{b}.jsonl", "w", encoding="utf-8") as f:
            for p in out:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        _, author, death, genre, status = meta[b][:5]
        cdir = root / "chunks_authorial" / b
        cdir.mkdir(exist_ok=True)
        for old in cdir.glob("*.txt"):
            old.unlink()
        n = 0
        for i, (p0, p1, words, text) in enumerate(chunk_paragraphs(out, a.chunk)):
            path = cdir / f"{i:04d}.txt"
            path.write_text(text, encoding="utf-8")
            manifest.append(f"{b}.{i:04d}\t{b}\t{author}\t{death}\t{genre}\t{status}"
                            f"\t{p0}\t{p1}\t{words}\t{path.relative_to(root)}")
            n += 1
        report.append(b + "\t" + "\t".join(str(stats[c]) for c in cols) + f"\t{n}")
        own = stats["words_author"] / max(stats["words_total"], 1)
        print(f"{b:48} own words {own:6.1%}  isnad {stats['I_isnad_words']:>7,}  "
              f"Q {stats['Q_block_words']:>6,}/{stats['Q_embedded_n']:>4}  "
              f"H {stats['H_block_words']:>6,}/{stats['H_embedded_n']:>4}  "
              f"M {stats['M_block_words']:>5,}/{stats['M_embedded_n']:>4}  chunks {n}")
    (root / "quotations.tsv").write_text("\n".join(report) + "\n", encoding="utf-8")
    (root / "manifest_authorial.tsv").write_text("\n".join(manifest) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
