"""Turn Muhyi al-Din's i'rab notes (from alfiyya_extract.py) into per-word fields.

Each note is a run of  " word(s) " gloss  segments. A quote opens a new entry
only when it matches the next verse words; other quotes are references inside
the gloss (e.g. متعلق بقوله " محصلا " الآتي).

Output per verse: tokens [{i, word, entry}], entries [{span, quote, gloss,
function, pos, head, head_index, head_external, hidden, mahall, clause}].
"""
import argparse, gzip, json, re
from alfiyya_extract import norm, strip_prefix

FUNCS = [
    (r'نائب فاعل', 'نائب فاعل'), (r'(?<!نائب )فاعل', 'فاعل'),
    (r'مبتدأ|مبتدا', 'مبتدأ'), (r'خبر', 'خبر'),
    (r'مفعول (?:به )?(?:اول|أول)', 'مفعول به أول'), (r'مفعول (?:به )?ثان', 'مفعول به ثان'),
    (r'مفعول به|مفعول مقدم|مفعول لفعل|مفعوله', 'مفعول به'), (r'عطف بيان', 'عطف بيان'),
    (r'(?:منصوب|نصب) على (?:التعظيم|نزع الخافض|الاستثناء|المدح|الاختصاص)', 'منصوب'),
    (r'(?<![\u0621-\u064A])اسمها|اسمه(?![\u0621-\u064A])|اسم أن', 'اسم ناسخ'), (r'مفعول مطلق', 'مفعول مطلق'),
    (r'مفعول (?:لاجله|لأجله|له)', 'مفعول لأجله'), (r'مفعول معه', 'مفعول معه'),
    (r'مفعول فيه|ظرف (?:زمان|مكان)|ظرف', 'ظرف'),
    (r'(?<![ء-ي])حال(?![ء-ي])', 'حال'), (r'تمييز', 'تمييز'),
    (r'نعت|صفة', 'نعت'), (r'معطوف', 'معطوف'), (r'بدل', 'بدل'), (r'توكيد', 'توكيد'),
    (r'مضاف (?:إليه|اليه)', 'مضاف إليه'), (r'منادى', 'منادى'), (r'مستثنى', 'مستثنى'),
    (r'اسم (?:كان|إن|ان|ليس|لا|ما|كاد|عسى|لعل|ليت|لكن|كأن|أصبح|اصبح|صار|ظل|بات|أمسى|امسى|أضحى|اضحى|زال|برح|فتئ|انفك|دام)', 'اسم ناسخ'),
    (r'خبر (?:كان|إن|ان|ليس|لا|ما|كاد|عسى|لعل|ليت|لكن|كأن)', 'خبر ناسخ'),
]
POS = [
    (r'جار ومجرور|جاران ومجروران|الجار والمجرور', 'جار ومجرور'), (r'ظرف', 'ظرف'), (r'فعل ماض', 'فعل ماض'),
    (r'فعل مضارع', 'فعل مضارع'), (r'فعل (?:أمر|امر)', 'فعل أمر'),
    (r'اسم موصول', 'اسم موصول'), (r'اسم شرط', 'اسم شرط'), (r'اسم استفهام', 'اسم استفهام'),
    (r'اسم إشارة|اسم اشارة', 'اسم إشارة'), (r'ضمير', 'ضمير'), (r'حرف', 'حرف'),
    (r'(?:أداة|اداة)', 'أداة'),
    (r'(?<![\u0621-\u064A])(?:شرطية|نافية|عاطفة|ناهية|زائدة|مصدرية|استفهامية|جارة|ناصبة|جازمة|موطئة|للتنبيه|للاستئناف|استئنافية|تعليلية)', 'حرف'),
]
HEAD_PATTERNS = [
    r'متعلق(?:ان)? (?:بقوله|ب)\s*"?\s*([ء-ي]+)',
    r'يتعلق(?:ان)? (?:بقوله|ب)\s*"?\s*([ء-ي]+)',
    r'معطوف(?:ان|ات)? على (?:قوله )?"?\s*([ء-ي]+)',
    r'مجرور (?:بقوله|ب)\s*"?\s*([\u0621-\u064A]+)',
    r'(?:نعت|صفة|توكيد|بدل|تمييز) (?:لقوله|ل|من)\s*"?\s*([ء-ي]+)',
    r'حال من (?:الضمير المستتر في |فاعل |قوله )?"?\s*([ء-ي]+)',
    r'مفعول (?:به )?(?:أول |اول |ثان )?(?:لقوله|ل)\s*"?\s*([ء-ي]+)',
    r'(?:فاعل|نائب فاعل|خبر|اسم) (?:قوله )?"?\s*([ء-ي]{2,})',
]
HIDDEN = r'ضمير مستتر فيه (?:جوازا|وجوبا)? ?تقديره (هو|هي|أنت|انت|أنا|انا|نحن|هما|هم)'
MAHALL = r'في محل (رفع|نصب|جر|جزم)'
NOT_HEAD = {'محذوف', 'بمحذوف', 'المبتدأ', 'المبتدا', 'مضاف', 'الضمير', 'الاعراب', 'له', 'لها',
            'مقدم', 'مؤخر', 'مستتر', 'الشرط', 'الجملة', 'ما', 'قوله', 'الذي', 'التي', 'ضمير',
            'المستتر', 'موصول', 'المحذوف', 'ذلك', 'وخبر', 'وهو', 'هو', 'الفاعل', 'المصدر'}


def first_match(table, text):
    best = None
    for pat, lab in table:
        m = re.search(pat, text)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), lab)
    return best[1] if best else None


def segments(note):
    """-> [(quote, gloss)] in order, quote None for leading text."""
    parts = re.split(r'"\s*([^"]*?)\s*"', note)
    out = [(None, parts[0].strip())]
    for i in range(1, len(parts), 2):
        out.append((parts[i].strip(), parts[i + 1].strip() if i + 1 < len(parts) else ''))
    return out


def words_match(qw, vw):
    def eq(a, b):
        a, b = strip_prefix(a), strip_prefix(b)
        return a == b or (abs(len(a) - len(b)) == 1 and min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a)))
    return len(qw) <= len(vw) and all(a == b or eq(a, b) for a, b in zip(qw, vw))


def resolve_head(h, verse_norm, self_span):
    h = norm(h)
    for cand in (h, strip_prefix(h), re.sub(r'^(?:ب|ل|ك)', '', h)):
        if len(cand) < 2 or cand in NOT_HEAD:
            continue
        for i, w in enumerate(verse_norm):
            if i in self_span:
                continue
            w0 = strip_prefix(w)
            for v in {w, w0, re.sub(r'^ال', '', w0), re.sub(r'^(?:بال|لل|كال|ب|ل|ك)', '', w0)}:
                if v == cand or (len(cand) >= 3 and (v.startswith(cand) or cand.startswith(v)) and min(len(v), len(cand)) >= 3):
                    return i
    return None


PARTICLE = r'^(?:الواو|الفاء|اللام|الباء|الكاف|الهمزة|السين|همزة)'

def analyse(gloss):
    prefix = None
    if re.match(PARTICLE, gloss):
        m = re.search(r'[،,]\s*([\u0621-\u064A]+)\s*:\s*', gloss)
        if m:
            prefix, gloss = gloss[:m.start()].strip(), gloss[m.end():]
    e = _analyse(gloss)
    e['prefix'] = prefix
    return e

def _analyse(gloss):
    gloss = re.sub(r'^الآتي\s*', '', gloss)
    gloss = re.sub(r'^[\u0621-\u064A]+\s*:\s*', '', gloss)
    segs = re.split(r'،|,', gloss)
    first = segs[0]
    e = {'pos': first_match(POS, ' '.join(first.split()[:4])), 'function': first_match(FUNCS, first)}
    m = re.search(r'بمحذوف ((?:خبر|نعت|صفة|حال|صلة)[^،]*)', ' '.join(segs[:2]))
    if m:
        e['function'] = first_match(FUNCS, m.group(1)) or ('صلة' if 'صلة' in m.group(1) else None)
        e['head'] = 'محذوف'
    m = re.search(HIDDEN, gloss)
    e['hidden'] = m.group(1) if m else None
    m = re.search(MAHALL, first)
    e['mahall'] = m.group(1) if m else None
    m = re.search(r'والجملة[^،.]*?(في محل (?:رفع|نصب|جر|جزم)[^،.]*|لا محل لها[^،.]*)', gloss)
    e['clause'] = m.group(1).strip() if m else None
    if 'قصد لفظه' in first:
        e['quoted_word'] = True
        e['function'] = first_match(FUNCS, first.split('قصد لفظه')[-1]) or e['function']
    if 'head' not in e:
        scope = first if not re.search(r'جار ومجرور|ظرف', first) else ' '.join(segs[:2])
        ms = [m for m in (re.search(p, scope) for p in HEAD_PATTERNS) if m]
        if ms:
            e['head'] = min(ms, key=lambda m: m.start()).group(1)
    e['alternatives'] = bool(re.search(r'يجوز (?:أن|فيه)|وقيل|أو بدل|ويجوز', gloss))
    if not e['pos'] and re.match(r'متعلق', gloss):
        e['pos'] = 'جار ومجرور'
    e['head_external'] = bool(re.search(r'(?:في|من) البيت السابق|في البيت الذي قبله', gloss))
    return e


def parse(rec):
    verse = rec['verse'].replace('...', ' ... ')
    toks = [re.sub(r'[^\u0621-\u0652\u0670ـ]', '', w) for w in verse.split() if w != '...']
    toks = [w for w in toks if w]
    vnorm = [norm(w) for w in toks]
    entries, ptr = [], 0
    pending_ref = ''
    for q, gloss in segments(rec['irab']):
        if q is None:
            continue
        qw = [norm(w) for w in q.split() if norm(w)]
        start = None
        ref = bool(entries) and re.search(r'(?:قوله|(?:^|\s)(?:ب|ل)|على|من|إلى|الى|في|هو)\s*$', entries[-1]['gloss'])
        if qw and not ref:
            for j in range(ptr, min(ptr + 4, len(vnorm))):
                if words_match(qw, vnorm[j:]):
                    start = j
                    break
        if start is None:  # a reference inside the previous gloss
            if entries:
                entries[-1]['gloss'] += ' "%s" %s' % (q, gloss)
            continue
        entries.append({'span': list(range(start, start + len(qw))), 'quote': q, 'gloss': gloss})
        ptr = start + len(qw)
    for k in range(len(entries) - 2, -1, -1):
        if re.fullmatch(r'[و،,:\s]*', entries[k]['gloss']):
            entries[k]['gloss'] = entries[k + 1]['gloss'].lstrip(': ')
            entries[k]['shared'] = True
    for e in entries:
        e.update(analyse(e['gloss'].lstrip(': ')))
        h = e.get('head')
        e['head_index'] = None
        if h and h != 'محذوف' and not e['head_external']:
            e['head_index'] = resolve_head(h, vnorm, set(e['span']))
    for k, e in enumerate(entries):
        if e['head_index'] is not None or e['head_external']:
            continue
        if e['function'] == 'مضاف إليه' and e['span'][0] > 0:
            e['head_index'] = e['span'][0] - 1
        elif e['function'] == 'خبر' and re.match(r'خبر (?:المبتدأ|المبتدا)', e['gloss']):
            prev = [x for x in entries[:k] if x['function'] == 'مبتدأ']
            if prev:
                e['head_index'] = prev[-1]['span'][0]
    covered = {i for e in entries for i in e['span']}
    tokens = []
    for i, w in enumerate(toks):
        ent = next((k for k, e in enumerate(entries) if i in e['span']), None)
        tokens.append({'i': i, 'word': w, 'entry': ent})
    return {'page': rec['page'], 'kind': rec['kind'], 'id': rec.get('id'), 'verse': rec['verse'],
            'tokens': tokens, 'entries': entries,
            'coverage': round(len(covered) / max(len(toks), 1), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('recs'); ap.add_argument('out')
    a = ap.parse_args()
    out = []
    for line in open(a.recs, encoding='utf-8'):
        r = json.loads(line)
        if r.get('irab'):
            out.append(parse(r))
    with gzip.open(a.out, 'wt', encoding='utf-8') as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    for k in ('alfiyya', 'shahid'):
        rs = [r for r in out if r['kind'] == k]
        nt = sum(len(r['tokens']) for r in rs)
        cov = sum(1 for r in rs for t in r['tokens'] if t['entry'] is not None)
        ents = [e for r in rs for e in r['entries']]
        print(k, 'verses', len(rs), 'tokens', nt, 'covered', cov, '(%.1f%%)' % (100 * cov / max(nt, 1)),
              'entries', len(ents), 'with function', sum(1 for e in ents if e['function']),
              'with pos', sum(1 for e in ents if e['pos']),
              'head named', sum(1 for e in ents if e.get('head')),
              'head resolved', sum(1 for e in ents if e['head_index'] is not None),
              'hidden', sum(1 for e in ents if e['hidden']))


if __name__ == '__main__':
    main()
