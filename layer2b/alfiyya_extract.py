"""Extract Alfiyya verses, shawahid and Muhyi al-Din's i'rab footnotes from
Shamela book 9904 (Ibn Aqil's sharh with Minhat al-Jalil).

Input: a directory of downloaded pages p<N>.html.
Output: JSONL records {kind: alfiyya|shahid, page, verse, irab, note}.
"""
import argparse, glob, html, json, os, re

DIG = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
AR_DIGITS = '٠-٩'

def clean(t):
    t = re.sub(r'<a href="#p\d+".*?</a>', '', t, flags=re.S)
    t = re.sub(r'<br\s*/?>', ' ', t)
    t = re.sub(r'<[^>]+>', '', t)
    return re.sub(r'\s+', ' ', html.unescape(t)).strip()

def read_page(path):
    s = open(path, encoding='utf-8').read()
    a = s.find('class="nass'); b = s.find('id="appended_pages"')
    if a < 0:
        return None
    body = s[a:b]
    num = int(re.search(r'data-page-id="(\d+)"', body).group(1))
    main, _, foot = body.partition('<hr>')
    paras = [clean(p) for p in re.findall(r'<p[^>]*>(.*?)</p>', main, re.S)]
    foot = ' '.join(clean(p) for p in re.findall(r'<p class="hamesh">(.*?)</p>', foot, re.S))
    return num, [p for p in paras if p], foot

NOTE_START = re.compile(r'(?:^|(?<=[\s.:،؛)]))(?:\(([%s]+)\)|([%s]+) - )' % (AR_DIGITS, AR_DIGITS))

def notes_stream(pages):
    """Join footnotes across pages ('=' marks continuation) and split them.
    -> list of dicts {page, n, shahid, text}."""
    notes = []
    for num, _, foot in pages:
        foot = foot.strip()
        cont = foot.startswith('=')
        if cont:
            foot = foot.lstrip('= ').strip()
        pos = 0; ms = list(NOTE_START.finditer(foot))
        head = foot[:ms[0].start()] if ms else foot
        if head.strip():
            if notes and cont:
                notes[-1]['text'] = notes[-1]['text'].rstrip(' =') + ' ' + head.strip()
                notes[-1]['pages'].append(num)
            else:
                notes.append({'page': num, 'pages': [num], 'n': None, 'shahid': None, 'text': head.strip()})
        for i, m in enumerate(ms):
            end = ms[i + 1].start() if i + 1 < len(ms) else len(foot)
            notes.append({'page': num, 'pages': [num],
                          'n': int(m.group(1).translate(DIG)) if m.group(1) else None,
                          'shahid': int(m.group(2).translate(DIG)) if m.group(2) else None,
                          'text': foot[m.end():end].strip()})
    for n in notes:
        n['text'] = n['text'].rstrip(' =').strip()
    return notes

NORM = str.maketrans({'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ٱ': 'ا', 'ى': 'ي', 'ة': 'ه', 'ؤ': 'و', 'ئ': 'ي', 'ء': ''})
def norm(w):
    w = re.sub(r'[ً-ْٰـ]', '', w)
    w = re.sub(r'[^ء-ي]', '', w)
    return w.translate(NORM)

def quotes(text):
    return [q.strip() for q in re.findall(r'"\s*([^"]+?)\s*"', text)]

def strip_prefix(w):
    return re.sub(r'^(?:و|ف)(?=..)', '', w)

def first_quote_matches(note, verse):
    q = quotes(note[:120])
    if not q or not note.lstrip().startswith('"'):
        return False
    vw = [norm(w) for w in verse.replace('...', ' ').split()]
    qw = [norm(w) for w in q[0].split()]
    return bool(qw) and vw[:len(qw)] == qw

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pages'); ap.add_argument('out')
    a = ap.parse_args()
    pages = [r for r in (read_page(f) for f in glob.glob(os.path.join(a.pages, 'p*.html'))) if r]
    pages.sort()
    notes = notes_stream(pages)
    by_page = {}
    for n in notes:
        by_page.setdefault(n['page'], []).append(n)
    recs = []; used = set()
    for num, paras, _ in pages:
        for p in paras:
            if ' ... ' not in p:
                continue
            p = re.sub(r'\s*\([%s]+\)\s*' % AR_DIGITS, ' ', p).strip()
            m = re.match(r'([%s]+)\s*-\s*(.*)' % AR_DIGITS, p)
            if m:  # shahid
                sid = int(m.group(1).translate(DIG))
                cand = [n for pg in (num, num + 1, num + 2) for n in by_page.get(pg, []) if n['shahid'] == sid]
                note = cand[0] if cand else None
                irab = None
                if note:
                    used.add(id(note))
                    k = note['text'].find('الاعراب:')
                    if k >= 0:
                        irab = note['text'][k + len('الاعراب:'):]
                        irab = re.split(r'\s(?:الشاهد فيه|الشاهد فيهما|الشاهد:)', irab)[0].strip()
                recs.append({'kind': 'shahid', 'id': sid, 'page': num, 'verse': m.group(2).strip(),
                             'irab': irab, 'note': note['text'] if note else None})
            else:
                recs.append({'kind': 'alfiyya', 'page': num, 'verse': p, 'irab': None})
    # match i'rab notes (they open with a quoted word) to verse lines by word overlap
    verses = [r for r in recs if r['kind'] == 'alfiyya']
    pairs = []
    for n in notes:
        if id(n) in used or not n['text'].lstrip().startswith('"'):
            continue
        qw = set()
        for q in quotes(n['text']):
            for w in q.split():
                qw.add(norm(w)); qw.add(strip_prefix(norm(w)))
        for i, r in enumerate(verses):
            if not (n['page'] - 1 <= r['page'] <= n['page']):
                continue
            vw = [norm(w) for w in r['verse'].replace('...', ' ').split() if norm(w)]
            hit = sum(1 for w in vw if w in qw or strip_prefix(w) in qw)
            pairs.append((hit / max(len(vw), 1), -abs(n['page'] - r['page']), id(n), i, n))
    pairs.sort(key=lambda x: (-x[0], -x[1]))
    taken_v = set()
    for score, _, nid, i, n in pairs:
        if score < 0.5 or nid in used or i in taken_v:
            continue
        used.add(nid); taken_v.add(i)
        verses[i]['irab'] = n['text']; verses[i]['match'] = round(score, 2)
    # unnumbered shawahid: match remaining lines to unused shahid notes' i'rab
    def irab_part(t):
        k = t.find('الاعراب:')
        if k < 0:
            return None
        return re.split(r'\s(?:الشاهد فيه|الشاهد فيهما|الشاهد:)', t[k + len('الاعراب:'):])[0].strip()
    pairs = []
    for n in notes:
        if id(n) in used or n['shahid'] is None:
            continue
        ir = irab_part(n['text'])
        if not ir:
            continue
        qw = {norm(w) for q in quotes(ir) for w in q.split()}
        qw |= {strip_prefix(w) for w in qw}
        for i, r in enumerate(verses):
            if r['irab'] or not (n['page'] - 1 <= r['page'] <= n['page']):
                continue
            vw = [norm(w) for w in r['verse'].replace('...', ' ').split() if norm(w)]
            hit = sum(1 for w in vw if w in qw or strip_prefix(w) in qw)
            pairs.append((hit / max(len(vw), 1), id(n), i, n, ir))
    pairs.sort(key=lambda x: -x[0])
    for score, nid, i, n, ir in pairs:
        r = verses[i]
        if score < 0.5 or nid in used or r['irab']:
            continue
        used.add(nid)
        r.update(kind='shahid', id=n['shahid'], irab=ir, note=n['text'], match=round(score, 2))
    with open(a.out, 'w', encoding='utf-8') as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    for k in ('alfiyya', 'shahid'):
        rs = [r for r in recs if r['kind'] == k]
        print(k, len(rs), 'with irab', sum(1 for r in rs if r['irab']))
    print('pages', len(pages), 'notes', len(notes))

if __name__ == '__main__':
    main()
