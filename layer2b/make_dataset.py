"""Build the i'rab training set for training/train_irab.py from open sources:

  MASAQ v5 (CC BY 4.0)    i'rab role of every Qur'an word          -> role
  QAC treebank (GitHub)   function, links, hidden subjects (~40%)  -> func, rel, target, hidden
  Alfiyya (Muhyi al-Din)  function, links, hidden subjects         -> func, rel, target, hidden

Each row is one Qur'an verse or one Alfiyya verse / shahid:
  {"src", "id", "w": [unvowelled words], "role", "func", "rel", "target", "hidden"}
Labels are ids into labels.json; -1 means "not annotated" and is skipped by the loss.
target: index of the linked word, or len(w)+k for the k-th محذوف slot.

usage: python layer2b/make_dataset.py --masaq MASAQ.csv --qac qac_words.jsonl.gz
           --alfiyya alfiyya_words.jsonl.gz --out DIR
"""
import argparse, collections, csv, gzip, json, re, zlib
from pathlib import Path

SLOTS = ['محذوف خبر', 'محذوف صفة', 'محذوف حال', 'محذوف صلة', 'محذوف غير ذلك']
RELS = ['none', 'تعلق', 'نعت', 'عطف', 'بدل', 'عطف بيان', 'توكيد', 'حال']
VERBS = {'فعل ماض', 'فعل مضارع', 'فعل أمر'}

QAC_REL = {'adj': 'نعت', 'conj': 'عطف', 'app': 'بدل', 'emph': 'توكيد', 'circ': 'حال'}

ALF_FUNC = {'نعت': 'صفة',
            'مفعول لأجله': 'المفعول لأجله', 'مفعول معه': 'المفعول معه',
            'منادى': 'منادي', 'مستثنى': 'مستثني', 'مجرور': 'جار ومجرور'}
ALF_DROP = {'منصوب', 'صلة', 'ظرف', None}
ALF_REL = {'صفة': 'نعت', 'معطوف': 'عطف', 'بدل': 'بدل', 'عطف بيان': 'عطف بيان', 'توكيد': 'توكيد', 'حال': 'حال'}
SLOT_OF = {'خبر': 0, 'نعت': 1, 'حال': 2, 'صلة': 3}
PARTICLE_FUNC = [(r'^(?:حرف )?نافية|^حرف نفي', 'نفي'), (r'^(?:حرف )?ناهية|^لا الناهية', 'نهي'),
                 (r'^حرف تحقيق|^تحقيق', 'تحقيق'), (r'^حرف استفهام|^(?:همزة )?الاستفهام', 'استفهام')]

# the nasikh families are kept apart: اسم كان (مرفوع) is not اسم إنّ (منصوب)
FAMILY = {}
for fam, ws in {'إنّ': 'إن ان أن إنّ أنّ كأن لكن ليت لعل',
                'كان': 'كان يكن يك يكون تكون الكون ليس أصبح اصبح صار يصير ظل بات أمسى امسى أضحى اضحى زال برح فتئ انفك دام',
                'كاد': 'كاد عسى أوشك اوشك كرب طفق جعل أخذ',
                'لا': 'لا', 'ما': 'ما'}.items():
    for w in ws.split():
        FAMILY[w] = fam
CONTINUITY = set('يزال تزال يزل تزل زال يبرح تبرح برح يفتأ تفتأ فتئ ينفك تنفك انفك دام يزالون'.split())
NOT_NASIKH = {'المبتدأ', 'المبتدا', 'لمبتدأ', 'لمبتدا', 'مبتدأ', 'للمبتدأ', 'مقدم', 'مؤخر', 'ثان', 'عن'}
HINTS = [(r'مقاربة|رجاء|شروع', 'كاد'), (r'ناقص', 'كان'),
         (r'حرف (?:توكيد|تشبيه|استدراك|ترج|تمن)[^،]*نصب|ينصب الاسم', 'إنّ'),
         (r'نافية للجنس', 'لا'), (r'حجازية|تعمل عمل ليس', 'ما')]


def nasikh_family(gloss, earlier):
    m = re.match(r'(?:اسم|خبر)\s+([\u0621-\u064A]+)(?:\s+([\u0621-\u064A]+))?', gloss)
    if m:
        w, nxt = m.group(1), m.group(2)
        if w in ('لا', 'ما') and nxt in CONTINUITY:      # خبر لا يزال، اسم ما دام
            return 'كان'
        for cand in (w, w[1:] if w.startswith('ل') else None):
            if cand in FAMILY:
                return FAMILY[cand]
        if w in NOT_NASIKH:
            return None
    if not re.match(r'(?:اسم|خبر)(?:ها|ه)(?![\u0621-\u064A])', gloss):
        return None
    for g in reversed(earlier[-3:]):     # اسمها / خبرها: look back for the nasikh itself
        for pat, fam in HINTS:
            if re.search(pat, g):
                return fam
    return None


def bucket(key):
    h = zlib.crc32(key.encode()) % 10
    return 'test' if h == 0 else 'val' if h == 1 else 'train'


class Vocab:
    def __init__(self, fixed=None):
        self.items = list(fixed or [])
        self.idx = {x: i for i, x in enumerate(self.items)}
        self.frozen = fixed is not None

    def __call__(self, x):
        if x is None:
            return -1
        if x not in self.idx:
            if self.frozen:
                raise KeyError(x)
            self.idx[x] = len(self.items)
            self.items.append(x)
        return self.idx[x]


def masaq_verses(path):
    words = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(path, encoding='utf-8-sig')):
        words[(int(r['Sura_No']), int(r['Verse_No']))][int(r['Word_No'])].append(r)
    out = {}
    for k, ws in words.items():
        row = []
        for n in sorted(ws):
            segs = ws[n]
            stems = [s['Syntactic_Role'] for s in segs if s['Morph_Type'] == 'Stem' and s['Syntactic_Role']]
            anyr = [s['Syntactic_Role'] for s in segs if s['Syntactic_Role']]
            row.append((segs[0]['Without_Diacritics'], (stems or anyr or [None])[0]))
        out[k] = row
    return out


def object_order(qac):
    """For verbs with exactly two objects in QAC (أعطى، ظنّ، جعل، اتخذ…), map each
    object (word, seg) to 'أول' / 'ثان' by position. Pairs where one object is a
    detached إيّا (اشتغال, as in وإياي فارهبون) are left out."""
    objs = collections.defaultdict(list)
    for v, ws in qac.items():
        for w, x in ws.items():
            for si, seg in enumerate(x['segments']):
                for e in seg['edges']:
                    h = e.get('head')
                    if e['rel'] == 'obj' and isinstance(h, dict) and 'word' in h:
                        objs[(h['word'], h['seg'])].append(((w, si), seg.get('lemma')))
    order = {}
    for head, deps in objs.items():
        if len(deps) == 2 and not any(l and l.startswith('<iy~') for _, l in deps):
            (a, _), (b, _) = sorted(deps)
            v = tuple(map(int, head[0].split(':')[:2]))
            order[(v, a)], order[(v, b)] = 'مفعول به أول', 'مفعول به ثان'
    return order


def qac_labels(x, nw, v, order=None):
    """-> func, rel, target, hidden for one QAC word (target may be None)."""
    func = x.get('function')
    if func == 'متعلق':
        func = None
    if func == 'مفعول به' and order:
        w = int(x['loc'].split(':')[2])
        stem = next((i for i, sg in enumerate(x['segments']) if sg['kind'] == 'stem'), 0)
        func = order.get((v, (w, stem)), func)
    rel, target = 'none', None
    is_verb = func in VERBS
    hidden = (1 if x.get('hidden') else 0) if is_verb else None

    def idx(h):
        if isinstance(h, dict) and 'word' in h:
            s, vv, w = map(int, h['word'].split(':'))
            if (s, vv) == v and 1 <= w <= nw:
                return w - 1
        return None

    pp = x.get('pp_head')
    if pp is not None:
        rel = 'تعلق'
        if 'word' in pp:
            target = idx(pp)
        elif pp.get('omitted') == 'V':
            target = nw + 4
    elif x.get('rel') in QAC_REL:
        rel = QAC_REL[x['rel']]
        if rel != 'حال':      # QAC attaches حال to its verb, not to صاحب الحال
            target = idx(x.get('head'))
    return func, rel, target, hidden


def alfiyya_rows(path):
    for r in map(json.loads, gzip.open(path, 'rt', encoding='utf-8')):
        n = len(r['tokens'])
        func = [None] * n; rel = [None] * n; target = [None] * n; hidden = [None] * n
        earlier = []
        for e in r['entries']:
            span, first = e['span'], e['span'][0]
            pos, f = e['pos'], e['function']
            g = re.sub(r'^[\u0621-\u064A]+\s*:\s*', '', e['gloss'].lstrip(': '))
            if e.get('prefix'):
                g = re.sub(r'^.*?[\u0621-\u064A]+\s*:\s*', '', g, count=1) if ':' in g else g
            if f in ('اسم ناسخ', 'خبر ناسخ', 'خبر'):
                fam = nasikh_family(g, earlier)
                if f == 'خبر' and fam is None:
                    pass                      # خبر المبتدأ
                elif fam is None:
                    f = None                  # a nasikh we cannot name: leave unlabelled
                else:
                    f = ('اسم ' if f == 'اسم ناسخ' else 'خبر ') + fam
            earlier.append(e['gloss'])
            f = ALF_FUNC.get(f, f)
            h = e['head_index']
            slot = None
            if e.get('head') == 'محذوف':
                slot = n + SLOT_OF.get(e['function'], 4)
            if pos == 'جار ومجرور':
                for i in span:
                    func[i], rel[i] = 'جار ومجرور', 'none'
                rel[first] = 'تعلق'
                target[first] = slot if slot is not None else h
                continue
            if pos == 'ظرف' or f == 'ظرف':
                zm = 'ظرف زمان' if re.search(r'زمان|للزمان', g) else 'ظرف مكان' if re.search(r'مكان|للمكان', g) else None
                func[first], rel[first] = zm, 'تعلق'
                target[first] = slot if slot is not None else h
                continue
            if pos in VERBS and (f is None or f in ALF_DROP):
                f = pos
            if pos in VERBS:
                hidden[first] = 1 if e['hidden'] else 0
            if f is None and pos == 'حرف':
                for pat, lab in PARTICLE_FUNC:
                    if re.search(pat, g):
                        f = lab
                        break
            if f in ALF_DROP:
                f = None
            func[first] = f
            rr = ALF_REL.get(f, 'none')
            rel[first] = rr if (f or pos) else None
            if rr != 'none' and not e['head_external']:
                target[first] = h
        words = [t['word'] for t in r['tokens']]
        key = '%s:%s:%s' % (r['kind'], r['page'], r['verse'][:20])
        yield {'src': r['kind'], 'id': key, 'w': words, 'role': [None] * n,
               'func': func, 'rel': rel, 'target': target, 'hidden': hidden}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--masaq', required=True); ap.add_argument('--qac', required=True)
    ap.add_argument('--alfiyya', required=True); ap.add_argument('--qac-morph', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--alfiyya-repeat', type=int, default=3,
                    help='copies of each Alfiyya/shahid row in train (they are the only non-Qur\'anic data)')
    a = ap.parse_args()

    # QAC word counts per verse, to skip the few verses where MASAQ's numbering differs
    qcount = collections.Counter()
    for line in open(a.qac_morph, encoding='utf-8'):
        m = re.match(r'\((\d+):(\d+):(\d+):\d+\)', line)
        if m:
            s, v, w = map(int, m.groups()); qcount[(s, v)] = max(qcount[(s, v)], w)
    qac = {}
    for x in map(json.loads, gzip.open(a.qac, 'rt', encoding='utf-8')):
        s, v, w = map(int, x['loc'].split(':'))
        qac.setdefault((s, v), {})[w] = x

    order = object_order(qac)
    rows, skipped = [], []
    for v, words in sorted(masaq_verses(a.masaq).items()):
        nw = len(words)
        if qcount[v] != nw:
            skipped.append('%d:%d' % v)
            continue
        row = {'src': 'quran', 'id': '%d:%d' % v, 'w': [w for w, _ in words], 'role': [r for _, r in words],
               'func': [None] * nw, 'rel': [None] * nw, 'target': [None] * nw, 'hidden': [None] * nw}
        for w, x in qac.get(v, {}).items():
            f, rl, t, h = qac_labels(x, nw, v, order)
            i = w - 1
            row['func'][i], row['rel'][i], row['target'][i], row['hidden'][i] = f, rl, t, h
        rows.append(row)
    rows.extend(alfiyya_rows(a.alfiyya))

    roles, funcs, rels = Vocab(), Vocab(), Vocab(RELS)
    cnt = collections.Counter(f for r in rows for f in r['func'] if f)
    keep = {f for f, c in cnt.items() if c >= 3}
    for r in rows:
        r['role'] = [roles(x) for x in r['role']]
        r['func'] = [funcs(x) if x in keep else -1 for x in r['func']]
        r['rel'] = [rels(x) for x in r['rel']]
        r['target'] = [-1 if t is None else t for t in r['target']]
        r['hidden'] = [-1 if h is None else h for h in r['hidden']]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    split = collections.defaultdict(list)
    for r in rows:
        s = bucket(r['id'])
        split[s].extend([r] * (a.alfiyya_repeat if s == 'train' and r['src'] != 'quran' else 1))
    for s, rs in split.items():
        with gzip.open(out / f'{s}.jsonl.gz', 'wt', encoding='utf-8') as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    json.dump({'roles': roles.items, 'funcs': funcs.items, 'rels': rels.items, 'slots': SLOTS},
              open(out / 'labels.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    def n(field, src=None):
        return sum(1 for r in rows if src in (None, r['src']) for x in r[field] if x >= 0)
    stats = {'rows': {s: len(rs) for s, rs in split.items()}, 'alfiyya_repeat': a.alfiyya_repeat,
             'by_source': {src: {'units': sum(1 for r in rows if r['src'] == src),
                                 'words': sum(len(r['w']) for r in rows if r['src'] == src),
                                 **{f: n(f, src) for f in ('role', 'func', 'rel', 'target', 'hidden')}}
                           for src in ('quran', 'alfiyya', 'shahid')},
             'labels': {'roles': len(roles.items), 'funcs': len(funcs.items), 'rels': len(rels.items)},
             'skipped_verses': skipped}
    json.dump(stats, open(out / 'stats.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
