"""Build the i'rab training set for training/train_irab.py from open sources:

  MASAQ v5 (CC BY 4.0)    i'rab role of every Qur'an word          -> role
  QAC treebank (GitHub)   function, links, hidden subjects (~40%)  -> func, rel, target, hidden
  Alfiyya (Muhyi al-Din)  function, links, hidden subjects         -> func, rel, target, hidden

Each row is one Qur'an verse or one Alfiyya verse / shahid:
  {"src", "id", "w": [unvowelled words], "role", "func", "rel", "target", "hidden",
   "conj", "part", "subj", "enc"}
The last four are the attached pieces of the word, each with its own function:
  conj  the و/ف prefix (عاطفة، استئنافية، رابطة للجواب …), "none" if absent
  part  another prefix (حرف جر، لام التعليل، أ الاستفهام، س …)
  subj  an attached subject pronoun on a verb (ت، وا، نا …): فاعل، نائب فاعل، اسم كان
  enc   an attached object / possessive pronoun: مفعول به، مضاف إليه، مجرور، اسم إنّ …
Three more word-level fields come from the rest of QAC:
  cm      the case or mood the word carries (QAC morphology, whole Qur'an): رفع، نصب، جر،
          مضارع مرفوع، منصوب، مجزوم; "none" for a word with no case or mood (مبني، حرف، ماض …)
  ptype   the kind of particle or relative the word is (QAC stem tag, whole Qur'an):
          حرف نفي، شرط، حرف مصدري، استفهام … "none" for other words
  clause  the function of the clause or phrase that starts at this word (QAC treebank:
          صلة، شرط، جواب شرط، خبر، مفعول به، صفة، حال …), "none" if it starts none;
          the plain متعلق / جار ومجرور of a prepositional phrase is left out
"none" is a label (the word has no such piece); -1 means unknown.
Labels are ids into labels.json; -1 means "not annotated" and is skipped by the loss.
target: index of the linked word, or len(w)+k for the k-th محذوف slot.

usage: python layer2b/make_dataset.py --masaq MASAQ.csv --qac qac_words.jsonl.gz
           --alfiyya alfiyya_words.jsonl.gz --out DIR
"""
import argparse, collections, csv, gzip, json, re, zlib
from pathlib import Path

SLOTS = ['محذوف خبر', 'محذوف صفة', 'محذوف حال', 'محذوف صلة', 'محذوف غير ذلك']
RELS = ['none', 'تعلق', 'نعت', 'عطف', 'عطف جملة', 'بدل', 'عطف بيان', 'توكيد', 'حال']
VERBS = {'فعل ماض', 'فعل مضارع', 'فعل أمر'}

CONJ = ['none', 'عاطفة', 'استئنافية', 'رابطة للجواب', 'حالية', 'سببية', 'زائدة', 'معية']
PART = ['none', 'جر', 'قسم', 'توكيد', 'استفهام', 'تسوية', 'تعليل', 'استقبال', 'أمر', 'نداء']
SUBJ = ['none', 'فاعل', 'نائب فاعل', 'اسم كان']
ACC_PARTICLES = ['إنّ', 'أنّ', 'كأنّ', 'لكنّ', 'ليت', 'لعلّ']
ENC = ['none', 'مفعول به', 'مفعول به أول', 'مفعول به ثان', 'مضاف إليه', 'مجرور'] + ['اسم ' + p for p in ACC_PARTICLES]
QAC_CONJ = {'CONJ': 'عاطفة', 'REM': 'استئنافية', 'RSLT': 'رابطة للجواب', 'CIRC': 'حالية',
            'CAUS': 'سببية', 'SUP': 'زائدة', 'COM': 'معية'}
QAC_PART = {'EMPH': 'توكيد', 'INTG': 'استفهام', 'EQ': 'تسوية', 'PRP': 'تعليل', 'FUT': 'استقبال',
            'IMPV': 'أمر', 'VOC': 'نداء'}
CM = ['none', 'رفع', 'نصب', 'جر', 'مضارع مرفوع', 'مضارع منصوب', 'مضارع مجزوم']
PTYPE_OF = {'REL': 'اسم موصول', 'NEG': 'حرف نفي', 'ACC': 'حرف ناسخ', 'COND': 'شرط', 'SUB': 'حرف مصدري',
            'RES': 'حصر', 'INTG': 'استفهام', 'CERT': 'تحقيق', 'PRO': 'نهي', 'PREV': 'كافة', 'RET': 'إضراب',
            'EXP': 'استثناء', 'INC': 'ابتداء', 'EXL': 'تفصيل', 'AMD': 'استدراك', 'INT': 'تفسير',
            'FUT': 'استقبال', 'ANS': 'جواب', 'EXH': 'تحضيض', 'SUR': 'فجاءة', 'AVR': 'ردع', 'INL': 'حروف مقطعة',
            'SUP': 'زائد', 'IMPN': 'اسم فعل', 'P': 'حرف جر', 'CONJ': 'حرف عطف'}
PTYPE = ['none'] + sorted(set(PTYPE_OF.values()))
CLAUSE_SKIP = {'متعلق', 'جار ومجرور'}
MASAQ_SUBJ = {'AGNT': 'فاعل', 'PASS_SUBJ': 'نائب فاعل', 'SUBJ_COP_V': 'اسم كان'}
MASAQ_ENC = {'OBJ': 'مفعول به', 'GEN_CONS': 'مضاف إليه', 'PREP_OBJ': 'مجرور', 'SUBJ_COP_PART': 'اسم إنّ'}


def acc_particle(word):
    """إنّ، أنّ، كأنّ … from the unvowelled word that carries the attached pronoun."""
    w = word[1:] if len(word) > 3 and word[0] in 'وف' else word
    for pre, name in (('كأن', 'كأنّ'), ('لكن', 'لكنّ'), ('ليت', 'ليت'), ('لعل', 'لعلّ'), ('إن', 'إنّ'), ('أن', 'أنّ'), ('لأن', 'أنّ'), ('بأن', 'أنّ')):
        if w.startswith(pre):
            return name
    return None

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
for fam, ws in {'إنّ': 'إن إنّ', 'أنّ': 'أن أنّ', 'كأنّ': 'كأن', 'لكنّ': 'لكن', 'ليت': 'ليت', 'لعلّ': 'لعل',
                'كان': 'كان يكن يك يكون تكون الكون ليس أصبح اصبح صار يصير ظل بات أمسى امسى أضحى اضحى زال برح فتئ انفك دام',
                'كاد': 'كاد عسى أوشك اوشك كرب طفق جعل أخذ',
                'لا': 'لا', 'ما': 'ما'}.items():
    for w in ws.split():
        FAMILY[w] = fam
CONTINUITY = set('يزال تزال يزل تزل زال يبرح تبرح برح يفتأ تفتأ فتئ ينفك تنفك انفك دام يزالون'.split())
NOT_NASIKH = {'المبتدأ', 'المبتدا', 'لمبتدأ', 'لمبتدا', 'مبتدأ', 'للمبتدأ', 'مقدم', 'مؤخر', 'ثان', 'عن'}
HINTS = [(r'مقاربة|رجاء|شروع', 'كاد'), (r'ناقص', 'كان'),
         (r'حرف (?:توكيد|تشبيه|استدراك|ترج|تمن)[^،]*نصب|ينصب الاسم', 'ACC'),
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
    for q, g in reversed(earlier[-3:]):  # اسمها / خبرها: look back for the nasikh itself
        for pat, fam in HINTS:
            if re.search(pat, g):
                if fam == 'ACC':           # which of إنّ's sisters: the quoted particle says
                    w = re.sub(r'^[وف](?=..)', '', q.strip())
                    return FAMILY.get(w) if FAMILY.get(w) in ('إنّ', 'أنّ', 'كأنّ', 'لكنّ', 'ليت', 'لعلّ') else None
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


def masaq_clitics(segs):
    """MASAQ marks the attached pieces but not the kind of و/ف (only that there
    is one): conj is then unknown (None), "none" when there is no و/ف."""
    c = {'conj': 'none', 'part': 'none', 'subj': 'none', 'enc': 'none'}
    for sg in segs:
        tag, role, typ = sg['Morph_Tag'], sg['Syntactic_Role'], sg['Morph_Type']
        if typ == 'Prefix':
            if tag == 'CONJ':
                c['conj'] = None
            elif tag == 'PREP':
                c['part'] = 'جر'
            elif tag == 'INTERROG':
                c['part'] = 'استفهام'
            elif tag == 'FUTURE_PART':
                c['part'] = 'استقبال'
        elif typ == 'Suffix':
            if role in MASAQ_SUBJ and c['subj'] == 'none':
                c['subj'] = MASAQ_SUBJ[role]
            elif role in MASAQ_ENC and c['enc'] == 'none':
                c['enc'] = MASAQ_ENC[role]
                if role == 'SUBJ_COP_PART':
                    p = acc_particle(segs[0]['Without_Diacritics'])
                    c['enc'] = 'اسم ' + p if p else None
    return c


def qac_clitics(x, order, v):
    """The attached pieces of a QAC word, from its segments' own tags and edges."""
    c = {'conj': 'none', 'part': 'none', 'subj': 'none', 'enc': 'none'}
    w = int(x['loc'].split(':')[2])
    for si, sg in enumerate(x['segments']):
        tag, rels = sg['tag'], [e['rel'] for e in sg['edges']]
        if sg['kind'] == 'prefix':
            if tag in QAC_CONJ:
                c['conj'] = QAC_CONJ[tag]
            elif tag == 'P':
                c['part'] = 'قسم' if sg['form'] == 'wa' else 'جر'
            elif tag in QAC_PART:
                c['part'] = QAC_PART[tag]
        elif sg['kind'] == 'suffix' and tag == 'PRON':
            for e in sg['edges']:
                r = e['rel']
                if r == 'subj':
                    c['subj'] = 'فاعل'
                elif r == 'pass':
                    c['subj'] = 'نائب فاعل'
                elif r == 'subjx':
                    if x.get('function') in VERBS or any(g['tag'] == 'V' for g in x['segments']):
                        c['subj'] = 'اسم كان'
                    else:
                        h = next((g for g in x['segments'] if g['tag'] == 'ACC'), None)
                        name = {'<in~': 'إنّ', '>an~': 'أنّ', 'ka>an~': 'كأنّ', 'la`kin~': 'لكنّ', 'la`kin': 'لكنّ',
                                'layot': 'ليت', 'laEal~': 'لعلّ'}.get(h.get('lemma') if h else None)
                        c['enc'] = 'اسم ' + name if name else None
                elif r == 'obj' and c['enc'] == 'none':
                    c['enc'] = order.get((v, (w, si)), 'مفعول به')
                elif r == 'poss' and c['enc'] == 'none':
                    c['enc'] = 'مضاف إليه'
                elif r == 'gen' and c['enc'] == 'none':
                    c['enc'] = 'مجرور'
    return c


ALF_CONJ = [(r'عاطفة|حرف عطف|للعطف', 'عاطفة'), (r'استئناف', 'استئنافية'),
            (r'جواب|رابطة|واقعة في', 'رابطة للجواب'), (r'الحال|حالية', 'حالية'),
            (r'سببية|للسببية|للتعليل|تعليلية', 'سببية'), (r'زائدة', 'زائدة'), (r'المعية', 'معية')]
ALF_PART = [(r'^(?:الباء|الكاف|اللام)[^،]*(?:حرف جر|جارة|الجر)', 'جر'), (r'قسم', 'قسم'),
            (r'^اللام[^،]*(?:التعليل|تعليل|كي)', 'تعليل'), (r'^اللام[^،]*(?:الأمر|الامر)', 'أمر'),
            (r'^اللام[^،]*(?:توكيد|ابتداء|المزحلقة)', 'توكيد'), (r'^(?:الهمزة|همزة)[^،]*استفهام', 'استفهام'),
            (r'^(?:الهمزة|همزة)[^،]*تسوية', 'تسوية'), (r'^السين', 'استقبال')]
ALF_PRON = re.compile(r'(?:الهاء|الضمير(?! المستتر)|ضمير (?:الغائب|الغائبة|الغائبين|المتكلم|المخاطب|المتكلمين)|ياء المتكلم|الياء|الكاف|'
                      r'نا(?![\u0621-\u064A])|تاء (?:المتكلم|المخاطب|الفاعل)|التاء|واو الجماعة|الواو ضمير|ألف الاثنين|نون النسوة)'
                      r'([^،.]{0,40}?)(مفعول به أول|مفعول أول|مفعول به ثان|مفعول ثان|مفعول به|مضاف إليه|نائب فاعل|فاعل|اسم إن|اسم كان|في محل جر)')


def alfiyya_clitics(e, word):
    """Only what Muhyi al-Din's note says; everything else stays unknown (None)."""
    c = {'conj': None, 'part': None, 'subj': None, 'enc': None}
    pre = (e.get('prefix') or '').strip()
    if pre:
        if re.match(r'^(?:الواو|الفاء|و|ف)', pre):
            c['conj'] = next((lab for pat, lab in ALF_CONJ if re.search(pat, pre)), None)
        else:
            c['part'] = next((lab for pat, lab in ALF_PART if re.search(pat, pre)), None)
    for m in ALF_PRON.finditer(e['gloss']):
        if 'مستتر' in m.group(1):
            continue
        f = m.group(2)
        if f in ('فاعل', 'نائب فاعل', 'اسم كان'):
            c['subj'] = f
        elif f in ('مفعول به أول', 'مفعول أول'):
            c['enc'] = 'مفعول به أول'
        elif f in ('مفعول به ثان', 'مفعول ثان'):
            c['enc'] = 'مفعول به ثان'
        elif f == 'مفعول به':
            c['enc'] = 'مفعول به'
        elif f == 'مضاف إليه':
            c['enc'] = 'مضاف إليه'
        elif f == 'اسم إن':
            p = acc_particle(word)
            c['enc'] = 'اسم ' + p if p else None
        elif f == 'في محل جر':
            c['enc'] = 'مجرور' if e.get('pos') == 'جار ومجرور' else 'مضاف إليه'
    return c


def qac_morph_words(path):
    """(sura, verse, word) -> (cm, ptype) for every word of the Qur'an, from QAC morphology."""
    segs = collections.defaultdict(list)
    for line in open(path, encoding='utf-8'):
        m = re.match(r'\((\d+):(\d+):(\d+):\d+\)\t[^\t]*\t([^\t]+)\t(\S+)', line)
        if m:
            segs[tuple(map(int, m.groups()[:3]))].append((m.group(4), m.group(5).split('|')))
    out = {}
    for k, ss in segs.items():
        stem = next(((t, f) for t, f in ss if f[0] == 'STEM'), None)
        cm, pt = 'none', 'none'
        if stem:
            tag, f = stem
            pt = PTYPE_OF.get(tag, 'none')
            fs = set(f)
            if 'NOM' in fs: cm = 'رفع'
            elif 'ACC' in fs: cm = 'نصب'
            elif 'GEN' in fs: cm = 'جر'
            elif tag == 'V' and 'IMPF' in fs:
                if 'MOOD:SUBJ' in fs: cm = 'مضارع منصوب'
                elif 'MOOD:JUS' in fs: cm = 'مضارع مجزوم'
                elif any(t == 'EMPH' or 'PRON:3FP' in f or 'PRON:2FP' in f for t, f in ss) or \
                        any(x in fs for x in ('3FP', '2FP')):
                    cm = 'none'                 # نون التوكيد / نون النسوة: مبني
                else: cm = 'مضارع مرفوع'
        out[k] = (cm, pt)
    return out


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
            row.append((segs[0]['Without_Diacritics'], (stems or anyr or [None])[0], masaq_clitics(segs)))
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


TENSE = {'PERF': 'فعل ماض', 'IMPF': 'فعل مضارع', 'IMPV': 'فعل أمر'}


def has_verb(x):
    return any(sg['tag'] == 'V' for sg in x['segments'])


def tense(x):
    return next((TENSE.get(sg.get('aspect')) for sg in x['segments'] if sg['tag'] == 'V'), None)


def qac_labels(x, nw, v, order=None, words=None):
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
    # coordination of clauses (عطف جملة على جملة): a verb joined to a verb, or the head
    # of a clause joined to another clause; the verb keeps its tense as its function
    h = x.get('head') if isinstance(x.get('head'), dict) else {}
    if x.get('rel') == 'conj' and has_verb(x) and words is not None and has_verb(words.get(h.get('word'), {'segments': []})):
        rel, func = 'عطف جملة', tense(x) or func
        hidden = 1 if x.get('hidden') else 0
    elif rel == 'none':
        for cl in x.get('clause') or []:
            if cl['rel'] == 'conj':
                rel, target = 'عطف جملة', idx(cl['head'])
                break
    return func, rel, target, hidden


def alfiyya_rows(path):
    for r in map(json.loads, gzip.open(path, 'rt', encoding='utf-8')):
        n = len(r['tokens'])
        func = [None] * n; rel = [None] * n; target = [None] * n; hidden = [None] * n
        cl = {k: [None] * n for k in ('conj', 'part', 'subj', 'enc')}
        earlier = []
        for e in r['entries']:
            span, first = e['span'], e['span'][0]
            for k, v in alfiyya_clitics(e, r['tokens'][first]['word']).items():
                cl[k][first] = v
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
            earlier.append((e['quote'], e['gloss']))
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
            if pos in VERBS and f == 'معطوف':       # a verb joined to a verb: عطف جملة
                func[first], rel[first], hidden[first] = pos, 'عطف جملة', 1 if e['hidden'] else 0
                target[first] = None if e['head_external'] else h
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
            mj = re.search(r'(?:والجملة|وجملة)[^،.]{0,30}?معطوفة على (?:جملة )?', g)
            if mj and f not in ALF_REL:
                rel[first], target[first] = 'عطف جملة', None
                continue
            rr = ALF_REL.get(f, 'none')
            rel[first] = rr if (f or pos) else None
            if rr != 'none' and not e['head_external']:
                target[first] = h
        words = [t['word'] for t in r['tokens']]
        key = '%s:%s:%s' % (r['kind'], r['page'], r['verse'][:20])
        yield {'src': r['kind'], 'id': key, 'w': words, 'role': [None] * n,
               'func': func, 'rel': rel, 'target': target, 'hidden': hidden, **cl,
               'cm': [None] * n, 'ptype': [None] * n, 'clause': [None] * n}


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

    morph = qac_morph_words(a.qac_morph)
    order = object_order(qac)
    qac_flat = {f"{v[0]}:{v[1]}:{w}": x for v, ws in qac.items() for w, x in ws.items()}
    rows, skipped = [], []
    for v, words in sorted(masaq_verses(a.masaq).items()):
        nw = len(words)
        if qcount[v] != nw:
            skipped.append('%d:%d' % v)
            continue
        row = {'src': 'quran', 'id': '%d:%d' % v, 'w': [w for w, _, _ in words], 'role': [r for _, r, _ in words],
               'func': [None] * nw, 'rel': [None] * nw, 'target': [None] * nw, 'hidden': [None] * nw}
        for k in ('conj', 'part', 'subj', 'enc'):
            row[k] = [c[k] for _, _, c in words]
        row['cm'] = [morph.get((*v, i + 1), (None, None))[0] for i in range(nw)]
        row['ptype'] = [morph.get((*v, i + 1), (None, None))[1] for i in range(nw)]
        row['clause'] = [None] * nw
        for w, x in qac.get(v, {}).items():
            f, rl, t, h = qac_labels(x, nw, v, order, qac_flat)
            i = w - 1
            row['func'][i], row['rel'][i], row['target'][i], row['hidden'][i] = f, rl, t, h
            cf = x.get('clause_function')
            row['clause'][i] = 'none' if cf is None or cf in CLAUSE_SKIP else cf
            for k, val in qac_clitics(x, order, v).items():
                if val is not None and (val != 'none' or row[k][i] is None or k == 'conj'):
                    row[k][i] = val          # QAC's analysis wins where it has one
        rows.append(row)
    rows.extend(alfiyya_rows(a.alfiyya))

    roles, funcs, rels = Vocab(), Vocab(), Vocab(RELS)
    cvoc = {'conj': Vocab(CONJ), 'part': Vocab(PART), 'subj': Vocab(SUBJ), 'enc': Vocab(ENC),
            'cm': Vocab(CM), 'ptype': Vocab(PTYPE)}
    ccnt = collections.Counter(c for r in rows for c in r['clause'] if c)
    clauses = Vocab(['none'] + sorted((c for c, k in ccnt.items() if k >= 10 and c != 'none'), key=lambda c: -ccnt[c]))
    cnt = collections.Counter(f for r in rows for f in r['func'] if f)
    keep = {f for f, c in cnt.items() if c >= 3}
    for r in rows:
        r['role'] = [roles(x) for x in r['role']]
        r['func'] = [funcs(x) if x in keep else -1 for x in r['func']]
        r['rel'] = [rels(x) for x in r['rel']]
        r['target'] = [-1 if t is None else t for t in r['target']]
        r['hidden'] = [-1 if h is None else h for h in r['hidden']]
        for k, voc in cvoc.items():
            r[k] = [voc(x) for x in r[k]]
        r['clause'] = [clauses(x) if x in clauses.idx or x is None else -1 for x in r['clause']]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    split = collections.defaultdict(list)
    for r in rows:
        s = bucket(r['id'])
        split[s].extend([r] * (a.alfiyya_repeat if s == 'train' and r['src'] != 'quran' else 1))
    for s, rs in split.items():
        with gzip.open(out / f'{s}.jsonl.gz', 'wt', encoding='utf-8') as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    json.dump({'roles': roles.items, 'funcs': funcs.items, 'rels': rels.items, 'slots': SLOTS,
               **{k: v.items for k, v in cvoc.items()}, 'clause': clauses.items},
              open(out / 'labels.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    def n(field, src=None):
        return sum(1 for r in rows if src in (None, r['src']) for x in r[field] if x >= 0)
    stats = {'rows': {s: len(rs) for s, rs in split.items()}, 'alfiyya_repeat': a.alfiyya_repeat,
             'by_source': {src: {'units': sum(1 for r in rows if r['src'] == src),
                                 'words': sum(len(r['w']) for r in rows if r['src'] == src),
                                 **{f: n(f, src) for f in ('role', 'func', 'rel', 'target', 'hidden',
                                                           'conj', 'part', 'subj', 'enc',
                                                           'cm', 'ptype', 'clause')}}
                           for src in ('quran', 'alfiyya', 'shahid')},
             'labels': {'roles': len(roles.items), 'funcs': len(funcs.items), 'rels': len(rels.items),
                        'clause': len(clauses.items)},
             'clause_counts': {c: ccnt[c] for c in clauses.items},
             'skipped_verses': skipped}
    json.dump(stats, open(out / 'stats.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
