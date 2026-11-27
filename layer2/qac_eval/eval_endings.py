"""Printed endings, layer 1+2 with and without 2b's predicted ending.
Sets: Tashkeela test segments (prose) and the Qur'an verses in 2b's test split."""
import sys, json, gzip, collections
R = "/home/claude/arabic_stylometrics"
sys.path.insert(0, R); sys.path.insert(0, R + "/layer2")
from restorer.restore import Restorer
from layer2.disambiguator import ClassicalDisambiguator
from layer2.supplement import ClassicalAnalyzer
from layer2.ending_from_irab import ending
from camel_tools.morphology.analyzer import Analyzer
from layer2.lane.lexicon_db import STEMS, lane_db

N = int(sys.argv[1])
L = json.load(open("/home/claude/data_ext/datasets/irab_v5/labels.json"))
rows = [json.loads(l) for l in gzip.open("/home/claude/data_ext/datasets/irab_v5/test.jsonl.gz", "rt")]
sets = {"tashkeela": [r for r in rows if r["src"] == "tashkeela"][:N],
        "quran": [r for r in rows if r["src"] == "quran"]}
LANE = Analyzer(lane_db(stems=STEMS), backoff="ADD_PROP")


def dis():
    d = ClassicalDisambiguator("/home/claude/models/morph_v5")
    d._analyzer = ClassicalAnalyzer(analyzer=LANE, keep_backoff=True, mode="text")
    return d


systems = {"2a endings": Restorer(disambiguator=dis()),
           "2b ending": Restorer(disambiguator=dis(), irab_model="/home/claude/models/irab_v5")}
VOW = {"ضمة", "فتحة", "كسرة", "تنوين ضم", "تنوين فتح", "تنوين كسر"}
res = {}
for sname, rs in sets.items():
    c = collections.Counter()
    for k, r in enumerate(rs):
        outs = {}
        for name, R_ in systems.items():
            _, _, toks = R_.run(" ".join(r["w"]))
            outs[name] = [t for t in toks if t.status != "not_word"]
        if any(len(o) != len(r["w"]) for o in outs.values()):
            c["skipped_segments"] += 1
            continue
        for i, g in enumerate(r["end"]):
            if g < 0:
                continue
            gold = L["end"][g]
            for scope in ("all", "vowel"):
                if scope == "vowel" and gold not in VOW:
                    continue
                c[(scope, "n")] += 1
                for name, o in outs.items():
                    e = ending(o[i].diac or o[i].restored or "")
                    c[(scope, name)] += e == gold
        if k % 200 == 0:
            print(sname, k, flush=True)
    res[sname] = {f"{a}|{b}": v for (a, b), v in ((k if isinstance(k, tuple) else (k, "")) for k in c)} if False else {
        ("%s|%s" % k if isinstance(k, tuple) else k): v for k, v in c.items()}
    print(sname, res[sname], flush=True)
json.dump(res, open(sys.argv[2] if len(sys.argv) > 2 else "eval_endings.json", "w"), ensure_ascii=False, indent=1)
