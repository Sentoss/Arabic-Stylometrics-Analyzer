"""Run one disambiguator over the aligned Qur'an verses and save its top reading
per word.

usage: python run_system.py SYSTEM aligned.jsonl OUT.jsonl [--limit N]
SYSTEM: msa | msa_sup (CAMeL's MSA model, without / with the Classical
supplement) | a folder with one of our morphology models (model.pt, config.json),
optionally followed by +imp (offer imperatives always), +q (offer
question-hamza readings), +lane (dictionary with Lane's stems)
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "layer2"))

KEEP = ["pos", "prc3", "prc2", "prc1", "prc0", "per", "asp", "vox", "mod", "gen", "num",
        "stt", "cas", "enc0", "lex", "diac"]


def build(system):
    from layer2.supplement import ClassicalAnalyzer
    if system in ("msa", "msa_sup"):
        from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
        d = BERTUnfactoredDisambiguator.pretrained("msa", top=1, use_gpu=False)
        if system == "msa_sup":
            d._analyzer = ClassicalAnalyzer(analyzer=d._analyzer, keep_backoff=True, mode="text")
        return d
    from layer2.disambiguator import ClassicalDisambiguator
    # MODEL_DIR[+imp][+q]: also offer imperatives always / question-hamza readings
    model, *opts = system.split("+")
    d = ClassicalDisambiguator(model, top=1)
    if "nocase" in opts:                    # choose readings without case/mood (they come from 2b)
        d.ignore = ("cas", "mod")
    if "lane" in opts:                      # dictionary with Lane's stems
        from camel_tools.morphology.analyzer import Analyzer
        from layer2.lane.lexicon_db import RARE_TRIAL, STEMS, lane_db
        d._analyzer = Analyzer(lane_db(stems=STEMS + ((RARE_TRIAL,) if "rare" in opts else ())), backoff="ADD_PROP")
    d._analyzer = ClassicalAnalyzer(analyzer=d._analyzer, keep_backoff=True, mode="text",
                                    imperatives="always" if "imp" in opts else None,
                                    questions=True if "q" in opts else None)
    return d


def main():
    system, src, dst = sys.argv[1:4]
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    d = build(system)
    t0 = time.time()
    done = set()
    if Path(dst).exists():
        done = {json.loads(l)["loc"] for l in open(dst, encoding="utf-8")}
    with open(dst, "a", encoding="utf-8") as out:
        for k, line in enumerate(open(src, encoding="utf-8")):
            if limit and k >= limit:
                break
            r = json.loads(line)
            if r["loc"] in done:
                continue
            res = d.disambiguate(r["words"])
            preds = []
            for dw in res:
                if dw.analyses:
                    a = dw.analyses[0].analysis
                    preds.append({f: a.get(f) for f in KEEP})
                else:
                    preds.append(None)
            out.write(json.dumps({"loc": r["loc"], "pred": preds}, ensure_ascii=False) + "\n")
            if k % 200 == 0:
                print(k, round(time.time() - t0), "s", flush=True)
    print("done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
