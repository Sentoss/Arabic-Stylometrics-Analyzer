"""Layer 2a + 2b together: 2a's reading (chosen ignoring case/mood), with the case or
mood that the iʿrāb model gives (layer2/case_from_irab.py).

usage: python run_combined.py MORPH_SYSTEM IRAB_DIR aligned.jsonl OUT.jsonl
MORPH_SYSTEM as in run_system.py (e.g. models/morph_v5+lane+nocase).
Each output line: {"loc", "pred" (combined), "pred_2a" (2a alone), "cm" (2b), "irab_test"}
irab_test: the verse is in the iʿrāb model's own test split (it never trained on it).
"""
import json
import re
import sys
import time
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "layer2b"))
from run_system import KEEP, build  # noqa: E402
from layer2.case_from_irab import pick  # noqa: E402

VOWELS = re.compile(r"[ً-ْٰـ]")


def main():
    system, irab_dir, src, dst = sys.argv[1:5]
    from irab import IrabTagger
    d = build(system)
    d.top = 100000                       # keep every analysis: the case variants are needed
    t = IrabTagger(irab_dir)
    t0 = time.time()
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    with open(dst, "w", encoding="utf-8") as out:
        for k, r in enumerate(rows):
            res = d.disambiguate(r["words"])
            ib = t.tag([VOWELS.sub("", w) for w in r["words"]])
            cms = [x.get("cm", "none") if "role" in x else None for x in ib]
            p2a, pc = [], []
            for dw, cm in zip(res, cms):
                if not dw.analyses:
                    p2a.append(None); pc.append(None)
                    continue
                a2, _ = dw.analyses[0].analysis, None
                a, _ = pick(dw.analyses, cm)
                p2a.append({f: a2.get(f) for f in KEEP})
                pc.append({f: a.get(f) for f in KEEP})
            out.write(json.dumps({"loc": r["loc"], "pred": pc, "pred_2a": p2a, "cm": cms,
                                  "irab_test": zlib.crc32(r["loc"].encode()) % 10 == 0},
                                 ensure_ascii=False) + "\n")
            if k % 100 == 0:
                print(k, round(time.time() - t0), "s", flush=True)
    print("done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
