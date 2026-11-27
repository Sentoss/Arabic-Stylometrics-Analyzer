"""Gold training labels for layer 2a from the Quranic Arabic Corpus morphology.

For every Qur'an word aligned 1:1 with the Wikisource text (align.py), the
dictionary readings (CAMeL + Classical supplement) are kept only if they agree
with QAC on every feature QAC defines (POS class, proper noun, aspect, voice,
mood, person, gender, number, case, و/ف, preposition clitic, article, question
hamza, pronoun suffix). The Wikisource vowels then narrow them further, as for
Tashkeela (build_labels.label). Unlike Tashkeela, QAC settles what vowels
can't: ما relative vs negative, imperatives vs perfects, من conditional vs
relative.

Fallbacks, in order: gold-consistent readings that the vowels reject are kept
anyway (Uthmani vs imla'i conventions); if no reading fits case or mood, those
two are relaxed; if still nothing fits, the word is left unlabelled (context
only).

Output: segments in the format of layer2/make_dataset.py, with the value ids
of an existing labels.json (so the model stays compatible), split by sūra:
  test  sūras with number % 10 == 3   (3, 13, 23, … 113)
  val   sūras with number % 10 == 7
  train the rest

usage: python layer2/qac_eval/qac_labels.py ALIGNED.jsonl LABELS.json OUT_DIR [--lane]
"""
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from score import compare  # noqa: E402
from build_labels import KEEP, label  # noqa: E402
from supplement import ClassicalAnalyzer, VARIANTS  # noqa: E402

FEATS = ["pos", "prc3", "prc2", "prc1", "prc0", "per", "asp", "vox", "mod", "gen", "num", "stt", "cas", "enc0"]
SKIP = ("ALL",)
# QAC tags whose CAMeL part of speech is clear; used to narrow the readings
# further when that leaves any (e.g. negative ما → part_neg only)
FINE_POS = {"NEG": {"part_neg"}, "REL": {"pron_rel"}, "INTG": {"part_interrog", "pron_interrog", "adv_interrog"},
            "PN": {"noun_prop"}, "DEM": {"pron_dem"}, "PRON": {"pron"}, "V": {"verb"},
            "VOC": {"part_voc"}, "FUT": {"part_fut"}, "SUB": {"conj_sub"}, "CONJ": {"conj"}}


def agrees(gold, reading, relax=()):
    r = compare(gold, reading)
    return all(v for k, v in r.items()
               if k not in SKIP and not k.endswith(("_left_open", "_when_decided")) and k not in relax)


def split_of(sura):
    return "test" if sura % 10 == 3 else "val" if sura % 10 == 7 else "train"


def main(aligned, labels_json, out_dir, max_words=100):
    vocab = json.load(open(labels_json, encoding="utf-8"))["values"]
    idx = {k: {v: i for i, v in enumerate(vocab[k])} for k in FEATS}
    base = None
    if "--lane" in sys.argv:                     # dictionary with Lane / Arramooz
        sys.path.insert(0, str(HERE.parents[1]))
        from camel_tools.morphology.analyzer import Analyzer
        from layer2.lane.lexicon_db import lane_db
        base = Analyzer(lane_db())
    an = ClassicalAnalyzer(analyzer=base, passives=True, variants=VARIANTS)
    cache = {}
    stats = Counter()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {s: gzip.open(out / f"qac_{s}.jsonl.gz", "wt", encoding="utf-8") for s in ("train", "val", "test")}
    words_per = Counter()
    for line in open(aligned, encoding="utf-8"):
        r = json.loads(line)
        sura = int(r["loc"].split(":")[0])
        split = split_of(sura)
        targets = []
        for bare, diac, gold in zip(r["words"], r["diac"], r["gold"]):
            if not gold:
                targets.append(None)
                stats["not aligned 1:1"] += 1
                continue
            if bare not in cache:
                forms = [bare] + ([bare[:-1] + "ي"] if bare.endswith("ى") else [])
                cache[bare] = [{k: a.get(k) for k in KEEP} for f in forms
                               for a in an.analyze(f) if a["pos"] != "abbrev"]
            readings = cache[bare]
            kept, how = [], None
            for relax, name in (((), "gold"), (("cas", "mod"), "gold, case/mood relaxed")):
                ok = [a for a in readings if agrees(gold, a, relax)]
                fine = FINE_POS.get(gold["qac_tag"])
                if fine and any(a["pos"] in fine for a in ok):
                    ok = [a for a in ok if a["pos"] in fine]
                if ok:
                    status, _, _, opts = label(diac, ok, secondary=False)
                    if status != "no_match":
                        kept = [o for o in opts.split(";")]
                        how = name + " + vowels"
                    else:
                        kept = sorted({f"{a['lex']}~{a['pos']}~" + "|".join(f"{f}:{a.get(f, 'na')}" for f in FEATS[1:])
                                       for a in ok})
                        how = name + " (vowels ignored)"
                    break
            if not kept:
                targets.append(None)
                stats["no reading agrees with QAC" if readings else "oov"] += 1
                continue
            stats[how] += 1
            allowed = defaultdict(set)
            for opt in kept:
                lex, pos, feats = opt.split("~", 2)
                d = {"pos": pos}
                for kv in feats.split("|"):
                    k, v = kv.split(":", 1)
                    d[k] = v
                for k in FEATS:
                    v = d.get(k)
                    if v in idx[k]:
                        allowed[k].add(idx[k][v])
            targets.append([sorted(allowed[k]) for k in FEATS])
        words = r["words"]
        for s in range(0, len(words), max_words):
            files[split].write(json.dumps({"w": words[s:s + max_words], "t": targets[s:s + max_words],
                                           "loc": r["loc"]}, ensure_ascii=False, separators=(",", ":")) + "\n")
        words_per[split] += len(words)
    for f in files.values():
        f.close()
    n = sum(stats.values())
    for k, v in stats.most_common():
        print(f"  {k:32} {v:6}  {v / n:.1%}")
    print("words per split:", dict(words_per))


if __name__ == "__main__":
    main(*sys.argv[1:4])
