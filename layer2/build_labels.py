"""Build layer 2 training labels from vowelled Classical text (Tashkeela).

For every word, the analyzer lists its possible readings of the bare form;
the vowels printed in the text rule out the readings they contradict. What
survives is the word's label.

usage: python layer2/build_labels.py TASHKEELA.txt OUT.tsv [--limit N_LINES]

Output: one row per word, with a blank line between sentences (CoNLL style):
  sent  idx  written  bare  status  n_readings  n_kept  lex  pos  feats  diac  options

lex/pos/feats/diac give the most probable surviving reading; options lists
every surviving reading (lex~pos~feats, ';'-separated), so training can treat
a tie as "any of these is right" instead of trusting the pick.

status
  unique     every surviving reading has the same lemma, POS and features
  case_tie   survivors differ only in features the text leaves unmarked;
             the most probable one is kept (flagged so training can weight it)
  tie        survivors differ in lemma or POS; the most probable one is kept
  no_match   the vowels contradict every reading (kept bare, no label)
  oov        the analyzer has no reading for the word
"""
import argparse
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from probe_labels import MARK, WORD, compatible  # noqa: E402

from supplement import ClassicalAnalyzer  # noqa: E402

FEATS = ["prc3", "prc2", "prc1", "prc0", "per", "asp", "vox", "mod",
         "gen", "num", "stt", "cas", "enc0"]
LEXICAL = ("lex", "pos")
KEEP = FEATS + ["pos", "lex", "diac", "d3tok", "pos_lex_logprob", "source"]
SECONDARY = ("classical_question", "classical_imperative")


def feats(a):
    return "|".join(f"{f}:{a.get(f, 'na')}" for f in FEATS)


def enc_len(a):
    if a.get("enc0", "0") in ("0", "na") or "_+" not in a.get("d3tok", ""):
        return 0
    return len(MARK.sub("", a["d3tok"].rsplit("_+", 1)[1]))


VOWEL = set("َُِْ")          # fatha, damma, kasra, sukun
FIXED_POS = ("part", "prep", "conj", "pron", "adv", "interj")


def is_fixed(a):
    return a.get("pos", "").startswith(FIXED_POS) or (a.get("pos") == "verb" and a.get("asp") in ("p", "c"))


def built_ending(written, a):
    """A مبنيّ word (perfect or imperative verb, particle, pronoun...) has a fixed
    final vowel that is not a case ending: the text's final vowel or sukun must
    not contradict it. وَقَدْ is the particle قد, not the verb وَقَدَ."""
    if a.get("enc0", "0") not in ("0", "na"):
        return True
    if not is_fixed(a):
        return True
    wm, am = final_marks(written) & VOWEL, final_marks(a["diac"]) & VOWEL
    return not (wm and am) or wm == am


def final_marks(d):
    """The marks on the last letter (sukun kept; CAMeL leaves it out, which
    then simply gives no evidence)."""
    out = set()
    for ch in reversed(d):
        if MARK.match(ch):
            out.add(ch)
        else:
            break
    return out


def letters(d):
    out = []
    for ch in d.replace("ٱ", "ا"):
        if MARK.match(ch):
            if out:
                out[-1][1].add(ch)
        else:
            out.append([ch, set()])
    return out


def sukun_ok(written, a):
    """A sukun the text writes inside the word contradicts a vowel the reading
    has on that letter (بَعْدَ is not the verb بَعُدَ). Long-vowel letters are
    left out (some texts put sukun on them)."""
    w, d = letters(written), letters(a["diac"])
    if [c for c, _ in w] != [c for c, _ in d]:
        return True
    for i, ((c, wm), (_, am)) in enumerate(zip(w[:-1], d[:-1])):
        if i == 0 or (c == "ل" and i == 1 and w[0][0] in "فو"):
            continue                   # بْنُ (dropped hamzat al-waṣl), فَلْيَكْتُبْ (lām al-amr)
        if "ْ" in wm and c not in "اويى" and am & set("َُِ"):
            return False
    return True


def strict(written, a, full):
    """compatible() plus the case vowel, when the text gives one."""
    diac = a["diac"]
    if not compatible(written, diac, full, enc_len(a)):
        return False
    w_last = set(MARK.findall(written[-3:]))
    a_last = set(MARK.findall(diac[-3:]))
    case = set("ًٌٍَُِ")
    return not (w_last & case) or (w_last & case) <= a_last


def label(written, readings):
    bare = MARK.sub("", written)
    full = len(MARK.findall(written)) >= len(bare) - 1
    # final ى may stand for ي (أَبِى); try the text both ways
    spellings = [written] + ([re.sub("ى(?=[ً-ْ]*$)", "ي", written)] if bare.endswith("ى") else [])
    readings = [a for a in readings if any(built_ending(w, a) and sukun_ok(w, a) for w in spellings)]
    # فَقَدَ: the verb's written final vowel beats a particle whose (unwritten)
    # final sukun the text does not show
    strict_k = [a for a in readings if any(strict(w, a, full) for w in spellings)]
    loose_k = [a for a in readings
               if any(compatible(w, a["diac"], full, enc_len(a)) for w in spellings)]
    # CAMeL writes names without a case ending, so خالِدٍ passes only as the
    # adjective; the name of the same letters joins it
    lexes = {MARK.sub("", a["lex"]) for a in strict_k if a["pos"] == "adj"}
    strict_k += [a for a in loose_k if a["pos"] == "noun_prop" and a not in strict_k
                 and MARK.sub("", a["lex"]) in lexes]
    wv = final_marks(written) & set("َُِ")
    if wv and any(is_fixed(a) and final_marks(a["diac"]) & VOWEL == wv for a in loose_k):
        silent = lambda a: is_fixed(a) and not final_marks(a["diac"]) & VOWEL
        strict_k = [a for a in strict_k if not silent(a)]
        loose_k = [a for a in loose_k if not silent(a)]
    # ordinary readings first (strict, then loose); the supplement's rare
    # constructions (interrogative أَ, imperatives) only when nothing else fits:
    # أَرَادَ stays the verb أَراد, not أَ + رادَ; ابْنِ stays "son of", not "build!"
    kept = []
    for group in (strict_k, loose_k):
        kept = [k for k in group if k.get("source") not in SECONDARY]
        if kept:
            break
    if not kept:
        kept = strict_k or loose_k
    if not kept:
        return "no_match", None, 0, ""
    best = max(kept, key=lambda a: a.get("pos_lex_logprob", -99))
    lexical = {tuple(a[k] for k in LEXICAL) for a in kept}
    full_keys = {tuple(a[k] for k in LEXICAL) + (feats(a),) for a in kept}
    status = "unique" if len(full_keys) == 1 else "case_tie" if len(lexical) == 1 else "tie"
    options = sorted({f"{a['lex']}~{a['pos']}~{feats(a)}" for a in kept})
    return status, best, len(kept), ";".join(options)


NAME_NEXT = {"بن", "ابن", "بنت", "ابنة"}                      # خالد بن الوليد / ابن خالد
NAME_AFTER = {"أبو", "أبي", "أبا", "أم", "بنو", "بني", "آل"}   # أبي طالب، بني أمية


def name_context(rows, stats):
    """A tie between a name and a common word (خالد adj/noun_prop) is settled
    as the name next to بن/ابن or after أبو/أم/بني/آل."""
    for j, r in enumerate(rows):
        if r[4] != "tie" or "~noun_prop~" not in r[11]:
            continue
        prev = rows[j - 1][3] if j else ""
        near = {prev, rows[j + 1][3] if j + 1 < len(rows) else ""}
        if near & NAME_NEXT or prev in NAME_AFTER:
            opts = [o for o in r[11].split(";") if o.split("~")[1] == "noun_prop"]
            lex, pos, fs = opts[0].split("~", 2)
            r[4] = "unique" if len(opts) == 1 else "case_tie"
            r[6], r[7], r[8], r[9], r[11] = len(opts), lex, pos, fs, ";".join(opts)
            stats["name_context"] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-passives", action="store_true", help="MSA analyzer without generated passives")
    ap.add_argument("--variants", default=None,
                    help="harvested variants TSV ('none' to disable; default: resources/classical_variants.tsv)")
    ap.add_argument("--lane", action="store_true", help="dictionary with Lane's stems added (lane/lexicon_db.py)")
    ap.add_argument("--lane-rare", action="store_true", help="with --lane: also the rare-words trial file")
    a = ap.parse_args()
    from supplement import VARIANTS
    base = None
    if a.lane:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from layer2.lane.lexicon_db import RARE_TRIAL, STEMS, lane_db
        from camel_tools.morphology.analyzer import Analyzer
        base = Analyzer(lane_db(stems=STEMS + ((RARE_TRIAL,) if a.lane_rare else ())))
    an = ClassicalAnalyzer(analyzer=base, passives=not a.no_passives,
                           variants=None if a.variants == "none" else (a.variants or VARIANTS))
    src = Counter()
    cache_r, cache_l = {}, {}
    stats, n, t0 = Counter(), 0, time.time()
    lines = open(a.src, encoding="utf-8").read().split("\n")
    if a.limit:
        lines = lines[:a.limit]
    with open(a.out, "w", encoding="utf-8") as out:
        out.write("sent\tidx\twritten\tbare\tstatus\tn_readings\tn_kept\t"
                  "lex\tpos\tfeats\tdiac\toptions\n")
        for s, line in enumerate(lines):
            words = WORD.findall(line)
            if not words:
                continue
            rows = []
            for i, w in enumerate(words):
                bare = MARK.sub("", w)
                if bare not in cache_r:
                    forms = [bare] + ([bare[:-1] + "ي"] if bare.endswith("ى") else [])
                    cache_r[bare] = [{k: r.get(k) for k in KEEP} for f in forms
                                     for r in an.analyze(f) if r["pos"] != "abbrev"]
                readings = cache_r[bare]
                if not readings:
                    status, best, k, opts = "oov", None, 0, ""
                else:
                    if w not in cache_l:
                        cache_l[w] = label(w, readings)
                    status, best, k, opts = cache_l[w]
                stats[status] += 1
                if best and str(best.get("source", "")).startswith("classical"):
                    src[best["source"]] += 1
                n += 1
                row = [s, i, w, bare, status, len(readings), k]
                row += ([best["lex"], best["pos"], feats(best), best["diac"]]
                        if best else ["", "", "", ""]) + [opts]
                rows.append(row)
            name_context(rows, stats)
            for row in rows:
                out.write("\t".join(map(str, row)) + "\n")
            out.write("\n")
            if s % 5000 == 0 and s:
                print(f"{s} lines, {n} words, {n / (time.time() - t0):.0f} w/s", flush=True)
    print(f"words: {n}")
    print(f"  ties settled as names: {stats['name_context']}")
    for k in ["unique", "case_tie", "tie", "no_match", "oov"]:
        print(f"  {k:9} {stats[k]:7}  {stats[k] / n:.1%}")
    for k, v in src.items():
        print(f"  label from {k}: {v} ({v / n:.1%})")


if __name__ == "__main__":
    main()
