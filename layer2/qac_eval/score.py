"""Score system predictions against QAC gold features.

usage: python score.py aligned.jsonl SYS1=pred1.jsonl [SYS2=pred2.jsonl ...] [--json OUT] [--heldout]
Only verses present in every prediction file are scored. --heldout: only the
sūras held out from training by qac_labels.py (number % 10 == 3), for models
trained on QAC.
"""
import json
import sys
from collections import Counter, defaultdict

NOMINAL = {"noun", "noun_prop", "noun_quant", "noun_num", "adj", "adj_comp", "adj_num"}
PRONOM = {"pron", "pron_dem", "pron_rel"}
PREP_CL = {"bi_prep", "bi_part", "li_prep", "ka_prep", "wa_prep", "ta_prep", "la_prep"}


def cls(pos):
    if pos == "verb":
        return "verb"
    if pos in NOMINAL:
        return "nominal"
    if pos in PRONOM:
        return "pronominal"
    return "particle"


def enc_match(g, p):
    p = (p or "0").split("_")[0]
    if p in ("na", ""):
        p = "0"
    if g == "0" or p == "0":
        return g == p
    if len(g) == 2 and g[1] == "d":          # QAC 3d / 2d: no gender
        return p[0] == g[0] and p[-1] == "d"
    return p == g


def compare(g, p):
    """dict feature -> bool, for the features gold defines on this word."""
    r = {}
    if p is None:
        p = {}
    pc = cls(p.get("pos"))
    if g["cls"]:
        r["pos_class"] = pc == g["cls"]
        if g["cls"] == "verb":
            r["verb_is_verb"] = pc == "verb"
        if g["cls"] == "nominal":
            r["nominal_is_nominal"] = pc == "nominal"
    if g.get("prop") is not None:
        r["proper_noun"] = (p.get("pos") == "noun_prop") == g["prop"]
    for f in ("asp", "vox", "per", "gen", "num"):
        if f in g:
            r[f] = p.get(f) == g[f]
    for f in ("cas", "mod"):
        if f in g:
            r[f] = p.get(f) == g[f]
            decided = p.get(f) not in ("u", "na", None)
            r[f + "_left_open"] = not decided          # share where the reading leaves it open
            if decided:
                r[f + "_when_decided"] = p.get(f) == g[f]
    prc2 = p.get("prc2") or "0"
    r["conj_clitic"] = (prc2 not in ("0", "na")) == g["conj"]
    r["det"] = (p.get("prc0") == "Al_det") == g["det"]
    r["prep_clitic"] = (p.get("prc1") in PREP_CL) == g["prep_clitic"]
    r["ques_hamza"] = (p.get("prc3") == ">a_ques") == g["q"]
    r["pron_suffix"] = enc_match(g["enc"], p.get("enc0"))
    r["ALL"] = all(v for k, v in r.items() if not k.endswith(("_left_open", "_when_decided")))
    return r


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    jout = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    if jout in args:
        args.remove(jout)
    gold = {}
    for line in open(args[0], encoding="utf-8"):
        r = json.loads(line)
        gold[r["loc"]] = r
    systems = {}
    for a in args[1:]:
        name, path = a.split("=", 1)
        systems[name] = {json.loads(l)["loc"]: json.loads(l)["pred"] for l in open(path, encoding="utf-8")}
    common = set(gold).intersection(*[set(s) for s in systems.values()])
    if "--heldout" in sys.argv:
        common = {l for l in common if int(l.split(":")[0]) % 10 == 3}
    res = {n: defaultdict(Counter) for n in systems}
    confusions = {n: Counter() for n in systems}
    for loc in common:
        g = gold[loc]
        for n, s in systems.items():
            for t, gw, pw in zip(g["words"], g["gold"], s[loc]):
                if not gw:
                    continue
                for f, ok in compare(gw, pw).items():
                    res[n][f]["n"] += 1
                    res[n][f]["ok"] += ok
                pc = cls((pw or {}).get("pos"))
                if gw["cls"] and pc != gw["cls"]:
                    confusions[n][(gw["cls"], pc)] += 1
                if "cas" in gw and (pw or {}).get("cas") != gw["cas"]:
                    confusions[n][("cas " + gw["cas"], "→" + str((pw or {}).get("cas")))] += 1
    feats = list(next(iter(res.values())).keys())
    order = ["ALL", "pos_class", "verb_is_verb", "nominal_is_nominal", "proper_noun", "asp", "vox",
             "per", "gen", "num", "mod", "mod_left_open", "mod_when_decided",
             "cas", "cas_left_open", "cas_when_decided", "conj_clitic", "prep_clitic", "det", "ques_hamza", "pron_suffix"]
    feats = [f for f in order if f in feats]
    print("verses", len(common))
    print("feature".ljust(20), "n".rjust(7), *[n.rjust(10) for n in systems])
    table = {}
    for f in feats:
        n0 = next(iter(res.values()))[f]["n"]
        row = [res[n][f]["ok"] / res[n][f]["n"] for n in systems]
        table[f] = {"n": n0, **{n: v for n, v in zip(systems, row)}}
        print(f.ljust(20), str(n0).rjust(7), *[f"{100*v:9.1f}%" for v in row])
    for n in systems:
        print("\n", n, "top confusions:", confusions[n].most_common(12))
    if jout:
        json.dump({"table": table, "confusions": {n: [[list(k), v] for k, v in c.most_common(40)]
                                                   for n, c in confusions.items()}},
                  open(jout, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
