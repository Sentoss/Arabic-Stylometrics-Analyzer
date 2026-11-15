"""Command line: restore spelling in a text file and write the three tiers.

usage: python restore_cli.py INPUT.txt OUTDIR [--force CONV=on|off ...]

Writes:
  OUTDIR/<name>.restored.txt   restored text (standard spelling, distinctions kept)
  OUTDIR/<name>.tokens.tsv     raw / restored / analysis per token, aligned
  OUTDIR/<name>.profile.json   detected printing conventions + evidence
"""
import argparse
import json
from pathlib import Path

from restorer import Restorer

COLS = ["sent", "start", "end", "raw", "restored", "changed", "status", "confidence",
        "conventions", "alternatives", "lemma", "pos", "root", "pattern", "diac", "quran"]


def rebuild(text, results):
    out, pos = [], 0
    for r in results:
        out.append(text[pos:r.start])
        out.append(r.restored)
        pos = r.end
    out.append(text[pos:])
    return "".join(out)


def write_outputs(name, outdir, text, profile, results):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{name}.restored.txt").write_text(rebuild(text, results), encoding="utf-8")
    (outdir / f"{name}.profile.json").write_text(
        json.dumps(profile.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
    with open(outdir / f"{name}.tokens.tsv", "w", encoding="utf-8") as f:
        f.write("\t".join(COLS) + "\n")
        for r in results:
            d = r.to_dict()
            d["conventions"] = ",".join(d["conventions"])
            d["alternatives"] = ",".join(d["alternatives"])
            f.write("\t".join(str(d[c]) for c in COLS) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("outdir")
    ap.add_argument("--force", nargs="*", default=[], help="CONV=on|off")
    a = ap.parse_args()
    overrides = {k: v == "on" for k, v in (x.split("=") for x in a.force)}
    text, profile, results = Restorer().run(Path(a.input).read_text(encoding="utf-8"), overrides)
    write_outputs(Path(a.input).stem, Path(a.outdir), text, profile, results)
    ch = [r for r in results if r.changed]
    print(f"{len(results)} tokens, {len(ch)} changed; conventions on: "
          f"{[k for k, v in profile.conventions.items() if v]}")


if __name__ == "__main__":
    main()
