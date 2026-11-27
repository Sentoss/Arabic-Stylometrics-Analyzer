"""Nouns from Arramooz (Taha Zerrouki, github.com/linuxscout/arramooz, GPL) for
the roots where Lane is thin: Lane's own work ends at qāf, so roots beginning
with ك ل م ن ه و ي. Writes rows in parse_lane.py's format, so the same
coverage.py and build_lane_stems.py steps (and the same add/skip rules) apply.

usage: python layer2/lane/arramooz_rows.py ARRAMOOZ/data/nouns OUT.jsonl
"""
import csv
import json
import re
import sys
from pathlib import Path

LETTERS = set("كلمنهوي")
FILES = {"jamid": "noun", "fa3il": "noun", "maf3oul": "noun", "masdar": "masdar", "moubalagha": "noun",
         "mouchabbaha": "noun", "sifates": "noun", "tafdil": "noun"}
ARABIC = re.compile(r"^[ء-ْٰ]+$")


def main(src, out_path):
    n = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for name, kind in FILES.items():
            p = Path(src) / f"{name}.csv"
            rows = list(csv.reader(open(p, encoding="utf-8"), delimiter="\t"))
            head = [h.strip().lstrip("#").strip() for h in rows[0]]
            col = {h: i for i, h in enumerate(head)}
            for r in rows[1:]:
                get = lambda h: r[col[h]].strip() if h in col and col[h] < len(r) else ""  # noqa: E731
                word, root = get("مشكول") or get("الكلمة"), get("جذر")
                if not word or not root or root[0] not in LETTERS or not ARABIC.match(word):
                    continue
                k = "plural" if get("عدد") == "جمع" else kind
                base = {"root": root, "file": f"arramooz/{name}", "entry": get("رقم") or get("")}
                out.write(json.dumps({**base, "kind": k, "word": word}, ensure_ascii=False) + "\n")
                n += 1
                for pl in re.split(r"[;،,]", get("الجمع")):
                    pl = pl.strip()
                    if pl and not pl.startswith("+") and ARABIC.match(pl):
                        out.write(json.dumps({**base, "kind": "plural", "word": pl, "of": word},
                                             ensure_ascii=False) + "\n")
                        n += 1
    print(n, "rows")


if __name__ == "__main__":
    main(*sys.argv[1:3])
