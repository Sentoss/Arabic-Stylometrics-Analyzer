"""Parse Lane's Arabic-English Lexicon (Perseus TEI XML, as restructured by
W. H. Leclaire, github.com/quranic-research/lexicon_xml) into a table of
vowelled Arabic words with what Lane says about them.

Each root (superEntry) holds entries: verb entries carry an iType (the verb
form number 1-10..., Form I with its aorist vowel, "aor. yaktubu"), the other
entries are nouns/adjectives/particles (orth = the vowelled headword). From
the entry text we also take the plurals ("pl. X and Y"), the infinitive nouns
("inf. n. X and Y") of verbs, and the feminine ("fem. X").

Arabic in the XML is in the edition's own transliteration (Encoding.dtd,
"internal" mapping): converted to Arabic script with internal_map.json.

usage: python layer2/lane/parse_lane.py XML_DIR OUT.jsonl
Each line: {"root", "kind": verb|noun|plural|masdar|fem|aorist, "word",
"form" (verb form number, verbs only), "aor" (Form I imperfect, if given),
"of" (the headword a plural/masdar/fem belongs to), "file", "entry"}.

Text provided by Perseus Digital Library, with funding from The U.S.
Department of Education and The Max Planck Society.
"""
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

T = "{http://www.tei-c.org/ns/1.0}"
MAP = json.load(open(Path(__file__).with_name("internal_map.json"), encoding="utf-8"))
AR_TOKEN = re.compile(r"[A-Za-z{}~`]+")


def ar(s):
    return "".join(MAP.get(c, c if c.isspace() else "") for c in s or "")


def load(path):
    t = open(path, encoding="utf-8").read()
    t = re.sub(r"<!DOCTYPE[^>]*>", "", t)
    t = re.sub(r"&[a-zA-Z]+;", "", t)
    return ET.fromstring(t)


def flat(el):
    """Entry text with Arabic marked: <foreign lang=ar> runs become «…» in Arabic script."""
    out = []

    def walk(e):
        if e.tag == T + "foreign" and e.get("lang") == "ar":
            out.append(" «" + ar("".join(e.itertext())) + "» ")
        else:
            if e.text:
                out.append(e.text)
            for c in e:
                walk(c)
                if c.tail:
                    out.append(c.tail)
    walk(el)
    return " ".join("".join(out).split())


ARWORDS = r"((?:«[^»]+»\s*(?:↓\s*)?(?:,\s*|and\s+|or\s+|also\s+)*)+)"
PL = re.compile(r"\bpl\.\s*(?:\[[^\]]*\]\s*)?(?:of pauc\.\s*)?" + ARWORDS)
INF = re.compile(r"\binf\.\s*ns?\.\s*" + ARWORDS)
FEM = re.compile(r"\bfem\.\s*" + ARWORDS)
AOR = re.compile(r"\baor\.\s*«([^»]+)»")


def words(run):
    return [w.strip() for w in re.findall(r"«([^»]+)»", run) if w.strip() and " " not in w.strip()]


def main(xml_dir, out_path):
    n = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for p in sorted(Path(xml_dir).glob("[0-9][0-9]_*.xml")):
            root_el = load(p)
            for se in root_el.iter(T + "superEntry"):
                tt = se.find(f".//{T}title[@type='root']/{T}foreign")
                root = ar(tt.text) if tt is not None else ""
                for e in se.findall(T + "entry"):
                    orths = [ar(o.text) for o in e.findall(f"{T}form/{T}orth")]
                    head = orths[0] if orths else ""
                    it = e.find(f".//{T}iType")
                    text = flat(e)
                    base = {"root": root, "file": p.name, "entry": e.get("n")}
                    rows = []
                    if it is not None and (it.text or "").strip().isdigit():
                        form = int(it.text.strip())
                        aor = None
                        if form == 1:          # aorist given as <choice type="infl"><orth orig="Bu">: the stem vowel
                            ch = e.find(f".//{T}choice[@type='infl']/{T}orth")
                            if ch is not None and ch.get("orig", "")[-1:] in "aui":
                                aor = ch.get("orig")[-1]
                            else:
                                m = AOR.search(text[:400])
                                aor = m.group(1) if m else None
                        rows.append({"kind": "verb", "word": head, "form": form, "aor": aor})
                        for run in INF.findall(text[:600]):
                            rows += [{"kind": "masdar", "word": w, "of": head, "form": form} for w in words(run)]
                    elif head and " " not in head:
                        rows.append({"kind": "noun", "word": head})
                        for rx, kind in ((PL, "plural"), (FEM, "fem")):
                            for run in rx.findall(text[:500]):
                                rows += [{"kind": kind, "word": w, "of": head} for w in words(run)]
                    for r in rows:
                        out.write(json.dumps({**base, **r}, ensure_ascii=False) + "\n")
                        n += 1
            print(p.name, n, flush=True)
    print("rows", n)


if __name__ == "__main__":
    main(*sys.argv[1:3])
