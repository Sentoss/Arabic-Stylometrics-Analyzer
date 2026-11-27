"""CAMeL's MSA morphology database with Lane's stems added.

The merged database is built once from calima-msa-r13 (downloaded by CAMeL
Tools) and `layer2/resources/lane_stems.tsv`, and cached under
~/.cache/arabic_stylometrics/. Neither CAMeL's database nor the merged copy is
stored in the repository.

    from layer2.lane.lexicon_db import lane_db
    from camel_tools.morphology.analyzer import Analyzer
    an = Analyzer(lane_db(), backoff="ADD_PROP")
"""
import hashlib
from pathlib import Path

from camel_tools.morphology.database import MorphologyDB

BASE = Path.home() / ".camel_tools/data/morphology_db/calima-msa-r13/morphology.db"
RES = Path(__file__).resolve().parents[1] / "resources"
STEMS = (RES / "lane_stems.tsv", RES / "lane_verbs.tsv", RES / "arramooz_stems.tsv", RES / "lane_names.tsv")
RARE_TRIAL = RES / "lane_rare_trial.tsv"                         # trial: rare competing words, low prior
CACHE = Path.home() / ".cache" / "arabic_stylometrics"


def merged_path(stems=STEMS):
    stems = [Path(x) for x in ([stems] if isinstance(stems, (str, Path)) else stems) if Path(x).exists()]
    h = hashlib.sha1(b"".join(x.read_bytes() for x in stems)).hexdigest()[:10]
    out = CACHE / f"calima-msa-r13+lane-{h}.db"
    if out.exists():
        return out
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in stems:
        for line in open(f, encoding="utf-8"):
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                rows.append("\t".join(p[:3]) + "\n")
    tmp = out.with_suffix(".tmp")
    with open(BASE, encoding="utf-8") as src, open(tmp, "w", encoding="utf-8") as dst:
        for line in src:
            if line.startswith("###TABLE AB###"):
                dst.writelines(rows)          # end of the STEMS section
            dst.write(line)
    tmp.rename(out)
    return out


def lane_db(flags="a", stems=STEMS):
    """stems: one TSV or a list of TSVs of stem lines (stem, category, features …)."""
    return MorphologyDB(str(merged_path(stems)), flags)
