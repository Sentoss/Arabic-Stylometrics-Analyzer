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
STEMS = Path(__file__).resolve().parents[1] / "resources" / "lane_stems.tsv"
CACHE = Path.home() / ".cache" / "arabic_stylometrics"


def merged_path(stems=STEMS):
    h = hashlib.sha1(Path(stems).read_bytes()).hexdigest()[:10]
    out = CACHE / f"calima-msa-r13+lane-{h}.db"
    if out.exists():
        return out
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = []
    for line in open(stems, encoding="utf-8"):
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
    return MorphologyDB(str(merged_path(stems)), flags)
