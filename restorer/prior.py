"""Classical word-form frequencies, used to break ties the MSA context
model cannot resolve (ألف vs إلف, الإمام vs الأمام, سنة vs سنه).

Built from any corpus in *standard* spelling. Diacritics are stripped but
letters are not normalized, so hamza / ى-ي / ة-ه distinctions are counted.
"""
import math
import re
from collections import Counter
from pathlib import Path

_MARKS = re.compile(r"[ً-ْٰـ]")
_WORD = re.compile(r"[ء-غف-ي]+")


class FormPrior:
    def __init__(self, counts=None):
        self.counts = Counter(counts or {})
        self.total = sum(self.counts.values()) or 1

    @classmethod
    def from_corpus(cls, paths):
        c = Counter()
        for p in paths:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    c.update(_WORD.findall(_MARKS.sub("", line)))
        return cls(c)

    @classmethod
    def load(cls, path):
        c = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                w, n = line.rstrip("\n").split("\t")
                c[w] = int(n)
        return cls(c)

    def save(self, path, min_count=2):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for w, n in self.counts.most_common():
                if n < min_count:
                    break
                f.write(f"{w}\t{n}\n")

    def __call__(self, form):
        return self.counts.get(form, 0)

    def logp(self, form):
        return math.log((self.counts.get(form, 0) + 0.5) / self.total)
