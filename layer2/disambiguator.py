"""Layer 2a at analysis time: the fine-tuned CAMeLBERT-CA tagger as a disambiguator.

Drop-in replacement for camel-tools' BERTUnfactoredDisambiguator (same
`disambiguate(sentence)` → [DisambiguatedWord]). The tagger predicts a
probability for every value of 14 features; each dictionary reading (CAMeL
calima-msa-r13 + the Classical supplement) is scored by the geometric mean of
the probabilities of its own values, then divided by the best reading's score
(so the best reading scores 1, as in camel-tools).

usage:
  from layer2.disambiguator import ClassicalDisambiguator
  dis = ClassicalDisambiguator("/path/to/morph_model")
"""
import json
import math
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))

MARK = re.compile("[ً-ْٰـ]")
FLOOR = 1e-4          # probability of a value the tagger never saw


class ClassicalDisambiguator:
    def __init__(self, model_dir, analyzer=None, device=None, max_len=256, top=1000):
        from transformers import AutoTokenizer
        from train_morph import Tagger
        model_dir = Path(model_dir)
        if (model_dir / "morph_model").is_dir():
            model_dir = model_dir / "morph_model"
        cfg = json.load(open(model_dir / "config.json", encoding="utf-8"))
        self.feats = cfg["labels"]["features"]
        self.values = cfg["labels"]["values"]
        self.index = {f: {v: i for i, v in enumerate(self.values[f])} for f in self.feats}
        self.tok = AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"))
        self.model = Tagger(cfg["base"], [len(self.values[f]) for f in self.feats])
        state = torch.load(model_dir / "model.pt", map_location="cpu")
        self.model.load_state_dict(state.get("model", state) if isinstance(state, dict) else state)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.max_len, self.top = max_len, top
        if analyzer is None:
            from camel_tools.morphology.database import MorphologyDB
            from camel_tools.morphology.analyzer import Analyzer
            analyzer = Analyzer(MorphologyDB.builtin_db("calima-msa-r13", "a"),
                                backoff="ADD_PROP", cache_size=100000)
        self._analyzer = analyzer          # same attribute name as camel-tools (Restorer swaps it)

    # -- the tagger --------------------------------------------------------
    def _pieces(self, words):
        """Split a long sentence so that each piece fits in max_len sub-tokens."""
        out, cur, n = [], [], 2
        for w in words:
            k = max(1, len(self.tok.tokenize(w)))
            if cur and n + k > self.max_len:
                out.append(cur)
                cur, n = [], 2
            cur.append(w)
            n += k
        if cur:
            out.append(cur)
        return out

    @torch.no_grad()
    def predict(self, words):
        """Per word: {feature: log-probability vector}."""
        res = []
        for piece in self._pieces([MARK.sub("", w) or w for w in words]):
            enc = self.tok(piece, is_split_into_words=True, truncation=True,
                           max_length=self.max_len, return_tensors="pt")
            wid = enc.word_ids()
            first = [i for i, w in enumerate(wid) if w is not None and (i == 0 or wid[i - 1] != w)]
            pos = torch.tensor([[0, i] for i in first])
            logits = self.model(enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device),
                                pos.to(self.device))
            lps = [lg.log_softmax(-1).cpu() for lg in logits]
            res += [{f: lps[k][j] for k, f in enumerate(self.feats)} for j in range(len(first))]
        return res

    def score(self, analysis, lp):
        s = 0.0
        for f in self.feats:
            i = self.index[f].get(analysis.get(f))
            s += float(lp[f][i]) if i is not None else math.log(FLOOR)
        return math.exp(s / len(self.feats))

    # -- camel-tools interface ----------------------------------------------
    def disambiguate(self, sentence):
        from camel_tools.disambig.common import DisambiguatedWord, ScoredAnalysis
        preds = self.predict(sentence)
        out = []
        for word, lp in zip(sentence, preds):
            analyses = self._analyzer.analyze(word)
            if not analyses:
                out.append(DisambiguatedWord(word, []))
                continue
            scored = [(self.score(a, lp), a) for a in analyses]
            best = max(s for s, _ in scored) or 1.0
            sa = [ScoredAnalysis(s / best, a, a.get("diac", ""), a.get("pos_lex_logprob", -99),
                                 a.get("lex_logprob", -99)) for s, a in scored]
            sa.sort(key=lambda x: (-x.score, -x.pos_lex_logprob, -x.lex_logprob))
            out.append(DisambiguatedWord(word, sa[:self.top]))
        return out
