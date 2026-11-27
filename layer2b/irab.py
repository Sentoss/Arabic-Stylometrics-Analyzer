"""Layer 2b: run the trained i'rab model (training/train_irab.py) on text.

    tagger = IrabTagger("models/irab_model")
    tagger.tag(["قال", "محمد", "هو", "ابن", "مالك"])
    -> [{"word", "role", "func", "rel", "target", "target_word", "hidden", "p_func",
         and, when the word has them, "conj", "part", "subj", "enc", "cm", "ptype", "clause"}, …]
    (conj: the و/ف prefix; part: another prefix; subj / enc: attached pronouns;
     cm: the case or mood the word carries; ptype: the kind of particle or relative;
     clause: the function of the clause that starts at this word. Models trained
     before these heads existed simply leave them out.)

target is the index of the linked word, or one of the محذوف slots ("محذوف خبر" …),
and is only given when rel is not "none". hidden (verbs only) is True when the
subject is a ضمير مستتر.

Corpus run (one .jsonl per chunk, one segment per line):
    python -m layer2b.irab MODEL_DIR CORPUS_DIR SAMPLE.tsv OUT_DIR [--analyzed analyzed_l2]
The words fed to the model are layer 1's restored spellings, with vowels removed,
cut into segments of at most 40 words at punctuation or before a connective.
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

N_SLOTS = 5
CLITICS = ("conj", "part", "subj", "enc", "cm", "ptype", "mn", "clause")
VOWELS = re.compile(r"[ً-ْٰـ]")
STOP = set(".!?؟:;؛")
CONN = re.compile(r"^(ف|ثم$|و)")


class _Model(nn.Module):
    """Same layout as IrabModel in training/train_irab.py, built from a config
    so that no download is needed at run time."""

    def __init__(self, enc_config, n_roles, n_funcs, n_rels, arc_dim=256, clitics=None):
        super().__init__()
        self.enc = AutoModel.from_config(enc_config)
        h = enc_config.hidden_size
        self.drop = nn.Dropout(0.0)
        self.role = nn.Linear(h, n_roles)
        self.func = nn.Linear(h, n_funcs)
        self.rel = nn.Linear(h, n_rels)
        self.hidden = nn.Linear(h, 2)
        self.dep = nn.Sequential(nn.Linear(h, arc_dim), nn.GELU())
        self.head = nn.Sequential(nn.Linear(h, arc_dim), nn.GELU())
        self.slots = nn.Parameter(torch.zeros(N_SLOTS, arc_dim))
        self.bil = nn.Parameter(torch.eye(arc_dim))
        self.clitic = nn.ModuleDict({k: nn.Linear(h, n) for k, n in (clitics or {}).items() if n})

    def forward(self, ids, att, first, wmask):
        hid = self.enc(input_ids=ids, attention_mask=att).last_hidden_state
        x = torch.gather(hid, 1, first.unsqueeze(-1).expand(-1, -1, hid.size(-1)))
        d, h = self.dep(x), self.head(x)
        cand = torch.cat([h, self.slots.unsqueeze(0).expand(h.size(0), -1, -1)], 1)
        arc = d @ self.bil @ cand.transpose(1, 2)
        cmask = torch.cat([wmask, torch.ones(wmask.size(0), N_SLOTS, dtype=torch.bool, device=wmask.device)], 1)
        W = wmask.size(1)
        self_mask = torch.eye(W, W + N_SLOTS, dtype=torch.bool, device=arc.device).unsqueeze(0)
        arc = arc.masked_fill(~cmask.unsqueeze(1) | self_mask, -1e4)
        out = {"role": self.role(x), "func": self.func(x), "rel": self.rel(x),
               "hidden": self.hidden(x), "target": arc}
        for k, layer in self.clitic.items():
            out[k] = layer(x)
        return out


class IrabTagger:
    def __init__(self, model_dir, device=None, max_len=256, batch=16):
        d = Path(model_dir)
        if not (d / "model.pt").exists() and (d / "irab_model" / "model.pt").exists():
            d = d / "irab_model"            # the folder inside irab_model.zip
        cfg = json.load(open(d / "config.json", encoding="utf-8"))
        self.labels = cfg["labels"]
        self.tok = AutoTokenizer.from_pretrained(d / "tokenizer")
        enc_cfg_path = d / "encoder_config"
        if enc_cfg_path.exists():
            enc_cfg = AutoConfig.from_pretrained(enc_cfg_path)
        else:                                # first use: fetch once, keep beside the model
            enc_cfg = AutoConfig.from_pretrained(cfg["base"])
            try:
                enc_cfg.save_pretrained(enc_cfg_path)
            except OSError:
                pass
        L = self.labels
        self.clitics = [k for k in CLITICS if k in L]
        self.model = _Model(enc_cfg, len(L["roles"]), len(L["funcs"]), len(L["rels"]),
                            clitics={k: len(L[k]) for k in self.clitics})
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(d / "model.pt", map_location="cpu")
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if [k for k in missing if not k.startswith("enc.pooler")] or unexpected:
            raise RuntimeError(f"model.pt does not fit: missing {missing[:5]}, unexpected {unexpected[:5]}")
        self.model.to(self.device).eval()
        self.max_len, self.batch = max_len, batch
        self.slots = L.get("slots", ["محذوف"] * N_SLOTS)

    @torch.no_grad()
    def tag_many(self, segments):
        """segments: list of word lists -> list of lists of per-word dicts."""
        out = [None] * len(segments)
        order = sorted(range(len(segments)), key=lambda i: len(segments[i]))
        for b in range(0, len(order), self.batch):
            idx = order[b:b + self.batch]
            enc = [self._encode(segments[i]) for i in idx]
            L = max(len(e[0]) for e in enc); W = max(max(len(e[1]) for e in enc), 1)
            ids = torch.full((len(enc), L), self.tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(enc), L), dtype=torch.long)
            first = torch.zeros((len(enc), W), dtype=torch.long)
            wmask = torch.zeros((len(enc), W), dtype=torch.bool)
            for k, (x, f) in enumerate(enc):
                ids[k, :len(x)] = torch.tensor(x); att[k, :len(x)] = 1
                first[k, :len(f)] = torch.tensor(f); wmask[k, :len(f)] = True
            o = self.model(ids.to(self.device), att.to(self.device), first.to(self.device), wmask.to(self.device))
            probs = {k: v.float().softmax(-1).cpu() for k, v in o.items()}
            for k, i in enumerate(idx):
                out[i] = self._decode(segments[i], enc[k][1], probs, k, W)
        return out

    def tag(self, words):
        return self.tag_many([words])[0]

    def _encode(self, words):
        clean = [VOWELS.sub("", w) or w for w in words]
        e = self.tok(clean, is_split_into_words=True, truncation=True, max_length=self.max_len)
        wid = e.word_ids()
        first = [i for i, w in enumerate(wid) if w is not None and (i == 0 or wid[i - 1] != w)]
        return e["input_ids"], first

    def _decode(self, words, first, P, k, W):
        L = self.labels
        n = len(first)
        res = []
        for i, w in enumerate(words):
            if i >= n:                       # cut off by max_len
                res.append({"word": w})
                continue
            fp, fi = P["func"][k, i].max(-1)
            rel = L["rels"][int(P["rel"][k, i].argmax())]
            r = {"word": w, "role": L["roles"][int(P["role"][k, i].argmax())],
                 "func": L["funcs"][int(fi)], "p_func": round(float(fp), 3), "rel": rel,
                 "target": None, "target_word": None, "hidden": None}
            if rel != "none":
                arc = P["target"][k, i].clone()
                arc[n:W] = 0                 # padding columns
                t = int(arc.argmax())
                if t < n:
                    r["target"], r["target_word"] = t, words[t]
                else:
                    r["target"] = r["target_word"] = self.slots[t - W]
            if r["func"] in ("فعل ماض", "فعل مضارع", "فعل أمر"):
                r["hidden"] = bool(P["hidden"][k, i].argmax())
            for c in self.clitics:           # attached pieces: "none" when the word has none
                lab = L[c][int(P[c][k, i].argmax())]
                if lab != "none":
                    r[c] = lab
            res.append(r)
        return res


def segments_from_tokens(tokens, maxw=40):
    """Cut a chunk's layer-1 tokens into model-sized segments."""
    segs, cur = [], []
    for t in tokens:
        if t["status"] == "not_word":
            if (t["raw"] in STOP or t["raw"] == "*" or (t["raw"] == "،" and len(cur) >= 20)) and cur:
                segs.append(cur); cur = []
            continue
        w = VOWELS.sub("", t["restored"] or t["raw"])
        if not re.fullmatch(r"[ء-ي]+", w):
            continue
        cur.append(w)
        if len(cur) >= maxw:
            cut = max((i for i in range(maxw // 2, maxw) if CONN.match(cur[i]) and len(cur[i]) > 2), default=maxw)
            segs.append(cur[:cut]); cur = cur[cut:]
    if cur:
        segs.append(cur)
    return [s for s in segs if len(s) >= 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("corpus"); ap.add_argument("sample"); ap.add_argument("out")
    ap.add_argument("--analyzed", default="analyzed_l2")
    ap.add_argument("--worker", default="0/1")
    ap.add_argument("--max-chunks", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    tagger = IrabTagger(a.model)
    root, out = Path(a.corpus), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(a.sample, encoding="utf-8"), delimiter="\t"))
    k, n = map(int, a.worker.split("/"))
    todo = [r for r in rows[k::n] if not (out / f"{r['chunk_id']}.jsonl").exists()
            and (root / a.analyzed / f"{r['chunk_id']}.tsv").exists()]
    if a.max_chunks:
        todo = todo[:a.max_chunks]
    print(f"{len(todo)} chunks to tag on {tagger.device}", flush=True)
    t0, words = time.time(), 0
    for i, r in enumerate(todo, 1):
        toks = list(csv.DictReader(open(root / a.analyzed / f"{r['chunk_id']}.tsv", encoding="utf-8"), delimiter="\t"))
        segs = segments_from_tokens(toks)
        tagged = tagger.tag_many(segs)
        tmp = out / f"{r['chunk_id']}.jsonl.part"
        with open(tmp, "w", encoding="utf-8") as f:
            for s in tagged:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        tmp.rename(out / f"{r['chunk_id']}.jsonl")
        words += sum(len(s) for s in segs)
        print(f"{i}/{len(todo)} chunks, {words / (time.time() - t0):.0f} words/s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
