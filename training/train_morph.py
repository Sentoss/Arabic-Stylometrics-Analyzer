"""Layer 2a: fine-tune CAMeLBERT-CA as a Classical morphological tagger.

For every (unvowelled) word the model predicts 14 features: part of speech,
the four proclitic slots, person, aspect, voice, mood, gender, number, state,
case and the pronoun suffix. Ties in the training data are sets of allowed
values; the loss counts any of them as right (partial-label loss). At analysis
time, the program picks the dictionary reading (CAMeL + Classical supplement)
that agrees best with these predictions; that gives the lemma too.

Data: the folder made by layer2/make_dataset.py (train/val/test .jsonl.gz,
labels.json).

usage:
  python training/train_morph.py --data DATA_DIR --out OUT_DIR [--epochs 3]
  python training/train_morph.py --data DATA_DIR --out OUT_DIR --smoke   # tiny CPU test
"""
import argparse
import gzip
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

BASE = "CAMeL-Lab/bert-base-arabic-camelbert-ca"


def read(path, limit=0):
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


class Segments(Dataset):
    def __init__(self, rows, tok, n_feats, max_len):
        self.items = []
        for r in rows:
            enc = tok(r["w"], is_split_into_words=True, truncation=True, max_length=max_len)
            wid = enc.word_ids()
            first = [i for i, w in enumerate(wid) if w is not None and (i == 0 or wid[i - 1] != w)]
            targets = [r["t"][wid[i]] for i in first]
            self.items.append((enc["input_ids"], first, targets))
        self.n_feats = n_feats

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch, pad_id, sizes):
    L = max(len(b[0]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    pos = []          # (batch index, token index) of every word's first sub-token
    allowed = [[] for _ in sizes]   # per feature: list of multi-hot vectors (or None)
    for b, (x, first, targets) in enumerate(batch):
        ids[b, :len(x)] = torch.tensor(x)
        att[b, :len(x)] = 1
        for i, t in zip(first, targets):
            pos.append((b, i))
            for k, n in enumerate(sizes):
                if t is None or not t[k]:
                    allowed[k].append(None)
                else:
                    m = torch.zeros(n, dtype=torch.bool)
                    m[t[k]] = True
                    allowed[k].append(m)
    masks, valid = [], []
    for k, n in enumerate(sizes):
        v = torch.tensor([a is not None for a in allowed[k]])
        m = torch.stack([a if a is not None else torch.zeros(n, dtype=torch.bool) for a in allowed[k]]) \
            if allowed[k] else torch.zeros((0, n), dtype=torch.bool)
        masks.append(m)
        valid.append(v)
    return ids, att, torch.tensor(pos, dtype=torch.long).reshape(-1, 2), masks, valid


class Tagger(nn.Module):
    def __init__(self, base, sizes, dropout=0.1):
        super().__init__()
        self.enc = AutoModel.from_pretrained(base)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.Linear(h, n) for n in sizes])

    def forward(self, ids, att, pos):
        hid = self.enc(input_ids=ids, attention_mask=att).last_hidden_state
        x = self.drop(hid[pos[:, 0], pos[:, 1]])
        return [head(x) for head in self.heads]


def partial_label_loss(logits, mask, valid):
    """-log of the total probability of the allowed values (ties count as right)."""
    if valid.sum() == 0:
        return logits.sum() * 0
    lp = logits[valid].log_softmax(-1)
    m = mask[valid].to(lp.device)
    return -(lp.masked_fill(~m, -1e9).logsumexp(-1)).mean()


@torch.no_grad()
def evaluate(model, loader, dev, names):
    model.eval()
    right = [0] * len(names)
    total = [0] * len(names)
    full_right = full_total = 0
    for ids, att, pos, masks, valid in loader:
        logits = model(ids.to(dev), att.to(dev), pos.to(dev))
        word_ok = None
        for k, lg in enumerate(logits):
            pred = lg.argmax(-1).cpu()
            ok = masks[k][torch.arange(len(pred)), pred]
            v = valid[k]
            right[k] += int(ok[v].sum())
            total[k] += int(v.sum())
            ok_or_skip = ok | ~v
            word_ok = ok_or_skip if word_ok is None else (word_ok & ok_or_skip)
        labelled = valid[0]
        full_right += int(word_ok[labelled].sum())
        full_total += int(labelled.sum())
    model.train()
    acc = {n: right[k] / max(total[k], 1) for k, n in enumerate(names)}
    acc["ALL_FEATURES"] = full_right / max(full_total, 1)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--limit", type=int, default=0, help="use only N training segments")
    ap.add_argument("--smoke", action="store_true", help="tiny run to check that everything works")
    a = ap.parse_args()
    if a.smoke:
        a.limit, a.epochs, a.batch = 40, 1, 8
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {dev}", flush=True)
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    labels = json.load(open(data / "labels.json", encoding="utf-8"))
    names = labels["features"]
    sizes = [len(labels["values"][n]) for n in names]
    tok = AutoTokenizer.from_pretrained(a.base)
    lim_eval = 20 if a.smoke else 0
    train = Segments(read(data / "train.jsonl.gz", a.limit), tok, len(names), a.max_len)
    val = Segments(read(data / "val.jsonl.gz", lim_eval), tok, len(names), a.max_len)
    test = Segments(read(data / "test.jsonl.gz", lim_eval), tok, len(names), a.max_len)
    print(f"segments: train {len(train)}, val {len(val)}, test {len(test)}", flush=True)
    col = lambda b: collate(b, tok.pad_token_id, sizes)
    tl = DataLoader(train, batch_size=a.batch, shuffle=True, collate_fn=col)
    vl = DataLoader(val, batch_size=a.batch * 2, collate_fn=col)
    sl = DataLoader(test, batch_size=a.batch * 2, collate_fn=col)
    model = Tagger(a.base, sizes).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * len(tl)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))
    best, t0 = -1, time.time()
    for ep in range(a.epochs):
        for step, (ids, att, pos, masks, valid) in enumerate(tl, 1):
            with torch.autocast(device_type=dev, dtype=torch.float16, enabled=(dev == "cuda")):
                logits = model(ids.to(dev), att.to(dev), pos.to(dev))
            loss = sum(partial_label_loss(lg.float(), masks[k], valid[k].to(lg.device))
                       for k, lg in enumerate(logits)) / len(logits)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
            if step % 200 == 0 or a.smoke:
                el = time.time() - t0
                print(f"epoch {ep + 1} step {step}/{len(tl)} loss {loss.item():.4f} "
                      f"({el / 60:.1f} min)", flush=True)
        acc = evaluate(model, vl, dev, names)
        print(f"epoch {ep + 1} validation: " + "  ".join(f"{k} {v:.3f}" for k, v in acc.items()), flush=True)
        if acc["ALL_FEATURES"] > best:
            best = acc["ALL_FEATURES"]
            torch.save(model.state_dict(), out / "model.pt")
            print(f"  saved (best so far: {best:.3f})", flush=True)
    model.load_state_dict(torch.load(out / "model.pt", map_location=dev))
    acc = evaluate(model, sl, dev, names)
    print("test: " + "  ".join(f"{k} {v:.3f}" for k, v in acc.items()), flush=True)
    tok.save_pretrained(out / "tokenizer")
    json.dump({"base": a.base, "labels": labels, "test": acc, "args": vars(a)},
              open(out / "config.json", "w"), ensure_ascii=False, indent=1)
    print(f"done in {(time.time() - t0) / 60:.1f} min; files in {out}", flush=True)


if __name__ == "__main__":
    main()
