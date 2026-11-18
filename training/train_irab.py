"""Layer 2b: fine-tune CAMeLBERT-CA as a Classical i'rab model.

For every (unvowelled) word the model predicts
  role    MASAQ's i'rab role
  func    the book's naḥw function (فاعل، حال، تمييز، مفعول مطلق…)
  rel     the kind of link: تعلق / نعت / عطف / بدل / توكيد / حال / none
  target  the word it links to (المتعلَّق به، المتبوع، صاحب الحال), or one of
          five "محذوف" slots (محذوف خبر / صفة / حال / صلة / غير ذلك)
  hidden  for verbs: whether the subject is a ضمير مستتر
Labels missing for a word (-1) are skipped by the loss.

Data: the folder made by layer2b/make_dataset.py (internal use only).

usage:
  python training/train_irab.py --data DATA_DIR --out OUT_DIR [--epochs 10]
  python training/train_irab.py --data DATA_DIR --out OUT_DIR --smoke
"""
import argparse
import gzip
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

BASE = "CAMeL-Lab/bert-base-arabic-camelbert-ca"
N_SLOTS = 5
FIELDS = ("role", "func", "rel", "target", "hidden")


def read(path, limit=0):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def encode(rows, tok, max_len):
    items = []
    for r in rows:
        enc = tok(r["w"], is_split_into_words=True, truncation=True, max_length=max_len)
        wid = enc.word_ids()
        first = [i for i, w in enumerate(wid) if w is not None and (i == 0 or wid[i - 1] != w)]
        n = len(first)                      # words that survived truncation
        lab = {k: list(r[k][:n]) for k in FIELDS}
        # a target pointing at a truncated word, or a slot, must be re-indexed to the kept length
        nw = len(r["w"])
        lab["target"] = [(t - nw + n if t >= nw else (t if t < n else -1)) if t >= 0 else -1
                         for t in lab["target"]]
        items.append((enc["input_ids"], first, lab))
    return items


def collate(batch, pad_id):
    L = max(len(b[0]) for b in batch)
    W = max(len(b[1]) for b in batch)
    B = len(batch)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    att = torch.zeros((B, L), dtype=torch.long)
    first = torch.zeros((B, W), dtype=torch.long)
    wmask = torch.zeros((B, W), dtype=torch.bool)
    lab = {k: torch.full((B, W), -1, dtype=torch.long) for k in FIELDS}
    for b, (x, f, l) in enumerate(batch):
        ids[b, :len(x)] = torch.tensor(x)
        att[b, :len(x)] = 1
        first[b, :len(f)] = torch.tensor(f)
        wmask[b, :len(f)] = True
        n = len(f)
        for k in FIELDS:
            v = torch.tensor(l[k], dtype=torch.long)
            if k == "target":
                # slots are numbered after this block's words; move them after the padded width
                v = torch.where(v >= n, v - n + W, v)
            lab[k][b, :n] = v
    return ids, att, first, wmask, lab


class IrabModel(nn.Module):
    def __init__(self, base, n_roles, n_funcs, n_rels, dropout=0.1, arc_dim=256):
        super().__init__()
        self.enc = AutoModel.from_pretrained(base)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.role = nn.Linear(h, n_roles)
        self.func = nn.Linear(h, n_funcs)
        self.rel = nn.Linear(h, n_rels)
        self.hidden = nn.Linear(h, 2)
        self.dep = nn.Sequential(nn.Linear(h, arc_dim), nn.GELU())
        self.head = nn.Sequential(nn.Linear(h, arc_dim), nn.GELU())
        self.slots = nn.Parameter(torch.randn(N_SLOTS, arc_dim) * 0.02)
        self.bil = nn.Parameter(torch.eye(arc_dim))

    def forward(self, ids, att, first, wmask):
        hid = self.enc(input_ids=ids, attention_mask=att).last_hidden_state
        x = self.drop(torch.gather(hid, 1, first.unsqueeze(-1).expand(-1, -1, hid.size(-1))))
        d, h = self.dep(x), self.head(x)                                   # B x W x A
        cand = torch.cat([h, self.slots.unsqueeze(0).expand(h.size(0), -1, -1)], 1)   # B x (W+S) x A
        arc = d @ self.bil @ cand.transpose(1, 2)                          # B x W x (W+S)
        cmask = torch.cat([wmask, torch.ones(wmask.size(0), N_SLOTS, dtype=torch.bool, device=wmask.device)], 1)
        W = wmask.size(1)
        self_mask = torch.eye(W, W + N_SLOTS, dtype=torch.bool, device=arc.device).unsqueeze(0)
        arc = arc.masked_fill(~cmask.unsqueeze(1) | self_mask, -1e4)
        return {"role": self.role(x), "func": self.func(x), "rel": self.rel(x),
                "hidden": self.hidden(x), "target": arc}


def losses(out, lab):
    ce = nn.CrossEntropyLoss(ignore_index=-1)
    total = 0
    for k in FIELDS:
        lg = out[k].float()
        y = lab[k].to(lg.device)
        if (y >= 0).any():
            total = total + ce(lg.reshape(-1, lg.size(-1)), y.reshape(-1))
    return total


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    right = {k: 0 for k in FIELDS}
    total = {k: 0 for k in FIELDS}
    for ids, att, first, wmask, lab in loader:
        out = model(ids.to(dev), att.to(dev), first.to(dev), wmask.to(dev))
        for k in FIELDS:
            y = lab[k]
            p = out[k].argmax(-1).cpu()
            m = y >= 0
            right[k] += int((p[m] == y[m]).sum())
            total[k] += int(m.sum())
    model.train()
    return {k: right[k] / max(total[k], 1) for k in FIELDS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    lim = 0
    if a.smoke:
        lim, a.epochs, a.batch = 24, 1, 8
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {dev}", flush=True)
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    labels = json.load(open(data / "labels.json", encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(a.base)
    splits = {s: encode(read(data / f"{s}.jsonl.gz", lim), tok, a.max_len) for s in ("train", "val", "test")}
    if a.smoke:
        splits["val"] = splits["val"] or splits["train"][:8]
        splits["test"] = splits["test"] or splits["train"][:8]
    print({s: len(v) for s, v in splits.items()}, flush=True)
    col = lambda b: collate(b, tok.pad_token_id)
    tl = DataLoader(splits["train"], batch_size=a.batch, shuffle=True, collate_fn=col)
    vl = DataLoader(splits["val"], batch_size=a.batch * 2, collate_fn=col)
    sl = DataLoader(splits["test"], batch_size=a.batch * 2, collate_fn=col)
    model = IrabModel(a.base, len(labels["roles"]), len(labels["funcs"]), len(labels["rels"])).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * len(tl)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))
    best, t0 = -1, time.time()
    for ep in range(a.epochs):
        for step, (ids, att, first, wmask, lab) in enumerate(tl, 1):
            with torch.autocast(device_type=dev, dtype=torch.float16, enabled=(dev == "cuda")):
                o = model(ids.to(dev), att.to(dev), first.to(dev), wmask.to(dev))
            loss = losses(o, lab)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
            if step % 100 == 0 or a.smoke:
                print(f"epoch {ep + 1} step {step}/{len(tl)} loss {loss.item():.4f} "
                      f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        acc = evaluate(model, vl, dev)
        score = (acc["func"] + acc["rel"] + acc["target"]) / 3
        print(f"epoch {ep + 1} validation: " + "  ".join(f"{k} {v:.3f}" for k, v in acc.items()), flush=True)
        if score > best:
            best = score
            torch.save(model.state_dict(), out / "model.pt")
            print(f"  saved (best so far: {best:.3f})", flush=True)
    model.load_state_dict(torch.load(out / "model.pt", map_location=dev))
    acc = evaluate(model, sl, dev)
    print("test: " + "  ".join(f"{k} {v:.3f}" for k, v in acc.items()), flush=True)
    tok.save_pretrained(out / "tokenizer")
    json.dump({"base": a.base, "labels": labels, "test": acc, "args": vars(a)},
              open(out / "config.json", "w"), ensure_ascii=False, indent=1)
    print(f"done in {(time.time() - t0) / 60:.1f} min; files in {out}", flush=True)


if __name__ == "__main__":
    main()
