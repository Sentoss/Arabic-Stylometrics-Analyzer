"""Figures for the layer 3 report.

usage: python -m stylometry.report DATA_DIR [--out reports/layer3]

Reads evaluation.json and verification.json from --out and writes:
  accuracy.png     book-level attribution accuracy per feature set and method
  dendrogram.png   bootstrap consensus tree of book profiles
  pca.png          samples of the eight best-represented authors on PC1/PC2
  impostors.png    calibration score distributions with the disputed works
  pca_loadings.json
Every figure has a matching table in README.md (colors alone never carry meaning).
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stylometry.features import load_sample
from stylometry.methods import Families, zscore, consensus_linkage, pca

# reference categorical palette, fixed order (validated: all checks pass on light)
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
SPEC = [("lemma", 300), ("function", 150), ("pos", 300), ("pattern", 200)]


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def author_name(uri):
    return re.sub(r"^\d{4}", "", uri.split(".")[0])


def label(book):
    a, b = book.split(".")[:2]
    return f"{author_name(a)}: {b}"


def fig_accuracy(ev, out):
    names = list(ev["summary"])
    methods = ["delta", "cosine", "logreg", "svm"]
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
    style(ax)
    y = np.arange(len(names))
    for k, m in enumerate(methods):
        vals = [ev["summary"][n][m][1] * 100 for n in names]
        ax.scatter(vals, y + (k - 1.5) * 0.17, s=46, color=SLOTS[k], marker=MARKERS[k],
                   edgecolor=SURFACE, linewidth=1.2, label=m, zorder=3)
    ax.set_yticks(y, names, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("held-out books attributed correctly (%)", color=INK2)
    ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.4, 1.0), fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "accuracy.png", dpi=160)
    plt.close(fig)


def book_profiles(metas, Z):
    books = sorted({m["book"] for m in metas})
    P = np.vstack([Z[[i for i, m in enumerate(metas) if m["book"] == b]].mean(0) for b in books])
    return books, P


def fig_dendrogram(metas, fams, out):
    from scipy.cluster.hierarchy import dendrogram
    known = [i for i, m in enumerate(metas) if m["status"] != "disputed"]
    mats = []
    for scale in (0.5, 1.0, 1.5):
        spec = [(f, int(n * scale)) for f, n in SPEC]
        X, _ = fams.select(spec, known)
        books, P = book_profiles(metas, zscore(X, known))
        mats.append(P)
    n_auth = len({m["author"] for m in metas})
    L, _ = consensus_linkage(mats, n_auth, None)
    status = {m["book"]: m["status"] for m in metas}
    labels = [label(b) + ("  [disputed]" if status[b] == "disputed" else "") for b in books]
    fig, ax = plt.subplots(figsize=(8, 0.19 * len(books) + 1), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    dendrogram(L, labels=labels, orientation="left", ax=ax, color_threshold=0,
               above_threshold_color=INK2, leaf_font_size=7.5)
    for t in ax.get_ymajorticklabels():
        t.set_color(INK)
        if "[disputed]" in t.get_text():
            t.set_fontweight("bold")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("1 − share of runs in the same cluster", color=INK2, fontsize=9)
    ax.tick_params(axis="x", colors=INK2, labelsize=8)
    fig.tight_layout()
    fig.savefig(out / "dendrogram.png", dpi=160)
    plt.close(fig)
    # leaf order, for the README table
    order = dendrogram(L, labels=labels, no_plot=True)["ivl"]
    return order


def fig_pca(metas, fams, out):
    secure = [i for i, m in enumerate(metas) if m["status"] == "secure"]
    nbooks = defaultdict(set)
    for i in secure:
        nbooks[metas[i]["author"]].add(metas[i]["book"])
    top = [a for a, _ in sorted(nbooks.items(), key=lambda kv: -len(kv[1]))[:8]]
    idx = [i for i in secure if metas[i]["author"] in top]
    X, names = fams.select(SPEC, idx)
    Z = zscore(X, idx)[idx]
    S, V, var = pca(Z, 2)
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=SURFACE)
    style(ax)
    for k, a in enumerate(top):
        rows = [j for j, i in enumerate(idx) if metas[i]["author"] == a]
        ax.scatter(S[rows, 0], S[rows, 1], s=30, color=SLOTS[k], marker=MARKERS[k],
                   edgecolor=SURFACE, linewidth=0.8, label=author_name(a), zorder=3)
    ax.set_xlabel(f"PC1 ({var[0]:.0%} of variation)", color=INK2)
    ax.set_ylabel(f"PC2 ({var[1]:.0%} of variation)", color=INK2)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(out / "pca.png", dpi=160)
    plt.close(fig)
    load = {}
    for c in range(2):
        o = np.argsort(V[c])
        load[f"PC{c + 1}"] = {"negative": [names[j] for j in o[:10]],
                              "positive": [names[j] for j in o[::-1][:10]],
                              "variance": float(var[c])}
    json.dump({"authors": top, "loadings": load}, open(out / "pca_loadings.json", "w"),
              ensure_ascii=False, indent=1)


def fig_impostors(ver, out):
    pos = [t["score"] for t in ver["trials"] if t["true"]]
    neg = [t["score"] for t in ver["trials"] if not t["true"]]
    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=SURFACE)
    style(ax)
    bins = np.linspace(0, 1, 21)
    ax.hist(neg, bins=bins, color=SLOTS[1], alpha=0.85, label=f"false candidate ({len(neg)} trials)",
            density=True, edgecolor=SURFACE, linewidth=2)
    ax.hist(pos, bins=bins, color=SLOTS[0], alpha=0.85, label=f"true author ({len(pos)} trials)",
            density=True, edgecolor=SURFACE, linewidth=2)
    top = ax.get_ylim()[1]
    ax.axvline(ver["threshold"], color=INK, linewidth=1, linestyle="--")
    ax.text(ver["threshold"] + 0.01, top * 0.97, f"threshold {ver['threshold']:.2f}",
            color=INK, fontsize=8, va="top")
    for d in ver["disputed"]:            # disputed works as ticks on the baseline
        ax.plot([d["impostors_score"]] * 2, [0, top * 0.06], color=INK, linewidth=1.6)
    listing = "Disputed works (black ticks):\n" + "\n".join(
        f"{d['book'].split('.')[1]}: {d['impostors_score']:.2f}" for d in ver["disputed"])
    ax.text(0.42, top * 0.92, listing, color=INK, fontsize=8, va="top")
    ax.set_xlabel("Impostors score (share of rounds the candidate wins)", color=INK2)
    ax.set_ylabel("density", color=INK2)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)
    fig.tight_layout()
    fig.savefig(out / "impostors.png", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--out", default="reports/layer3")
    a = ap.parse_args()
    out = Path(a.out)
    ev = json.load(open(out / "evaluation.json", encoding="utf-8"))
    ver = json.load(open(out / "verification.json", encoding="utf-8"))
    metas, feats = load_sample(a.data_dir)
    fams = Families(feats)
    fig_accuracy(ev, out)
    order = fig_dendrogram(metas, fams, out)
    fig_pca(metas, fams, out)
    fig_impostors(ver, out)
    json.dump(order, open(out / "dendrogram_order.json", "w"), ensure_ascii=False, indent=1)
    print("figures written to", out)


if __name__ == "__main__":
    main()
