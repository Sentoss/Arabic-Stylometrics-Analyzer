"""Stylometric methods: feature matrices, Delta, classifiers, Impostors, Zeta,
clustering, PCA.

Conventions: X is a (samples x features) matrix of relative frequencies;
z-scores always use statistics from the *training* samples only.
"""
from collections import Counter

import numpy as np

# ------------------------------------------------------------------ matrices
FAMILIES = ("surface", "lemma", "function", "pos", "pattern", "char")


def choose_vocab(feats, idx, family, n):
    """The n most frequent features of a family, by summed relative frequency
    over the training samples idx (so the test book never chooses features)."""
    tot = Counter()
    for i in idx:
        c = feats[i][family]
        s = sum(c.values()) or 1
        for k, v in c.items():
            tot[k] += v / s
    return [k for k, _ in tot.most_common(n)]


def matrix(feats, family, vocab):
    X = np.zeros((len(feats), len(vocab)))
    col = {k: j for j, k in enumerate(vocab)}
    for i, f in enumerate(feats):
        c = f[family]
        s = sum(c.values()) or 1
        for k, v in c.items():
            j = col.get(k)
            if j is not None:
                X[i, j] = v / s
    return X


def build(feats, train_idx, spec):
    """spec: list of (family, n). Returns (matrix for all samples, feature names)."""
    blocks, names = [], []
    for fam, n in spec:
        v = choose_vocab(feats, train_idx, fam, n)
        blocks.append(matrix(feats, fam, v))
        names += [f"{fam}|{k}" for k in v]
    return np.hstack(blocks), names


class Families:
    """Relative-frequency matrices for every feature family, computed once.
    select() then picks the n most frequent features of the training rows."""

    def __init__(self, feats, families=FAMILIES, min_docs=2):
        from scipy.sparse import csr_matrix
        self.n = len(feats)
        self.mats, self.names = {}, {}
        for fam in families:
            df = Counter(k for f in feats for k in f[fam])
            vocab = [k for k, c in df.items() if c >= min_docs]
            col = {k: j for j, k in enumerate(vocab)}
            rows, cols, vals = [], [], []
            for i, f in enumerate(feats):
                c = f[fam]
                s = sum(c.values()) or 1
                for k, v in c.items():
                    j = col.get(k)
                    if j is not None:
                        rows.append(i); cols.append(j); vals.append(v / s)
            self.mats[fam] = csr_matrix((vals, (rows, cols)), shape=(self.n, len(vocab)))
            self.names[fam] = vocab

    def select(self, spec, train_idx):
        blocks, names = [], []
        for fam, n in spec:
            M = self.mats[fam]
            mean = np.asarray(M[train_idx].mean(0)).ravel()
            top = np.argsort(-mean)[:n]
            blocks.append(M[:, top].toarray())
            names += [f"{fam}|{self.names[fam][j]}" for j in top]
        return np.hstack(blocks), names


def zscore(X, train_idx):
    mu = X[train_idx].mean(0)
    sd = X[train_idx].std(0)
    sd[sd == 0] = 1
    return (X - mu) / sd


# ------------------------------------------------------------------ distances
def burrows_delta(A, B):
    """Mean absolute z-score difference between every row of A and every row of B."""
    return np.abs(A[:, None, :] - B[None, :, :]).mean(2)


def cosine_delta(A, B):
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return 1 - An @ Bn.T


def centroids(Z, labels, classes):
    return np.vstack([Z[np.array(labels) == c].mean(0) for c in classes])


# ------------------------------------------------------------------ classifiers
def fit_predict(kind, Ztr, ytr, Zte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    if kind == "logreg":
        m = LogisticRegression(C=0.1, max_iter=2000)
    elif kind == "svm":
        m = LinearSVC(C=0.01, max_iter=5000)
    else:
        raise ValueError(kind)
    m.fit(Ztr, ytr)
    return m.predict(Zte), m


# ------------------------------------------------------------------ impostors
def impostors_score(Zx, cand, pool, iters=200, feat_frac=0.5, n_imp=10, rng=None,
                    dist=cosine_delta):
    """General Impostors (after Koppel & Winter 2014), with author profiles.

    Zx    z-scored samples of the questioned text
    cand  profile (mean z vector) of the candidate author, from his secure works
    pool  dict impostor_name -> profile
    Each round draws half the features and n_imp impostors; the candidate wins
    a round if the questioned text (its mean profile) is closer to him than to
    every impostor drawn. Returns the share of rounds won (0-1).
    """
    rng = rng or np.random.default_rng(0)
    x = Zx.mean(0, keepdims=True)
    names = list(pool)
    P = np.vstack([pool[k] for k in names])
    d = x.shape[1]
    wins = 0
    for _ in range(iters):
        f = rng.choice(d, int(d * feat_frac), replace=False)
        imp = rng.choice(len(names), min(n_imp, len(names)), replace=False)
        dc = dist(x[:, f], cand[None, f])[0, 0]
        di = dist(x[:, f], P[imp][:, f])[0]
        wins += dc < di.min()
    return wins / iters


# ------------------------------------------------------------------ explanation
def contributions(zx, zc, names, k=12):
    """Features that most separate a questioned profile from a candidate's:
    returns [(feature, z_questioned, z_candidate)] sorted by |difference|."""
    diff = zx - zc
    order = np.argsort(-np.abs(diff))[:k]
    return [(names[j], float(zx[j]), float(zc[j])) for j in order]


def zeta(feats_a, feats_b, family="lemma", min_docs=3, k=15):
    """Craig's Zeta: share of A's samples containing a word minus share of B's.
    High = A's consistent habit that B avoids; low = the reverse."""
    da = Counter(w for f in feats_a for w in f[family])
    db = Counter(w for f in feats_b for w in f[family])
    words = [w for w in set(da) | set(db) if da[w] + db[w] >= min_docs]
    na, nb = len(feats_a), len(feats_b)
    z = sorted(((da[w] / na - db[w] / nb, w) for w in words), reverse=True)
    return z[:k], z[-k:][::-1]


# ------------------------------------------------------------------ clustering
def consensus_linkage(profiles, n_clusters, feature_sets, frac=0.8, reps=10, seed=0):
    """Bootstrap consensus: cluster book profiles many times (different feature
    counts, random feature subsets), count how often each pair lands in the
    same cluster, and build the final tree on that co-membership."""
    from scipy.cluster.hierarchy import linkage, fcluster
    rng = np.random.default_rng(seed)
    n = profiles[0].shape[0]
    co = np.zeros((n, n))
    runs = 0
    for P in profiles:                      # one matrix per feature-set size
        for _ in range(reps):
            cols = rng.choice(P.shape[1], int(P.shape[1] * frac), replace=False)
            D = cosine_delta(P[:, cols], P[:, cols])
            iu = np.triu_indices(n, 1)
            L = linkage(D[iu], method="average")
            lab = fcluster(L, n_clusters, criterion="maxclust")
            co += lab[:, None] == lab[None, :]
            runs += 1
    co /= runs
    D = 1 - co
    iu = np.triu_indices(n, 1)
    return linkage(D[iu], method="average"), co


def pca(Z, k=2):
    Zc = Z - Z.mean(0)
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    var = S ** 2 / (S ** 2).sum()
    return Zc @ Vt[:k].T, Vt[:k], var[:k]
