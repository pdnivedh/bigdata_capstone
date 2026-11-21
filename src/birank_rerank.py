"""
Member 2: personalized BiRank re-ranking of Member 1's LSH candidates.

Modes
-----
  tune   : tune alpha/beta/edge-weight scheme on a VALIDATION split carved out of
           ratings_train (ratings_test is never read in this script).
  rerank : build graph from full ratings_train, run personalized BiRank, re-rank
           the 100 candidates per user, write the output parquet for Member 3.

Usage
-----
  python birank_rerank.py tune   --data data/processed
  python birank_rerank.py rerank --data data/processed --alpha 0.85 --beta 0.85 \
                                 --weight binary --out data/processed/member2_birank_ranked.parquet
"""
import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def _norm_cols(df):
    """Accept either user_id/item_id or userId/movieId style names."""
    rename = {"userId": "user_id", "movieId": "item_id", "movie_id": "item_id"}
    return df.rename(columns={k: v for k, v in rename.items() if k in df.columns})


def load_train(data_dir):
    path = os.path.join(data_dir, "ratings_train")
    df = _norm_cols(pd.read_parquet(path))
    need = {"user_id", "item_id", "rating"}
    assert need <= set(df.columns), f"ratings_train missing {need - set(df.columns)}; has {list(df.columns)}"
    return df


def load_candidates(data_dir):
    path = os.path.join(data_dir, "member1_candidates_top100")
    return _norm_cols(pd.read_parquet(path))


# ----------------------------------------------------------------------------
# Graph
# ----------------------------------------------------------------------------
def build_graph(train, weight="binary", like_threshold=4.0):
    """Return (W, user_ids, item_ids). W is users x items, sparse CSR."""
    df = train[["user_id", "item_id", "rating"]].copy()
    if weight == "thresholded":
        df = df[df["rating"] >= like_threshold]
        df["w"] = 1.0
    elif weight == "rating":
        df["w"] = df["rating"].astype(float)
    elif weight == "binary":
        df["w"] = 1.0
    else:
        raise ValueError(weight)

    user_ids = np.sort(df["user_id"].unique())
    item_ids = np.sort(df["item_id"].unique())
    u_idx = pd.Index(user_ids).get_indexer(df["user_id"])
    i_idx = pd.Index(item_ids).get_indexer(df["item_id"])
    W = sp.csr_matrix((df["w"].values, (u_idx, i_idx)), shape=(len(user_ids), len(item_ids)))
    return W, user_ids, item_ids


def symmetric_normalize(W):
    """S = D_u^{-1/2} W D_p^{-1/2} (BiRank paper, Eq. for symmetric normalisation)."""
    du = np.asarray(W.sum(axis=1)).ravel()
    dp = np.asarray(W.sum(axis=0)).ravel()
    du_inv = np.where(du > 0, 1.0 / np.sqrt(np.maximum(du, 1e-12)), 0.0)
    dp_inv = np.where(dp > 0, 1.0 / np.sqrt(np.maximum(dp, 1e-12)), 0.0)
    return sp.diags(du_inv) @ W @ sp.diags(dp_inv)


# ----------------------------------------------------------------------------
# BiRank (batched over all users at once)
# ----------------------------------------------------------------------------
def birank_all_users(W, alpha=0.85, beta=0.85, tol=1e-6, max_iter=200, item_prior=True):
    """
    Personalised BiRank for every user simultaneously.

    Column j of the returned item-score matrix is the ranking for user j.
      u0 (user query)  : indicator of the target user
      p0 (item query)  : the user's training history, normalised to sum 1
                         (set item_prior=False to personalise on the user node only)
    Iteration (He et al., BiRank):
        p <- alpha * S^T u + (1 - alpha) * p0
        u <- beta  * S   p + (1 - beta)  * u0
    Returns (P: n_items x n_users, n_iter, final_delta)
    """
    S = symmetric_normalize(W).tocsr()
    St = S.T.tocsr()
    n_users, n_items = W.shape

    U0 = np.eye(n_users)
    if item_prior:
        P0 = np.asarray(W.T.todense(), dtype=float)          # n_items x n_users
        colsum = P0.sum(axis=0, keepdims=True)
        P0 = np.divide(P0, colsum, out=np.zeros_like(P0), where=colsum > 0)
    else:
        P0 = np.zeros((n_items, n_users))

    P, U = P0.copy(), U0.copy()
    delta = np.inf
    for it in range(1, max_iter + 1):
        P_new = alpha * (St @ U) + (1 - alpha) * P0
        U_new = beta * (S @ P_new) + (1 - beta) * U0
        delta = max(np.abs(P_new - P).max(), np.abs(U_new - U).max())
        P, U = P_new, U_new
        if delta < tol:
            break
    return P, it, delta


# ----------------------------------------------------------------------------
# Re-ranking the candidates
# ----------------------------------------------------------------------------
def rerank_candidates(cands, P, user_ids, item_ids, train_users):
    """Attach BiRank scores to candidates and re-rank within each user."""
    u_pos = pd.Series(np.arange(len(user_ids)), index=user_ids)
    i_pos = pd.Series(np.arange(len(item_ids)), index=item_ids)

    c = cands.copy()
    c = c.rename(columns={"rank": "lsh_rank"})
    c["u_i"] = c["user_id"].map(u_pos)
    c["i_i"] = c["item_id"].map(i_pos)

    has_graph = c["u_i"].notna() & c["i_i"].notna()
    c["birank_score"] = np.nan
    rows = c.loc[has_graph]
    c.loc[has_graph, "birank_score"] = P[rows["i_i"].astype(int).values, rows["u_i"].astype(int).values]

    # users with no training ratings (cold start) keep original LSH order
    c["birank_used"] = c["user_id"].isin(train_users)          # per USER, not per row
    c["birank_score"] = c["birank_score"].fillna(0.0)          # item missing from graph -> ranked last

    # sort: BiRank score desc, ties broken by original LSH rank (deterministic)
    c = c.sort_values(["user_id", "birank_score", "lsh_rank"], ascending=[True, False, True])
    c["rank"] = c.groupby("user_id").cumcount() + 1

    # cold-start users: restore LSH order
    cold = ~c["birank_used"]
    c.loc[cold, "rank"] = c.loc[cold, "lsh_rank"]
    c = c.sort_values(["user_id", "rank"])
    return c.drop(columns=["u_i", "i_i"]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Validation (inside ratings_train only)
# ----------------------------------------------------------------------------
def make_validation_split(train, frac=0.2, seed=42):
    """Hold out the last `frac` of each user's TRAIN interactions (by time if available)."""
    df = train.copy()
    if "timestamp" in df.columns:
        df = df.sort_values(["user_id", "timestamp"])
    else:
        df = df.sample(frac=1.0, random_state=seed).sort_values("user_id", kind="stable")
    pos = df.groupby("user_id").cumcount()
    size = df.groupby("user_id")["item_id"].transform("size")
    is_val = pos >= (size * (1 - frac)).round().astype(int)
    # keep at least one interaction in the inner-train graph
    return df[~is_val], df[is_val]


def recall_at_k(P, W, user_ids, item_ids, val, k=100):
    """Recall@k over ALL unseen items (not restricted to candidates)."""
    u_pos = pd.Series(np.arange(len(user_ids)), index=user_ids)
    i_pos = pd.Series(np.arange(len(item_ids)), index=item_ids)
    val = val[val["user_id"].isin(user_ids) & val["item_id"].isin(item_ids)]
    truth = val.groupby("user_id")["item_id"].apply(set)

    Wd = W.tocsr()
    recalls = []
    for u, rel in truth.items():
        j = u_pos[u]
        scores = P[:, j].copy()
        seen = Wd[j].indices
        scores[seen] = -np.inf                       # never recommend already-seen items
        top = np.argpartition(-scores, k)[:k]
        top_items = set(item_ids[top])
        recalls.append(len(top_items & rel) / len(rel))
    return float(np.mean(recalls))


def tune(args):
    train = load_train(args.data)
    inner, val = make_validation_split(train)
    results = []
    for weight, alpha, beta in itertools.product(args.weights, args.alphas, args.betas):
        W, uids, iids = build_graph(inner, weight=weight)
        P, n_it, delta = birank_all_users(W, alpha=alpha, beta=beta)
        r = recall_at_k(P, W, uids, iids, val, k=args.k)
        results.append(dict(weight=weight, alpha=alpha, beta=beta, recall_at_k=r, iters=n_it))
        print(f"weight={weight:11s} alpha={alpha:.2f} beta={beta:.2f}  Recall@{args.k}={r:.4f}  iters={n_it}")
    # popularity baseline for context
    pop = np.asarray(W.sum(axis=0)).ravel()
    P_pop = np.tile(pop[:, None], (1, W.shape[0]))
    r_pop = recall_at_k(P_pop, W, uids, iids, val, k=args.k)
    print(f"\npopularity baseline Recall@{args.k} = {r_pop:.4f}")
    res = pd.DataFrame(results).sort_values("recall_at_k", ascending=False)
    res.to_csv(args.tune_out, index=False)
    print("\nbest:\n", res.head(3).to_string(index=False))
    print(f"\nsaved -> {args.tune_out}")


# ----------------------------------------------------------------------------
# Final run
# ----------------------------------------------------------------------------
def rerank(args):
    train = load_train(args.data)
    cands = load_candidates(args.data)

    # --- sanity checks on Member 1's handoff -------------------------------
    seen = set(zip(train["user_id"], train["item_id"]))
    overlap = sum((u, i) in seen for u, i in zip(cands["user_id"], cands["item_id"]))
    print(f"candidates: {len(cands)} rows, {cands['user_id'].nunique()} users; already-rated overlap = {overlap}")
    assert overlap == 0, "candidates contain already-rated movies"

    W, uids, iids = build_graph(train, weight=args.weight)
    print(f"graph: {W.shape[0]} users x {W.shape[1]} items, {W.nnz} edges (weight={args.weight})")
    P, n_it, delta = birank_all_users(W, alpha=args.alpha, beta=args.beta)
    print(f"BiRank converged in {n_it} iterations (final delta {delta:.2e})")

    out = rerank_candidates(cands, P, uids, iids, set(uids))
    n_cold = out.loc[~out["birank_used"], "user_id"].nunique()
    print(f"cold-start users (kept LSH order): {n_cold}")

    out.to_parquet(args.out, index=False)
    topn = out[out["rank"] <= args.topn]
    topn.to_parquet(args.out.replace(".parquet", f"_top{args.topn}.parquet"), index=False)

    settings = dict(weight=args.weight, alpha=args.alpha, beta=args.beta, tol=1e-6,
                    iterations=n_it, topn=args.topn, cold_start_users=int(n_cold),
                    n_users=int(W.shape[0]), n_items=int(W.shape[1]), n_edges=int(W.nnz))
    with open(args.out.replace(".parquet", "_settings.json"), "w") as f:
        json.dump(settings, f, indent=2)
    print("saved:", args.out, "| settings:", settings)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tune", "rerank"])
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--beta", type=float, default=0.85)
    ap.add_argument("--weight", default="binary", choices=["binary", "rating", "thresholded"])
    ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--out", default="data/processed/member2_birank_ranked.parquet")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--weights", nargs="+", default=["binary", "rating", "thresholded"])
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.5, 0.7, 0.85, 0.95])
    ap.add_argument("--betas", nargs="+", type=float, default=[0.5, 0.7, 0.85, 0.95])
    ap.add_argument("--tune-out", default="birank_tuning_results.csv")
    args = ap.parse_args()
    tune(args) if args.mode == "tune" else rerank(args)
