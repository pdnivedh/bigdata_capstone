import argparse, os
import numpy as np, pandas as pd
import scipy.sparse as sp


def evaluate(ranked, test, k=10, like_threshold=4.0, valid_items=None):
    t = test[test["rating"] >= like_threshold]
    if valid_items is not None:
        t = t[t["item_id"].isin(valid_items)]
    rel = t.groupby("user_id")["item_id"].apply(set)
    rows = []
    top = ranked[ranked["rank"] <= k].sort_values(["user_id", "rank"])
    for u, items in top.groupby("user_id")["item_id"]:
        truth = rel.get(u, set())
        if not truth:
            continue
        hits = [i in truth for i in items]
        dcg = sum(h / np.log2(p + 2) for p, h in enumerate(hits))
        idcg = sum(1 / np.log2(p + 2) for p in range(min(len(truth), k)))
        rows.append({"user_id": u, "precision": sum(hits) / k, "recall": sum(hits) / len(truth),
                     "ndcg": dcg / idcg, "hit": float(any(hits))})
    return pd.DataFrame(rows)


def itemcf(train):
    user_ids = np.sort(train["user_id"].unique())
    item_ids = np.sort(train["item_id"].unique())
    u_map = {u: i for i, u in enumerate(user_ids)}
    i_map = {m: j for j, m in enumerate(item_ids)}
    R = sp.csr_matrix((train["rating"].astype("float32"),
                       (train["user_id"].map(u_map), train["item_id"].map(i_map))),
                      shape=(len(user_ids), len(item_ids)))
    norms = np.sqrt(R.multiply(R).sum(axis=0)).A1
    Rn = R @ sp.diags(1.0 / norms)
    S = (Rn.T @ Rn).toarray().astype("float32")
    np.fill_diagonal(S, 0)
    scores = np.asarray(R @ S)
    rr, cc = R.nonzero()
    scores[rr, cc] = -np.inf
    return user_ids, item_ids, u_map, i_map, scores


def reorder(bir, score_col):
    d = bir[["user_id", "item_id", "lsh_rank", score_col]].copy()
    d = d.sort_values(["user_id", score_col, "lsh_rank"], ascending=[True, False, True])
    d["rank"] = d.groupby("user_id").cumcount() + 1
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    train = pd.read_parquet(f"{a.data}/ratings_train")
    test = pd.read_parquet(f"{a.data}/ratings_test")
    bir = pd.read_parquet(f"{a.data}/member2_birank_ranked.parquet")
    train_items = set(train["item_id"])
    users = bir["user_id"].unique()

    # popularity
    pop = train.groupby("item_id").size().sort_values(ascending=False)
    seen = train.groupby("user_id")["item_id"].apply(set)
    rows = []
    for u in users:
        s = seen.get(u, set())
        recs = [i for i in pop.index if i not in s][:100]
        rows += [(u, i, r + 1) for r, i in enumerate(recs)]
    pop_full = pd.DataFrame(rows, columns=["user_id", "item_id", "rank"])
    bir["pop"] = bir["item_id"].map(pop).fillna(0)
    pop_cand = reorder(bir, "pop")

    # item-based CF
    user_ids, item_ids, u_map, i_map, scores = itemcf(train)
    top = np.argsort(-scores, axis=1)[:, :100]
    cf_full = pd.DataFrame({"user_id": np.repeat(user_ids, 100),
                            "item_id": item_ids[top].ravel(),
                            "rank": np.tile(np.arange(1, 101), len(user_ids))})
    bir["cf"] = scores[bir["user_id"].map(u_map).values, bir["item_id"].map(i_map).values]
    cf_cand = reorder(bir, "cf")

    orders = {
        "LSH order": bir[["user_id", "item_id", "lsh_rank"]].rename(columns={"lsh_rank": "rank"}),
        "LSH + BiRank": bir[["user_id", "item_id", "rank"]],
        "Popularity (same candidates)": pop_cand,
        "Item-CF (same candidates)": cf_cand,
        "Popularity (full catalog)": pop_full,
        "Item-CF (full catalog)": cf_full,
    }
    out = []
    for setting, valid in {"strict": None, "graph_only": train_items}.items():
        for k in (10, 20):
            for name, order in orders.items():
                res = evaluate(order, test, k=k, valid_items=valid)
                out.append({"setting": setting, "K": k, "method": name, "users": len(res),
                            "precision": res["precision"].mean(), "recall": res["recall"].mean(),
                            "ndcg": res["ndcg"].mean(), "hit": res["hit"].mean()})
    df = pd.DataFrame(out)
    df.to_csv(f"{a.out}/summary_pipeline.csv", index=False)
    print(df[(df.setting == "strict") & (df.K == 10)].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
