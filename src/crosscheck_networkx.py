"""
Cross-check birank_rerank.py against networkx.bipartite.birank for a few users.

  python src/crosscheck_networkx.py --data data/processed --alpha 0.5 --beta 0.5 --weight rating

Convention (same in both): ITEMS use alpha, USERS use beta.
Needs networkx >= 3.5 (older versions have no bipartite.birank).
"""
import argparse
import sys

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "src")
from birank_rerank import birank_all_users, build_graph, load_train  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/processed")
ap.add_argument("--alpha", type=float, default=0.85)
ap.add_argument("--beta", type=float, default=0.85)
ap.add_argument("--weight", default="binary", choices=["binary", "rating", "thresholded"])
ap.add_argument("--n-users", type=int, default=5)
args = ap.parse_args()

train = load_train(args.data)
W, uids, iids = build_graph(train, weight=args.weight)
P, n_it, _ = birank_all_users(W, alpha=args.alpha, beta=args.beta, tol=1e-10, max_iter=500)

# build the same graph in networkx (items = "top" set, users = "bottom" set)
Wc = W.tocoo()
G = nx.Graph()
items = [("i", int(i)) for i in iids]
G.add_nodes_from(items, bipartite=0)
G.add_nodes_from([("u", int(u)) for u in uids], bipartite=1)
G.add_weighted_edges_from(
    (("u", int(uids[r])), ("i", int(iids[c])), float(v)) for r, c, v in zip(Wc.row, Wc.col, Wc.data)
)

rng = np.random.default_rng(0)
for j in rng.choice(len(uids), size=args.n_users, replace=False):
    row = W.getrow(j)
    hist = {("i", int(iids[c])): float(v) for c, v in zip(row.indices, row.data)}
    s = sum(hist.values())
    hist = {k: v / s for k, v in hist.items()}                 # same item prior as our code

    scores = nx.bipartite.birank(
        G, items, alpha=args.alpha, beta=args.beta,
        top_personalization=hist, bottom_personalization={("u", int(uids[j])): 1.0},
        max_iter=500, tol=1e-10,
    )
    nx_vec = np.array([scores[("i", int(i))] for i in iids])
    mine = P[:, j]
    unseen = np.setdiff1d(np.arange(len(iids)), row.indices)    # compare on items we would recommend
    rho = spearmanr(nx_vec[unseen], mine[unseen]).correlation
    top_nx = set(np.argsort(-nx_vec[unseen])[:100])
    top_me = set(np.argsort(-mine[unseen])[:100])
    print(f"user {int(uids[j]):>4}: Spearman={rho:.4f}  top-100 overlap={len(top_nx & top_me)}/100")
