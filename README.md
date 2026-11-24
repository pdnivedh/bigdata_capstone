
# Hybrid Recommendation System Using LSH and BiRank

Big Data and Cloud Computing Capstone Project

## Project Overview

This project combines Locality-Sensitive Hashing (LSH) for
candidate generation with personalized graph ranking (BiRank)
for movie recommendations.

## Member 1 - Data Processing and LSH

Implemented components:
- MovieLens dataset preprocessing using PySpark
- Data cleaning and validation
- Chronological 80/20 train-test splitting
- User and movie feature engineering
- MinHashLSH candidate generation
- Parameter tuning and candidate recall evaluation
- Popularity-based fallback
- Top-100 candidate generation for BiRank

## Dataset

MovieLens Latest Small (GroupLens)

- Users: 610
- Movies: 9,742
- Ratings: 100,836
- Training interactions: 80,419
- Testing interactions: 20,417

Dataset source:
https://grouplens.org/datasets/movielens/

## Requirements

- Python 3.14
- Java 17
- PySpark 4.1.3

Install dependencies:

    pip install -r requirements.txt

Download and extract `ml-latest-small.zip` into
`data/raw/` before running the pipeline.

## Running Member 1 Pipeline

Run these scripts from the repository root:

    python src/data_preprocessing.py
    python src/build_features.py
    python src/lsh_candidates.py
    python src/finalize_candidates.py

The current LSH configuration uses:
- Top 5 preferred genres per user
- MinHashLSH with 4 hash tables
- Jaccard distance threshold: 0.8
- Top 100 candidates per user
- Popularity-based tie-breaking

User profiles are derived from positively rated training
movies (rating >= 4.0).

## Evaluation

Development experiments:

| Configuration | Recall@100 | Hit Rate@100 | Runtime |
|---|---:|---:|---:|
| Top-3, 2 hash tables | 4.24% | 36.84% | 10.38s |
| Top-3, 4 hash tables | 4.40% | 38.20% | 15.36s |
| Top-3, 8 hash tables | 4.40% | 38.20% | 20.97s |
| Top-5, 4 hash tables | 4.92% | 40.07% | 18.48s |

Recall is calculated against held-out positive ratings
for movies present in the training graph.

The current Top-5 configuration performed best among
the configurations tested.

These are development results, not unbiased final-test
estimates, because configuration selection used the
held-out test set.

## Final Output

Location:

    data/processed/member1_candidates_top100

Parquet schema:

- user_id
- item_id
- similarity
- rank
- candidate_source

Results:
- 610 users covered
- 61,000 candidates
- 100 candidates per user
- 60,700 LSH-generated candidates
- 300 popularity fallback candidates
- No duplicate pairs
- No previously rated movies
- All candidate movies exist in the training graph

## Handoff to Member 2

Member 2 can use:

    data/processed/ratings_train
    data/processed/member1_candidates_top100

The training data contains the user-item interactions
needed to construct the bipartite graph.

The candidate file contains 100 movies per user for
BiRank re-ranking.

The Parquet files are generated locally and are excluded
from Git because they are derived datasets.

## Limitations

- Genre features do not capture all user preferences.
- Top-100 candidate recall remains relatively low.
- Cold-start movies without training interactions are
  excluded from the graph-based recommendation pipeline.
- Runtime measurements are single-run observations.
- A separate validation/final-test protocol is needed
  for unbiased final evaluation.



## Stage 2: Graph Construction and Personalized BiRank (Member 2)

This stage takes Member 1's 100 LSH candidates per user and re-ranks them with
personalized BiRank on a user-movie bipartite graph built from the full training
ratings. Its output is passed to Member 3 for evaluation.

```
ratings_train ──► bipartite graph (610 users x 8,230 movies, 80,419 edges)
                         │
member1_candidates_top100 ──► personalized BiRank ──► re-ranked candidates
```

The candidate list only decides **which movies may be ranked**. The graph always
uses **all** training interactions, not just the candidates.

### Files

| File | Purpose |
|---|---|
| `src/birank_rerank.py` | Tuning (`tune`) and final re-ranking (`rerank`). Pure pandas/scipy, no Spark needed. |
| `src/crosscheck_networkx.py` | Verifies our BiRank scores against `networkx.bipartite.birank`. |
| `birank_tuning_results.csv` | Validation results for every tuned configuration. |
| `data/processed/member2_birank_ranked.parquet` | All 100 candidates per user, re-ranked. |
| `data/processed/member2_birank_ranked_top20.parquet` | Top-20 per user. |
| `data/processed/member2_birank_ranked_settings.json` | Exact settings used for the final run. |

### Setup

```bash
pip install -r requirements.txt
pip install -U "networkx>=3.5"   # only needed for the cross-check
```

`src/birank_rerank.py` does not use Spark, so it runs on Windows without Hadoop/winutils.
(Member 1's Spark scripts do need a working Spark install; on Windows, running them
inside Docker or WSL avoids the `winutils.exe` error.)

### Usage

```bash
# 1. Tune on a validation split carved out of ratings_train (ratings_test is never read)
python src/birank_rerank.py tune --data data/processed --weights rating binary --alphas 0.7 0.85 0.95 --betas 0.05 0.1 0.2

# 2. Final re-ranking with the chosen settings
python src/birank_rerank.py rerank --data data/processed --weight rating --alpha 0.7 --beta 0.1 --topn 20

# 3. Verify against NetworkX
python src/crosscheck_networkx.py --data data/processed --weight rating --alpha 0.7 --beta 0.1
```

### Method

**Graph.** Users and movies are the two node types. Each training rating is an edge,
weighted by the rating value (1 to 5). Three schemes were compared: `binary`,
`rating` and `thresholded` (only ratings of 4 or higher create edges).

**BiRank.** With `W` the user-movie weight matrix and `S = D_u^(-1/2) W D_p^(-1/2)`:

```
p <- alpha * S^T u + (1 - alpha) * p0      (movie scores; alpha = movie-side damping)
u <- beta  * S   p + (1 - beta)  * u0      (user scores;  beta  = user-side damping)
```

iterated until the largest score change is below `1e-6`. For each target user, `u0` is
the indicator of that user's node and `p0` is that user's rating history normalised to
sum to 1, so every user receives a personal ranking. All 610 users are computed in one
batch with sparse matrices.

**Re-ranking.** Each user's 100 candidates are sorted by BiRank score. Ties are broken by
the original LSH rank, so results are deterministic. A user with no training ratings would
keep the LSH order (none of the 610 users are affected).

### Tuning protocol (no test data used)

The last 20% of each user's *training* ratings (chronological) are held out as a
validation set, the graph is built from the remaining 80%, and Recall@100 is measured
over **all unseen movies**, averaged over users. This is not restricted to the LSH
candidates because candidates exclude already-rated movies and so cannot contain
validation items.

| Edge weight | alpha | beta | Validation Recall@100 |
|---|---|---|---|
| **rating** | **0.70** | **0.10** | **0.4122** (final choice) |
| rating | 0.70 | 0.05 | 0.4118 |
| rating | 0.85 | 0.10 | 0.4105 |
| rating | 0.50 | 0.50 | 0.4021 |
| binary | 0.85 | 0.50 | 0.3988 |
| thresholded | 0.50 | 0.50 | 0.3762 |
| *popularity baseline (most-rated)* | | | *0.2542* |

BiRank converged in 5 iterations with the final settings. Small `beta` values worked
consistently best, and the top configurations differ by less than 0.003, so the exact
optimum is not distinguishable from a single validation split.

### Verification

`src/crosscheck_networkx.py` rebuilds the same graph and personalization in NetworkX and
compares rankings over unseen movies. For five random users: Spearman correlation
`1.0000` and `100/100` top-100 movies in common.

### Output schema (`member2_birank_ranked.parquet`)

| Column | Meaning |
|---|---|
| `user_id`, `item_id` | User and candidate movie |
| `similarity` | Member 1's LSH genre similarity (0 for popularity fallback) |
| `lsh_rank` | Member 1's original candidate rank (1 to 100) |
| `candidate_source` | `lsh` or `popularity_fallback` |
| `birank_score` | Personalized BiRank score |
| `birank_used` | `True` if the user had training ratings and was re-ranked |
| `rank` | **New BiRank rank (1 to 100).** Use this for the proposed pipeline. |

To compare LSH order with BiRank order on the same candidates, sort by `lsh_rank`
versus `rank`.

### Important notes

- `ratings_test` was never used for graph construction or tuning.
- Validation Recall@100 (about 0.41) measures BiRank over all unseen movies. It is **not**
  comparable to the final pipeline, which re-ranks only 100 candidates whose own recall
  is low (4.92% in development). BiRank's final gain is bounded by candidate quality.
- Tuning used a single validation split.
- Member 1 selected the LSH configuration using test ratings, so absolute candidate
  recall is optimistic. All ranking methods share the same candidates, so comparisons
  between them remain fair.
- The training graph has 8,230 movies, while Member 1 reports 8,215 movies in the LSH
  catalog. The difference is probably movies without genre tags being dropped from the
  LSH features (to be confirmed by Member 1).


## Member 3: Baselines, evaluation and deployment

### Evaluation protocol
- A test item is relevant if its rating is >= 4 ("liked"). Users with no liked test items are skipped, which leaves 592 of 610 users.
- Metrics: Precision@K, Recall@K, NDCG@K, Hit rate@K at K = 10 (primary) and K = 20. The full set, including K = 20 and a graph-only variant, is in `results/summary_pipeline.csv`.
- `ratings_test` was used once for the final numbers. ALS was tuned only on a validation split taken from `ratings_train` (last 20% of each user's history); frozen settings: rank 30, regParam 0.1, maxIter 20, implicit feedback.

### Results (strict relevance, K = 10)

Same 100 LSH candidates per user (isolates ranking quality):

| Method | Precision@10 | Recall@10 | NDCG@10 | Hit rate@10 |
|---|---|---|---|---|
| LSH order | 0.0103 | 0.0107 | 0.0135 | 0.0946 |
| Popularity | 0.0296 | 0.0273 | 0.0416 | 0.2230 |
| Item-based CF | 0.0340 | 0.0333 | 0.0493 | 0.2568 |
| ALS | 0.0331 | 0.0330 | 0.0478 | 0.2568 |
| **LSH + BiRank** | **0.0392** | **0.0357** | **0.0557** | **0.2855** |

End to end (baselines rank the whole unseen catalog; the pipeline is limited to its 100 candidates):

| Method | Precision@10 | Recall@10 | NDCG@10 | Hit rate@10 |
|---|---|---|---|---|
| Popularity | 0.0578 | 0.0487 | 0.0753 | 0.3226 |
| Item-based CF | 0.0726 | 0.0730 | 0.0901 | 0.4274 |
| ALS | 0.0544 | 0.0692 | 0.0713 | 0.3666 |
| LSH + BiRank | 0.0392 | 0.0357 | 0.0557 | 0.2855 |

### Is the difference real? (paired, per user, 592 users)
Mean difference LSH + BiRank minus the other method, bootstrap 95% CI, Wilcoxon signed-rank p:

| Comparison (same candidates) | NDCG@10 diff (95% CI) | p | Recall@10 diff (95% CI) |
|---|---|---|---|
| vs LSH order | +0.0422 (0.0338 to 0.0508) | < 0.0001 | +0.0250 (0.0184 to 0.0321) |
| vs Popularity | +0.0142 (0.0098 to 0.0190) | < 0.0001 | +0.0084 (0.0058 to 0.0113) |
| vs Item-based CF | +0.0065 (0.0025 to 0.0106) | 0.0012 | +0.0024 (-0.0004 to 0.0051) |
| vs ALS | +0.0079 (0.0026 to 0.0134) | 0.0033 | +0.0027 (-0.0002 to 0.0057) |

BiRank clearly beats the LSH order and popularity ordering. Against Item-CF and ALS it has a small NDCG@10 advantage and no reliable recall difference. These are 16 tests on one split without multiple-comparison correction.

### Runtime
Google Colab (2 vCPU, 12 GB RAM, Intel Xeon 2.20 GHz):

| Stage | Wall-clock |
|---|---|
| Item-CF (build + scoring) | 1.5 s |
| ALS final training | 21.5 s |
| BiRank re-rank (mean of 3 runs) | 5.4 s |
| Preprocessing | 65.0 s |
| Build features | 51.4 s |
| LSH candidates | 106.7 s |
| Finalize candidates | 41.8 s |

Airflow in Docker (AMD Ryzen 5 7235HS, 8 logical processors), one run: preprocessing 1 m 29 s, build features 21.9 s, LSH candidates 52.9 s, finalize candidates 25.4 s, BiRank 2.1 s, evaluate 4.3 s; full DAG run 3 m 19 s. Spark times include JVM start-up and are single runs; the two machines are not comparable with each other.

### Deployment
- `Dockerfile` (Python 3.12 + Java 17) runs every script; `docker build -t recsys-pipeline .`.
- `airflow/` holds the Docker Compose setup and the DAG `recsys_lsh_birank` (`data_preprocessing -> build_features -> lsh_candidates -> finalize_candidates -> birank_rerank -> evaluate`). To reproduce: create `airflow/.env` with `AIRFLOW_UID=50000` and `AIRFLOW_IMAGE_NAME=recsys-airflow:latest`, run `docker build -f airflow/Dockerfile -t recsys-airflow:latest .` from the repo root, then `docker compose up airflow-init` and `docker compose up -d` inside `airflow/`, and trigger the DAG at http://localhost:8080 (login airflow / airflow).
- The DAG regenerates the candidates from raw MovieLens and reproduced the same metrics as the Colab run.
- ALS runs in the notebook `notebooks/member3_evaluation.ipynb` (Spark), not in the DAG. The optional FastAPI endpoint was not built.

### Limitations
- The LSH configuration was chosen using test ratings, so absolute candidate recall is optimistic. Every method that uses the candidates shares this, so comparisons among them are fair.
- BiRank was tuned on a validation split only, and ALS only on a validation split from `ratings_train`; the top ALS configurations differed by about 0.005 in validation recall.
- Candidate recall caps the pipeline's accuracy: full-catalog Item-CF, ALS and popularity beat it end to end.
- 8.34% of test rows (6.59% of liked ones) are movies absent from the training graph and cannot be recommended by any graph-based method.
- On this small dataset the pipeline is slower than the baselines. Any claim that LSH scales better is an expectation, not something measured here.
- Single dataset (MovieLens small), single chronological split, single-run timings, and different hardware for Colab and Airflow.

