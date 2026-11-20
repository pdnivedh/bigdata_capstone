
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
