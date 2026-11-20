
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("EvaluateLSH")
    .master("local[2]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# Load training and testing data
train = spark.read.parquet(
    "data/processed/ratings_train"
)

test = spark.read.parquet(
    "data/processed/ratings_test"
)

users = spark.read.parquet(
    "data/processed/user_features"
).select("user_id")

# Only movies present in the training graph
train_items = train.select("item_id").distinct()

# Relevant held-out movies (rating >= 4)
relevant = (
    test.filter(F.col("rating") >= 4.0)
    .select("user_id", "item_id")
    .dropDuplicates()
    .join(train_items, "item_id", "inner")
    .join(users, "user_id", "inner")
    .cache()
)

total_relevant = relevant.count()
total_users = relevant.select("user_id").distinct().count()

print("\n===== LSH EVALUATION =====")
print("Eligible relevant test interactions:", total_relevant)
print("Users with relevant test movies:", total_users)

experiments = {
    "Top-3, 2 tables": "data/processed/lsh_candidates_tuning_2",
    "Top-3, 4 tables": "data/processed/lsh_candidates_top100",
    "Top-3, 8 tables": "data/processed/lsh_candidates_tuning_8",
    "Top-5, 4 tables": "data/processed/lsh_candidates_top5_4"
}

for name, path in experiments.items():

    candidates = (
        spark.read.parquet(path)
        .select("user_id", "item_id")
        .dropDuplicates()
    )

    # Find relevant test movies retrieved by LSH
    hits = relevant.join(
        candidates,
        on=["user_id", "item_id"],
        how="inner"
    )

    hit_count = hits.count()

    # Micro-averaged candidate recall
    recall = (
        hit_count / total_relevant
        if total_relevant > 0 else 0.0
    )

    # Percentage of eligible users with at least one hit
    users_with_hits = hits.select(
        "user_id"
    ).distinct().count()

    hit_rate = (
        users_with_hits / total_users
        if total_users > 0 else 0.0
    )

    print(f"\n{name}")
    print("Relevant movies retrieved:", hit_count)
    print(f"Candidate Recall@100: {recall:.4f}")
    print(f"User Hit Rate@100: {hit_rate:.4f}")

print("\nEvaluation completed successfully!")

spark.stop()
