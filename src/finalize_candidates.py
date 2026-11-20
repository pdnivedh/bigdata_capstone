
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = (
    SparkSession.builder
    .appName("FinalizeLSHCandidates")
    .master("local[2]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# Load existing LSH candidates
lsh_candidates = spark.read.parquet(
    "data/processed/lsh_candidates_top5_4"
)

train = spark.read.parquet(
    "data/processed/ratings_train"
)

# Find users without LSH candidates
all_users = train.select("user_id").distinct()

covered_users = lsh_candidates.select(
    "user_id"
).distinct()

missing_users = all_users.join(
    covered_users,
    on="user_id",
    how="left_anti"
)

print("Users requiring fallback:", missing_users.count())

# Calculate popularity using training data only
popularity = train.groupBy("item_id").agg(
    F.count("*").alias("popularity")
)

# Find movies users have already rated
watched = train.select(
    "user_id", "item_id"
).distinct()

# Generate possible recommendations
fallback = missing_users.crossJoin(popularity)

# Exclude already rated movies
fallback = fallback.join(
    watched,
    on=["user_id", "item_id"],
    how="left_anti"
)

# Rank by movie popularity
window = Window.partitionBy(
    "user_id"
).orderBy(
    F.desc("popularity"),
    F.asc("item_id")
)

fallback = (
    fallback
    .withColumn("rank", F.row_number().over(window))
    .filter(F.col("rank") <= 100)
    .select(
        "user_id",
        "item_id",
        F.lit(0.0).alias("similarity"),
        "rank"
    )
)

# Identify the recommendation source
lsh_candidates = lsh_candidates.withColumn(
    "candidate_source",
    F.lit("lsh")
)

fallback = fallback.withColumn(
    "candidate_source",
    F.lit("popularity_fallback")
)

# Combine both methods
final_candidates = lsh_candidates.unionByName(
    fallback
)

# Save final Member 1 handoff
final_candidates.write.mode("overwrite").parquet(
    "data/processed/member1_candidates_top100"
)

print("\n===== FINAL RESULTS =====")

print(
    "Total users:",
    final_candidates.select("user_id").distinct().count()
)

print("Total candidates:", final_candidates.count())

# Validate that no previously rated movies are recommended
seen_count = final_candidates.join(
    watched,
    on=["user_id", "item_id"],
    how="inner"
).count()

print("Previously rated movies recommended:", seen_count)

print("\nFinal candidate dataset saved successfully!")

spark.stop()
