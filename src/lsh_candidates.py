
import argparse
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import MinHashLSH

parser = argparse.ArgumentParser()
parser.add_argument("--top-genres", type=int, default=5)
parser.add_argument("--tables", type=int, default=4)
parser.add_argument("--threshold", type=float, default=0.8)
args = parser.parse_args()
TOP_K, TABLES, THRESHOLD = args.top_genres, args.tables, args.threshold
OUT_PATH = f"data/processed/lsh_candidates_top{TOP_K}_{TABLES}"

spark = (
    SparkSession.builder
    .appName("MovieLensLSH")
    .master("local[2]")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# Load feature vectors
users = spark.read.parquet(
    f"data/processed/user_features_top{TOP_K}"
)

items = spark.read.parquet(
    "data/processed/item_features"
)

train = spark.read.parquet(
    "data/processed/ratings_train"
)

# Keep only movies connected to the training graph
train_items = train.select("item_id").distinct()

items = items.join(
    train_items,
    on="item_id",
    how="inner"
)

print("Graph-connected movies:", items.count())

print("\n===== MINHASH LSH =====")

start_time = time.perf_counter()

# Create and train the MinHashLSH model
lsh = MinHashLSH(
    inputCol="features",
    outputCol="hashes",
    numHashTables=TABLES,
    seed=42
)

model = lsh.fit(items)

# Find approximate similar user-movie pairs
matches = model.approxSimilarityJoin(
    users,
    items,
    THRESHOLD,
    distCol="jaccard_distance"
)

# Extract useful columns
candidates = matches.select(
    F.col("datasetA.user_id").alias("user_id"),
    F.col("datasetB.item_id").alias("item_id"),
    F.col("jaccard_distance")
)

# Handle any duplicate candidate pairs
candidates = candidates.groupBy(
    "user_id", "item_id"
).agg(
    F.min("jaccard_distance").alias("jaccard_distance")
)

# Exclude movies already rated in training
watched = train.select(
    "user_id", "item_id"
).distinct()

candidates = candidates.join(
    watched,
    on=["user_id", "item_id"],
    how="left_anti"
)

# Convert distance to similarity
candidates = candidates.withColumn(
    "similarity",
    F.lit(1.0) - F.col("jaccard_distance")
)


# Calculate movie popularity from training data only
popularity = train.groupBy("item_id").agg(
    F.count("*").alias("popularity")
)

# Add popularity to LSH candidates
candidates = candidates.join(
    popularity,
    on="item_id",
    how="left"
).fillna({"popularity": 0})

# Rank candidate movies for every user
rank_window = Window.partitionBy(
    "user_id"
).orderBy(
    F.desc("similarity"),
    F.desc("popularity"),
    F.asc("item_id")
)

top_candidates = (
    candidates
    .withColumn("rank", F.row_number().over(rank_window))
    .filter(F.col("rank") <= 100)
    .select("user_id", "item_id", "similarity", "rank")
)

# Save output for the graph-ranking stage
top_candidates.write.mode("overwrite").parquet(
    OUT_PATH
)

elapsed = time.perf_counter() - start_time

print("\n===== TOP LSH CANDIDATES =====")
top_candidates.orderBy("user_id", "rank").show(
    30, truncate=False
)

print("Users receiving candidates:",
      top_candidates.select("user_id").distinct().count())

print("Total candidates:", top_candidates.count())

print(f"Execution time: {elapsed:.2f} seconds")
print("\nLSH candidate generation successful!")

spark.stop()
