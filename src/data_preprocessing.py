from pyspark.sql.window import Window
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Start Spark
spark = (
    SparkSession.builder
    .appName("MovieLensPreprocessing")
    .master("local[2]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# Load MovieLens datasets
ratings = spark.read.csv(
    "data/raw/ml-latest-small/ratings.csv",
    header=True,
    inferSchema=True
)

movies = spark.read.csv(
    "data/raw/ml-latest-small/movies.csv",
    header=True,
    inferSchema=True
)

# Dataset statistics
print("\n===== DATASET STATISTICS =====")

print("Total ratings:", ratings.count())
print("Total movies:", movies.count())
print("Total users:", ratings.select("userId").distinct().count())
print("Rated movies:", ratings.select("movieId").distinct().count())

# Display columns and data types
print("\n===== RATINGS SCHEMA =====")
ratings.printSchema()

print("\n===== MOVIES SCHEMA =====")
movies.printSchema()

# Display sample data
print("\n===== SAMPLE RATINGS =====")
ratings.show(5, truncate=False)

print("\n===== SAMPLE MOVIES =====")
movies.show(5, truncate=False)

# Check missing values in important columns
print("\n===== MISSING VALUES =====")
ratings.select([
    F.count(F.when(F.col(c).isNull(), c)).alias(c)
    for c in ratings.columns
]).show()

# ===== DATA CLEANING =====

print("\n===== CLEANING DATA =====")

# Standardize column names
cleaned = ratings.select(
    F.col("userId").alias("user_id"),
    F.col("movieId").alias("item_id"),
    "rating",
    "timestamp"
)

# Remove null and invalid values
cleaned = cleaned.dropna().filter(
    (F.col("user_id") > 0) &
    (F.col("item_id") > 0) &
    (F.col("rating").between(0.5, 5.0)) &
    (F.col("timestamp") > 0)
)

# Keep only movies present in movie metadata
valid_movies = movies.select(
    F.col("movieId").alias("item_id")
).dropDuplicates()

cleaned = cleaned.join(
    valid_movies, on="item_id", how="inner"
)

# If a user rated a movie more than once,
# retain the most recent rating
window = Window.partitionBy(
    "user_id", "item_id"
).orderBy(F.desc("timestamp"), F.desc("rating"))

cleaned = (
    cleaned.withColumn("row_num", F.row_number().over(window))
    .filter(F.col("row_num") == 1)
    .drop("row_num")
)

print("Cleaned ratings:", cleaned.count())

# Save cleaned dataset as Parquet
cleaned.write.mode("overwrite").parquet(
    "data/processed/ratings_clean"
)

print("Cleaned dataset saved successfully!")


# ===== TRAIN / TEST SPLIT =====

print("\n===== TRAIN / TEST SPLIT =====")

# Sort each user's ratings chronologically
user_window = Window.partitionBy(
    "user_id"
).orderBy(
    F.asc("timestamp"),
    F.asc("item_id")
)

count_window = Window.partitionBy("user_id")

# Number each user's ratings and count them
split_data = (
    cleaned
    .withColumn(
        "row_num",
        F.row_number().over(user_window)
    )
    .withColumn(
        "total_ratings",
        F.count("*").over(count_window)
    )
)

# Reserve the latest 20% of each user's ratings
split_data = split_data.withColumn(
    "test_count",
    F.greatest(
        F.lit(1),
        F.ceil(F.col("total_ratings") * 0.2).cast("int")
    )
)

# Training data: older interactions
train = split_data.filter(
    F.col("row_num") <=
    F.col("total_ratings") - F.col("test_count")
)

# Testing data: most recent interactions
test = split_data.filter(
    F.col("row_num") >
    F.col("total_ratings") - F.col("test_count")
)

# Remove temporary columns
temporary_columns = [
    "row_num", "total_ratings", "test_count"
]

train = train.drop(*temporary_columns)
test = test.drop(*temporary_columns)

# Save as Parquet
train.write.mode("overwrite").parquet(
    "data/processed/ratings_train"
)

test.write.mode("overwrite").parquet(
    "data/processed/ratings_test"
)

print("Training records:", train.count())
print("Testing records:", test.count())

print("Total records:", train.count() + test.count())

print("\nTrain/test split completed successfully!")

spark.stop()
