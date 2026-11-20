from pyspark.sql.window import Window
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import CountVectorizer

spark = (
    SparkSession.builder
    .appName("MovieLensFeatureEngineering")
    .master("local[2]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# Load training data only
train = spark.read.parquet(
    "data/processed/ratings_train"
)

movies = spark.read.csv(
    "data/raw/ml-latest-small/movies.csv",
    header=True,
    inferSchema=True
)

# Convert movie genres into arrays
movie_genres = (
    movies.select(
        F.col("movieId").alias("item_id"),
        F.split(F.col("genres"), r"\|").alias("tokens")
    )
    .filter(~F.array_contains("tokens", "(no genres listed)"))
)

# Learn a shared genre vocabulary
vectorizer = CountVectorizer(
    inputCol="tokens",
    outputCol="features",
    binary=True,
    minDF=1.0
)

model = vectorizer.fit(movie_genres)

# Movie feature vectors
item_features = (
    model.transform(movie_genres)
    .select("item_id", "features")
)

# Use only positively rated training movies
liked_movies = (
    train.filter(F.col("rating") >= 4.0)
    .join(movie_genres, on="item_id", how="inner")
)


# Build user profiles using top 3 preferred genres

# Extract genres from positively rated movies
liked_genres = liked_movies.select(
    "user_id",
    F.explode("tokens").alias("genre")
)

# Count how frequently each user likes each genre
genre_counts = liked_genres.groupBy(
    "user_id", "genre"
).agg(
    F.count("*").alias("genre_count")
)

# Rank genres for each user
genre_window = Window.partitionBy(
    "user_id"
).orderBy(
    F.desc("genre_count"),
    F.asc("genre")
)

top_genres = (
    genre_counts
    .withColumn("rank", F.row_number().over(genre_window))
    .filter(F.col("rank") <= 5)
)

# Collect the top 3 genres
user_genres = top_genres.groupBy("user_id").agg(
    F.collect_list("genre").alias("tokens")
)

# Convert user preferences into numerical vectors
user_features = (
    model.transform(user_genres)
    .select("user_id", "features")
)


# Save vectors
item_features.write.mode("overwrite").parquet(
    "data/processed/item_features"
)

user_features.write.mode("overwrite").parquet(
    "data/processed/user_features_top5"
)

print("\n===== FEATURE ENGINEERING RESULTS =====")
print("Number of genres:", len(model.vocabulary))
print("Movie vectors:", item_features.count())
print("User vectors:", user_features.count())

print("\nSample movie vectors:")
item_features.show(3, truncate=False)

print("\nSample user vectors:")
user_features.show(3, truncate=False)

print("\nFeature engineering completed successfully!")

spark.stop()
