'''
The code below creates a user based recommendation system on a small dataset (100k rows) that can be ran on a local environment. 
It is intended to insure the collaborative based filter algorithm works, BEFORE deploying system to analyze a much larger dataset
using Elastic Map Reduce (EMR) from AWS
'''

from pyspark.sql import SparkSession # Enable use of Spark's Dataframe API
from pyspark.sql import functions as func # Get functions for performing operations on Spark Dataframe
from pyspark.sql.types import StructType, StructField, IntegerType, LongType, StringType # Get the required data types and structures for defining the dataframe

# Initialize spark session and enable the use of all local cpu cores
spark = SparkSession.builder.appName("MovieRecommendationSystem").getOrCreate()

# Create schema for u.item
movieNameSch = StructType([ \
    StructField("movieId", IntegerType(), True), \
    StructField("movieTitle", StringType(), True) \
])

# Create Sceham for u.data
userRatingsch = StructType([ \
        StructField("userId", IntegerType(), True), \
        StructField("movieId", IntegerType(), True), \
        StructField("rating", IntegerType(), True), \
        StructField("timestamp", LongType(), True) \
    ])

# Load movie data as dataset
movieName = spark.read \
    .option("sep","|") \
    .option("charset","ISO-8859-1") \
    .schema(movieNameSch) \
    .csv("ml-100k/u.item")

# Load user data as dataset
userRating = spark.read \
    .option("sep","\t") \
    .option("charset","ISO-8859-1") \
    .schema(userRatingsch) \
    .csv("ml-100k/u.data")

# We omit the timestamp column
ratings = userRating.select("userId","movieId","rating")

# Get user rating counts
user_rating_counts = ratings.groupBy("userId").count().orderBy("userId")

# Get count value of 75 percentile
percentile_75 = user_rating_counts.approxQuantile("count",[0.75],0.001)[0]

# Get user id of users within the 75 percentile
percentile_75_users = user_rating_counts.filter(user_rating_counts["count"] <= percentile_75)

# Get the filtered ratings
filtered_ratings = ratings.join(percentile_75_users,"userId","inner")


