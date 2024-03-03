'''
The code below creates a user based recommendation system on a small dataset (100k rows) that can be ran on a local environment. 
Note, I will not be including hyper tuning capablity do to it's computational intensity.
'''

from pyspark.sql import SparkSession # Enable use of Spark's Dataframe API
from pyspark.sql.types import StructType, StructField, IntegerType, LongType, StringType # Get the required data types and structures for defining the dataframe
from pyspark.ml.recommendation import ALS # Alternating Least Square algorithm will be used to build the recommendation system
from pyspark.ml.evaluation import RegressionEvaluator

# Initialize spark session and enable the use of all local cpu cores
spark = SparkSession.builder.appName("MovieRecommendationSystem").getOrCreate()

# Create schema for u.item
movieNameSch = StructType([
    StructField("movieId", IntegerType(), True),
    StructField("movieTitle", StringType(), True)
])

# Create Sceham for u.data
userRatingsch = StructType([ \
        StructField("userId", IntegerType(), True),
        StructField("movieId", IntegerType(), True),
        StructField("rating", IntegerType(), True),
        StructField("timestamp", LongType(), True)
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

# Get the filtered ratings. THis will preprocessed dataset to be used for modeling
filtered_ratings = ratings.join(percentile_75_users,"userId","inner")

# Split dataset for training and testing
(training,test) = ratings.randomSplit([0.8,.2])

# Build the recommendation model using ALS on the training data
# Hypertuning ALS can be done by utilizing paramgridbuilder. However do to the lack of computation power
# I will not be including this capability within the scope of this project
als = ALS(rank=25,maxIter=5,regParam=0.01,
          userCol="userId",itemCol="movieId",ratingCol="rating",
          coldStartStrategy="drop",
          implicitPrefs=False)

model = als.fit(training)

# Evalulate the model on the test data by using RMSE
predictions = model.transform(test)
eval = RegressionEvaluator(metricName="rmse",labelCol="rating",predictionCol="prediction")

rmse = eval.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

# Get the top 10 movie recommendation for each user
recommendations = model.recommendForAllUsers(10).show(10)


