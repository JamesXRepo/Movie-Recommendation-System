'''
The code below creates a user based recommendation system on a small dataset (100k rows) that can be ran on a local environment. 
It is intended to insure the collaborative based filter algorithm works, BEFORE deploying system to analyze a much larger dataset
using Elastic Map Reduce (EMR) from AWS
'''

from pyspark.sql import SparkSession # Enable use of Spark's Dataframe API
from pyspark.sql import functions as func # Get functions for performing operations on Spark Dataframe
from pyspark.sql.types import Structype, StructField, IntegerType, LongType, StringType