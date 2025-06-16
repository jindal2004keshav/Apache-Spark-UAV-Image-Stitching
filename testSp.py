from pyspark.sql import SparkSession
import time

# Replace with your actual Spark master URL
spark = SparkSession.builder \
    .appName("ClusterTest") \
    .master("spark://10.0.42.43:7077") \
    .config("spark.ui.port", "4050") \
    .getOrCreate()

print("[INFO] Spark session started. Application ID:", spark.sparkContext.applicationId)

# Create a simple DataFrame
data = [(i, i * 2) for i in range(1000000)]
df = spark.createDataFrame(data, ["number", "double"])

# Perform a transformation
df_filtered = df.filter(df["double"] % 10 == 0)

# Trigger an action (forces execution)
count = df_filtered.count()
print("[INFO] Count of rows where double % 10 == 0:", count)

# Sleep so you can view the job in the Spark UI
print("[INFO] Sleeping for 60 seconds so you can check Spark UI at http://10.0.42.43:8081")
# time.sleep(6000)
input("Press enter to exit")

spark.stop()
print("[INFO] Spark session stopped.")
