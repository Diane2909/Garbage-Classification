from PIL import Image
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, StructField, StructType, IntegerType, StringType
import numpy as np

# Initialize Spark Session
spark = SparkSession.builder.appName("GarbageClassification").getOrCreate()

schema = StructType([
    StructField("file", StringType(), True)
])
file_size = (64, 64)

def predict_images_to_parquet(data_path, sub_folder):

    directory = f"{data_path}/{sub_folder}"
    files = os.listdir(directory)

    file_data = ((file,) for file in files)

    # Create DataFrame with file paths
    df = spark.createDataFrame(file_data, schema=schema)

    # Define UDF to process images
    def process_image(filename):
        #print(f'Processing: {directory}/{filename}')
        img = Image.open(f'{directory}/{filename}')
        img = img.convert('L')  # Convert to grayscale
        img = img.resize(file_size)  # Resize for vector
        arr = np.array(img).astype(int).tolist()
        return arr
        
    process_image_udf = udf(process_image, ArrayType(ArrayType(IntegerType())))

    # Apply transformation
    df = df.withColumn(f"vector", process_image_udf(col("file")))

    df = df.select("vector", "file")

    # Save as Parquet
    df.write.mode("append").parquet(f"{data_path}/prediction_data.parquet")

    for file in files:
        os.rename(f"{directory}/{file}", f"{data_path}/archive/{file}")

predict_images_to_parquet("../data", "input")