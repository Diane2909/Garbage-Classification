from PIL import Image
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, concat, lit
from pyspark.sql.types import ArrayType, StructField, StructType, IntegerType, StringType
import numpy as np

# Initialize Spark Session
spark = SparkSession.builder.appName("GarbageClassification").getOrCreate()

schema = StructType([
    StructField("class", StringType(), True),
    StructField("file", StringType(), True)
])
file_size = (64, 64)

def transform_images_to_parquet(data_path, train_test):

    directory = f"{data_path}/{train_test}"
    folders = os.listdir(directory)

    file_data = ((folder, file) for folder in folders for file in os.listdir(directory + '/' + folder))

    # Create DataFrame with file paths
    df = spark.createDataFrame(file_data, schema=schema)


    # Define UDF to process images
    def process_image(filename):
        try:
            #print(f'Processing: {directory}/{filename}')
            img = Image.open(f'{directory}/{filename}')
            img = img.convert('L')  # Convert to grayscale
            img = img.resize(file_size)  # Resize for consistency
            arr = np.array(img).astype(int).tolist()
            return arr
        except:
            return None
        
    process_image_udf = udf(process_image, ArrayType(ArrayType(IntegerType())))

    # Define UDF to assign one-hot encoding based on class name
    def get_one_hot(class_name):
        classes = ["biodegradable", "cardboard", "glass", "metal", "paper", "plastic"]
        one_hot = [0] * len(classes)
        index = classes.index(class_name)
        one_hot[index] = 1
        return one_hot

    get_one_hot_udf = udf(get_one_hot, ArrayType(IntegerType()))

    # Apply transformation
    df = df.withColumn(f"x_{train_test}", process_image_udf(concat(col("class"), lit("/"), col("file"))))
    df = df.withColumn(f"y_{train_test}", get_one_hot_udf(col("class")))

    df = df.select(f"x_{train_test}", f"y_{train_test}").filter(col(f"x_{train_test}").isNotNull())

    # Save as Parquet (optimized format for Spark)
    df.write.mode("overwrite").parquet(f"{data_path}/{train_test}_data.parquet")

for data_type in ['train', 'test']:
    transform_images_to_parquet('data', data_type)