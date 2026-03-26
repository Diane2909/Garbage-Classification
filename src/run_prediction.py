from PIL import Image
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, udf, array_max, array_position, array, element_at
from pyspark.sql.types import ArrayType, StructField, StructType, IntegerType, StringType, DoubleType
import numpy as np
from keras.models import load_model

# Initialize Spark Session
spark = SparkSession.builder.appName("GarbageClassification").getOrCreate()

schema = StructType([
    StructField("file", StringType(), True)
])
file_size = (64, 64)

model = load_model("../models/final_CNN.keras")
classes = ["biodegradable", "cardboard", "glass", "metal", "paper", "plastic"]

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
        arr = np.array(img).astype(int)
        return arr.tolist()
        
    process_image_udf = udf(process_image, ArrayType(ArrayType(IntegerType())))

    def predict(vector):
        X = np.array(vector)
        X = X.reshape((1,) + X.shape + (1,))
        prediction = model.predict(X)
        return prediction.flatten().tolist()
    
    predict_udf = udf(predict, ArrayType(DoubleType()))

    # Apply transformation
    df = df.withColumn("vector", process_image_udf(col("file")))
    df = df.withColumn("prediction", predict_udf(col("vector")))
    df = df.withColumn("confidence", array_max(col("prediction")))
    df = df.withColumn("class_id", array_position(col("prediction"), col("confidence")).astype("INT"))
    df = df.withColumn("class", element_at(array(*[lit(c) for c in classes]), col("class_id")))

    df = df.select("file", "prediction", "class_id", "class", "confidence")
    df.show()

    # Save as Parquet
    df.write.mode("append").parquet(f"{data_path}/prediction_data.parquet")

    for file in files:
        os.rename(f"{directory}/{file}", f"{data_path}/archive/{file}")

predict_images_to_parquet("../data", "input")