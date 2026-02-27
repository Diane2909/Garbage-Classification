import io
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, IntegerType
from PIL import Image


def image_to_pixels(content, size=(32, 32)):
    if content is None:
        return None
    im = Image.open(io.BytesIO(content)).convert("L").resize(size)
    return list(im.getdata())


def load_images_df(spark, base_dir):
    return (
        spark.read.format("binaryFile")
        .option("recursiveFileLookup", "true")
        .load(base_dir)
        .withColumn("category", F.element_at(F.split(F.col("path"), "/"), -2))
        .select("category", "content")
    )


def preprocess_df(df, size=(32, 32)):
    to_pixels_udf = F.udf(lambda b: image_to_pixels(b, size), ArrayType(IntegerType()))
    return (
        df.withColumn("pixels_array", to_pixels_udf(F.col("content")))
        .where(F.col("pixels_array").isNotNull())
        .withColumn("pixels", F.concat_ws(",", F.col("pixels_array")))
        .select("category", "pixels")
    )


def encode_labels(df):
    categories = [r["category"] for r in df.select("category").distinct().orderBy("category").collect()]
    mapping = spark.createDataFrame([(c, i) for i, c in enumerate(categories)], ["category", "label"])
    return df.join(mapping, on="category", how="inner").select("label", "pixels")


if __name__ == "__main__":
    spark = SparkSession.builder.appName("GarbagePreprocess").getOrCreate()

    train_df = encode_labels(preprocess_df(load_images_df(spark, "data/train")))
    test_df = encode_labels(preprocess_df(load_images_df(spark, "data/test")))

    train_df.coalesce(1).write.mode("overwrite").option("header", True).csv("out_csv/train")
    test_df.coalesce(1).write.mode("overwrite").option("header", True).csv("out_csv/test")

    spark.stop()