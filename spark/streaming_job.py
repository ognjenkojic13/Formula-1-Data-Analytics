from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, LongType, StructField, StructType, TimestampType

CHUNKS_DIR = Path(__file__).resolve().parent.parent / "data" / "streaming_chunks" / "f1-car-data"

SCHEMA = StructType([
    StructField("date", TimestampType(), True),
    StructField("driver_number", LongType(), True),
    StructField("speed", DoubleType(), True),
    StructField("throttle", DoubleType(), True),
    StructField("brake", DoubleType(), True),
    StructField("rpm", DoubleType(), True),
    StructField("n_gear", DoubleType(), True),
    StructField("gap_to_leader", DoubleType(), True),
])


def build_spark() -> SparkSession:
    return (
        SparkSession.builder.appName("F1SparkStreamingDemo")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


def run(max_files_per_trigger: int = 5, await_termination_seconds: int = 30) -> None:
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    stream_df = (
        spark.readStream.schema(SCHEMA)
        .option("maxFilesPerTrigger", max_files_per_trigger)
        .parquet(str(CHUNKS_DIR))
    )

    windowed = (
        stream_df
        .withWatermark("date", "20 seconds")
        .groupBy(F.window("date", "10 seconds"), "driver_number")
        .agg(
            F.avg("speed").alias("avg_speed"),
            F.avg("throttle").alias("avg_throttle"),
            F.max("speed").alias("max_speed"),
            F.count("*").alias("n_readings"),
        )
        .orderBy("window")
    )

    query = (
        windowed.writeStream
        .format("memory")
        .queryName("speed_windows")
        .outputMode("complete")
        .start()
    )

    query.awaitTermination(await_termination_seconds)
    query.stop()

    result = spark.sql("SELECT * FROM speed_windows ORDER BY window, driver_number")
    result.show(20, truncate=False)
    print(f"Total windowed rows accumulated: {result.count()}")

    spark.stop()


if __name__ == "__main__":
    run()
