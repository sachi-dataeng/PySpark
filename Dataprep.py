# Databricks notebook source
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("SparkPractice").config("spark.sql.session.timezone","UTC").getOrCreate()

# COMMAND ----------

df = spark.read.csv("/Volumes/workspace/default/csv_files/dataset.csv", header=True, inferSchema=True, multiLine=True, escape='"')

df.show()

# COMMAND ----------

import re

newcols = []
df.printSchema()
for c in df.columns:
    nc = re.sub(r"[\s\-]+", "_", c.strip().lower())
    nc = re.sub(r"[^a-zA-Z0-9_]", "", nc)
    nc = re.sub(r"_+", "_", nc). strip("_")
    newcols.append(nc)

for old, new in zip(df.columns, newcols):
    if old != new:
        df = df.withColumnRenamed(old, new)
df.printSchema()
    

# COMMAND ----------

from pyspark.sql import functions as F

na_tokens = ["", "na", "n/a", "none" , "null", "-", "--", "unknown"]
df.show(10, truncate=False)

for c, t in df.dtypes:
    if t == "string":
        df = df.withColumn(c, F.regexp_replace(F.col(c), "\xa0", " "))
        df = df.withColumn(c, F.trim(F.col(c)))
        df = df.withColumn(c, F.regexp_replace(F.col(c), r"\s+", " "))
        df = df.withColumn(c, F.when(F.lower(F.col(c)).isin(na_tokens), None).otherwise(F.col(c)))

df.show(10, truncate=False)

# COMMAND ----------

df.show(10, truncate=False)
df = df.withColumn("amount", F.regexp_replace("amount", "O", "0"))

df = df.withColumn("amount", F.regexp_replace("amount", r"[^0-9\.]", ""))
df = df.withColumn("amount", F.regexp_replace("amount", r"\.", ""))
df = df.withColumn("amount", F.regexp_replace("amount", r",", "."))
df = df.withColumn("amount", F.col("amount").cast("double"))

df.show(10, truncate=False)

# COMMAND ----------

df = df.withColumn("is_active",
                   F.when(F.lower("is_active").isin(["true", "yes", "t", "y", "1", "active"]),F.lit(True))
                   .when(F.lower("is_active").isin(["false", "no", "f", "n", "0", "inactive"]),F.lit(False))
                   .otherwise(F.lit(None).cast("boolean"))
                   )
df.show(10, truncate=False)

# COMMAND ----------

df = df.withColumn("updated_at", F.trim(F.col("updated_at")))
df = df.withColumn("updated_at", F.regexp_replace(F.col("updated_at"), r"[-,]","/"))
df = df.withColumn("updated_at", F.when(F.col("updated_at").isNotNull() & (~F.col("updated_at").rlike(r"\d{1,2}/\d{1,2}/\d{4}\s\d{1,2}:\d{2}")),
    F.concat_ws(" ", F.col("updated_at"), F.lit("00:00")))
    .otherwise(F.col("updated_at"))
    )

df = df.withColumn("updated_at_ts", F.expr("try_to_timestamp(updated_at, 'M/d/yyyy H:mm')"))

df = df.withColumn("updated_at",F.when(F.col("updated_at_ts").isNotNull(), F.date_format(F.col("updated_at_ts"), "yyyy-MM-dd HH:mm")).otherwise(None)).drop("updated_at_ts")

df.show(10, truncate=False)


# COMMAND ----------

df = df.withColumn("order_date", F.trim(F.col("order_date")))
df = df.withColumn("order_date", F.regexp_replace(F.col("order_date"), r"[-,]","/"))
df = df.withColumn("order_date", F.when(F.col("order_date").isNotNull() & (~F.col("order_date").rlike(r"\d{1,2}/\d{1,2}/\d{4}\s\d{1,2}:\d{2}")),
    F.concat_ws(" ", F.col("order_date"), F.lit("00:00")))
    .otherwise(F.col("order_date"))
    )

df = df.withColumn("order_date_ts", F.expr("try_to_timestamp(order_date, 'M/d/yyyy H:mm')"))

df = df.withColumn("order_date",F.when(F.col("order_date_ts").isNotNull(), F.date_format(F.col("order_date_ts"), "yyyy-MM-dd HH:mm")).otherwise(None)).drop("order_date_ts")

df.show(10, truncate=False)


# COMMAND ----------

from pyspark.sql.window import Window as W

pk = ["id"]

if "updated_at" in df.columns:
    w = W.partitionBy(*pk).orderBy(F.col("updated_at").desc())
    df = df.withColumn("_rn", F.row_number().over(w))\
        .where(F.col("_rn") == 1)\
        .drop("_rn")

else:
    df = df.dropDuplicates(pk)
df.show(10, truncate=False)

# COMMAND ----------

m ={"m": "Male", "f": "Female", "male": "Male", "female": "Female", "m/f": "Male/Female", "other" : "Other", "o" : "Other"}

map_expr = F.create_map([F.lit(x) for x in sum(m.items(),())])
df = df.withColumn("sex", F.coalesce(F.element_at(map_expr, F.lower("sex")), F.col("sex")))
df.show(10, truncate=False)

# COMMAND ----------

q1, q3 = df.approxQuantile("amount", [0.25, 0.75], 0.01)

iqr = q3-q1
lower = q1 - 1.5*iqr
upper = q3 + 1.5*iqr

df = df.withColumn("amount_capped", F.when(df.amount < lower, lower).when(df.amount > upper, upper).otherwise(df.amount))

df.show(20, truncate=False)

# COMMAND ----------

assert df.filter(F.col("id").isNull()).count() == 0

# COMMAND ----------

df.write.format("delta") \
    .mode("overwrite") \
    .option("mergeSchema", "true") \
    .saveAsTable("default.processed_data")

# COMMAND ----------

# MAGIC %sql
# MAGIC Select * from default.processed_data