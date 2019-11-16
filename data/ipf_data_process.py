from RareDiseaseDetection.utils.spark_utils import spark_session
from pyspark.sql.functions import *
import pandas as pd

spark = spark_session()

path = "./"

df = spark.read.parquet(path)

df = df.select("patient_id", "week_id", "code", "prefix_flag", "cohort")
s = df.select("patient_id", "cohort")
s = s.dropDuplicates()
s = s.sampleBy("cohort", fractions={'positive': 0.5, 'scoring': 0.5}, seed=10)
te = s.sampleBy("cohort", fractions={'positive': 0.001, 'scoring': 0.001}, seed=10) # 10000:4
# tr = s.subtract(s).sampleBy("cohort", fractions={'positive': 0.025, 'scoring': 0.001}, seed=10) # 100:1
# tr = s.subtract(s).sampleBy("cohort", fractions={'positive': 0.25, 'scoring': 0.001}, seed=10) # 10:1
tr = s.subtract(s).sampleBy("cohort", fractions={'positive': 0.001, 'scoring': 0.0000004}, seed=10) # 1:1

df.createOrReplaceTempView("df")
tr.createOrReplaceTempView("trdf")
te.createOrReplaceTempView("tedf")
train_data = spark.sql("select df.patient_id, df.week_id, df.code, df.prefix_flag, df.cohort from df, trdf where df.patient_id = trdf.patient_id")
train_data = train_data.sort(asc('patient_id'), asc('week_id'))


test_data = spark.sql("select df.patient_id, df.week_id, df.code, df.prefix_flag, df.cohort from df, tedf where df.patient_id = tedf.patient_id")
test_data = test_data.sort(asc('patient_id'), asc('week_id'))


train_s = train_data.dropDuplicates()
# train_s = train_data.drop('week_id')
train_s = train_s.withColumn("codes", concat_ws(', ', train_s['code'], train_s['prefix_flag']))
train_s = train_s.drop('code')
train_s = train_s.drop('prefix_flag')
train_s = train_s.groupBy("patient_id", "cohort", "week_id").agg(collect_list('codes').cast('string').alias('visit'))
train_s = train_s.drop('week_id')
train_s = train_s.groupBy("patient_id", "cohort").agg(collect_list('visit').cast('string').alias('visits'))
train_s = train_s.drop('patient_id')
train_s = train_s.toPandas()
train_s.to_csv("RareDiseaseDetection/data/sample_ipf_data.csv")


test_s = test_data.dropDuplicates()
# test_s = test_data.drop('week_id')
test_s = test_s.withColumn("codes", concat_ws(', ', test_s['code'], test_s['prefix_flag']))
test_s = test_s.drop('code')
test_s = test_s.drop('prefix_flag')
test_s = test_s.groupBy("patient_id", "cohort", "week_id").agg(collect_list('codes').cast('string').alias('visit'))
test_s = test_s.drop('week_id')
test_s = test_s.groupBy("patient_id", "cohort").agg(collect_list('visit').cast('string').alias('visits'))
test_s = test_s.drop('patient_id')
test_s = test_s.toPandas()
test_s.to_csv("RareDiseaseDetection/data/ipf_test_data.csv")
