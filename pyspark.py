from pyspark.ml import feature
from pyspark.sql.functions import Column, regexp_replace
from pyspark.ml.feature import Binarizer, VectorAssembler
from typing import cast
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import Imputer
from pyspark.ml import Pipeline
from pyspark.ml.pipeline import PipelineModel

spark = SparkSession.builder.appName("Practice").getOrCreate()
df = spark.read.parquet("path")
indexer = StringIndexer(inputCol='room type', outputCol='room type index')
encoder = OneHotEncoder(inputCol='room type index',
                        outputCol='encoded room type')
imputeCols = ['colA', 'colB', 'colC']
imputer = Imputer(strategy='mean', inputCol=imputeCols, outputCol=imputeCols)

pipeline = Pipeline(stages=[indexer, encoder, imputer])
pipelineModel = pipeline.fit(df)
transformedDf = pipelineModel.transform(df)
display(transformedDf)

# Step 01 : Binning the Columns

binarizer = Binarizer(
    threshold=97, inputCol="review score", outputCol="high rating")
transformedBinnedDf = binarizer.transform(transformedDf)

# Step 02 : Regular Expressions on Strings

transformedBinnedDfRegexDf = (transformedBinnedDf
.withColumnRenamed("price", "price_raw")
.withColumn("price", regexp_replace(Column("price_raw"), "[1$]", "").cast("Decimal(10,2)"))

# Step 03 : Filter Extremes

filteredDf=transformedBinnedDfRegexDf.filter(
    col("price") > 0).filter(col("min_nights") <= 365)

- -----------------------------------------------------------------------------------------------
# Linear Regression Modelling
from pyspark.ml.feature import VectorAssembler
featureCols='colA'
assembler=VectorAssembler(inputCols=featureCols, outputCol="Features")
featurizedDf=assembler.transform(df)

from pyspark.ml.regression import LinearRegression
lr=LinearRegression(featureCol="Features", labelCol="price")
lrModel=lr.fit(featurizedDf)

summary=lrModel.summary
summary.pValues
