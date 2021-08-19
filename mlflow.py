import mlflow
import mlflow.spark
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

mlflow.set_experiment("path")
with mlflow.start_run(run_name='LR-Single-Feature') as run:
    assembler = VectorAssembler(inputCols=["bedrooms"], outputCol="Features")
    lr = LinearRegression(featuresCol="Features", labelCol="Price")
    pipeline = Pipeline(stages=[assembler, lr])
    pipelineModel = pipeline.fit(trainDF)

    # log parameters
    mlflow.log_param("label", "price_bedroom")
    # log Model
    mlflow.spark.log_model(pipelineModel, 'model')

    # Evaluate Predictions
    predDF = pipelineModel.transform(testDf)
    regressionEvaluator = RegressionEvaluator(
        predictionCol="Prediction", labelCol="Price", metricName="rmse")
    rmse = regressionEvaluator.evaluate(predDF)

    # log metrics
    mlflow.log_metric("rmse", rmse)
