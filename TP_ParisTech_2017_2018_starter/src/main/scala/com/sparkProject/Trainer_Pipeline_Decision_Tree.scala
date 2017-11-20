package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression,DecisionTreeClassifier,RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.{DataFrame, SparkSession}


object Trainer_Pipeline_Decision_Tree {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()




    /*******************************************************************************
      *
      *       TP 4-5
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

   /** CHARGER LE DATASET **/

   //val parquetFileDF: DataFrame = spark.read.parquet("/home/vaio/Documents/MDI729/TP_ParisTech_2017_2018_starter/data/prepared_trainingset")


    val parquetFileDF: DataFrame = spark
      .read
      .option("header", true)        // Use first line of all files as header
      .option("inferSchema", "true") // Try to infer the data types of each column (ne sert pas à grand chose ici, car il met en string et retraiter au e))
      .option("nullValue", "false")  // replace strings "false" (that indicates missing data) by null values
      .parquet("/home/vaio/Documents/INF729/SPARK/TP_ParisTech_2017_2018_starter/data/pre_pipeline_trainingset")

    /** 8) Stage

    val lr= new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("rawPrediction")
      .setThresholds(Array(0.7,0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)
    **/
/**val dt = new DecisionTreeClassifier()
      .setLabelCol("final_status")
      .setFeaturesCol("features")
      .setPredictionCol("predictions")
      .setRawPredictionCol("rawPrediction")

  **/

    val rf = new RandomForestClassifier()
      .setLabelCol("final_status")
      .setFeaturesCol("features")
      .setPredictionCol("predictions")
      .setRawPredictionCol("rawPrediction")
      .setNumTrees(10)

    /** PIPELINE **/
    val pipeline = new Pipeline()
        .setStages(Array(rf))


    /** Split Training Set **/
    val Array (training, test) = parquetFileDF.randomSplit(Array(0.9, 0.1), seed = 12345)

    //val training_cache = training.cache()


    // Train model.  This also runs the indexers.
    val model = pipeline.fit(training)

    // Make predictions.
    val predictions = model.transform(test)

    // Select (prediction, true label) and compute test error
    val multi_evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")


    val accuracy = multi_evaluator.evaluate(predictions)


    println("f1 "+ multi_evaluator.setMetricName("f1").evaluate(predictions))

    predictions.groupBy("final_status", "predictions").count.show()

    //cvModel.write.overwrite().save("TP_SPARK_4&5_model")


    /** TRAINING AND GRID-SEARCH **/

  }
}
