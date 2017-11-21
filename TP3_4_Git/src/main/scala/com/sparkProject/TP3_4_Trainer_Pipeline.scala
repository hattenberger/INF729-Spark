package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.{DataFrame, SparkSession}


object TP3_4_Trainer_Pipeline {

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



    /******************************************************
      *
      * Ce fichier traite de la question Q5 et de la mise en place du
      * pipeline jusqu'à la sauvegarde du modèle de regression logistic
      * et de la sauvegarde des données associées
      *
      *******************************************************************/

    /** CHARGER LE DATASET **/


    val parquetFileDF: DataFrame = spark
      .read
      .option("header", true)        // Use first line of all files as header
      .option("inferSchema", "true") // Try to infer the data types of each column (ne sert pas à grand chose ici, car il met en string et retraiter au e))
      .option("nullValue", "false")  // replace strings "false" (that indicates missing data) by null values
      .parquet("/home/vaio/Documents/INF729/SPARK/TP_ParisTech_2017_2018_starter/data/prepared_trainingset")


    /** 1) TF-IDF **/
    val tokenizer​ = new  RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    /** 2) Word Remover **/
    val wordremover = new StopWordsRemover()
      .setInputCol(tokenizer​.getOutputCol)
      .setOutputCol("filtered");


    /** 3 ) VECTOR ASSEMBLER **/
    val countvectorizer = new CountVectorizer()
      .setInputCol(wordremover.getOutputCol)
      .setOutputCol("vectorized")

    /** 4) IDF **/
    val idf= new IDF()
      .setInputCol(countvectorizer.getOutputCol)
      .setOutputCol("tfidf")


    /** 5) Indexer _ Country **/
    val indexer_country = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")


    /** 6) Indexer _ currency**/
    val indexer_currency = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")


    /** 7) Stage **/
    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf","days_campaign", "hours_prepa", "goal", "country_indexed","currency_indexed"))
      .setOutputCol("features")


    /** 8) Stage **/
    val lr= new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol(assembler.getOutputCol)
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("rawPrediction")
      .setThresholds(Array(0.7,0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)


    /** PIPELINE **/
    val pipeline = new Pipeline()
        .setStages(Array(tokenizer​,wordremover,countvectorizer,idf,indexer_country,indexer_currency,assembler,lr))


    /** Split Training Set **/
    val Array (training, test) = parquetFileDF.randomSplit(Array(0.9, 0.1), seed = 12345)

    /** Mise en cache pour accélérer les accès mémoires **/
    val training_cache = training.cache()


    /** Test des différents paramétrage de la regression
      * Lors des tests l'utilisation de l'ElasticNetParam a déterioré
      * le score f1, le grid search du paramètre a été supprimé
       ********************************************************/
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array[Double](10e-8, 10e-6, 10e-4, 10e-2))
      .addGrid(countvectorizer.minDF, Array[Double](55.0,75.0,95.0))
      //.addGrid(lr.elasticNetParam, Array[Double](0.0,0.3,0.5,0.8,1.0))
      .build()


    /** Utilisation du Multi_evaluator pour le calcul du score F1
      * en effet le Binomiale evaluator ne l'a pas
      * ************************************* */
    val multi_evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")

    /** Pour des raisons de temps de calcul nous n'utiliserons pas
      * de cross validation mais un seul split avec 70% pour le train  */
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(multi_evaluator)
      .setEstimatorParamMaps(paramGrid)
      // 70% of the data will be used for training and the remaining 30% for validation.
      .setTrainRatio(0.7)


    /**   Entrainement du modèle sur le train dataset puis évaluation sur le test dataset  */
    val cvModel = trainValidationSplit.fit(training_cache)
    val df_with_predictions  = cvModel.transform(test)


    /** Affichage du score et de la matrice de confusion   */
    println("f1 "+ multi_evaluator.setMetricName("f1").evaluate(df_with_predictions))
    df_with_predictions.groupBy("final_status", "predictions").count.show()


    /** Sauvegarde du modèle pour un usage ultérieur    */
    cvModel.write.overwrite().save("TP_Model_Predictions")


    /** J'ai effectué plusieurs tests avec des classifieurs différents tels que :
      *  + Random Forest
      *  + gbt
      *  Mais je n'ai pas obtenu de meilleurs résulats qu'avec la regression Logistic
      *
      * ************************************* */



  }
}
