package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{CountVectorizer, IDF, RegexTokenizer, StopWordsRemover, StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}


object Trainer {

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

   val parquetFileDF = spark.read.parquet("/home/vaio/Documents/INF729/SPARK/TP_ParisTech_2017_2018_starter/data/prepared_trainingset")

   parquetFileDF.show();


    /** TF-IDF **/
    val tokenizer​ = new  RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    val df_tokenizer​ = tokenizer​.transform(parquetFileDF)

    df_tokenizer​.select("text").show()
    df_tokenizer​.select("tokens").show()
    //println(s"Number of columns ${df.withColumn("tokens").length}")

    val wordremover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filtered");

    val df_wordremover = wordremover.transform(df_tokenizer​)
    df_wordremover.select("filtered").show()
   // df_filter.show()

    /** VECTOR ASSEMBLER **/
    val countvectorizer = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("vectorized")


    val countvectorizer_model = countvectorizer.fit(df_wordremover)

    val df_countvectorizer = countvectorizer_model.transform(df_wordremover)
    df_countvectorizer.select("vectorized").show(4,false)


    // Affiche le mot correspondant à l'index du vecteur
    println(countvectorizer_model.vocabulary(121))

    val idf= new IDF()
      .setInputCol("vectorized")
      .setOutputCol("tfidf")

    val idf_model = idf.fit(df_countvectorizer)

    val df_idf = idf_model.transform(df_countvectorizer)
    df_idf.select("tfidf").show(5, false) // false pour affficher toute les colonnes
    /** MODEL **/

    val indexer_country = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")

    val indexer_country_model = indexer_country.fit(df_idf)

    val df_country= indexer_country_model.transform(df_idf)

    df_country.select("country_indexed") show(false)



    val indexer_currency = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")

    val indexer_currency_model = indexer_currency.fit(df_country)

    val df_currency= indexer_currency_model.transform(df_country)

    df_currency.select("currency_indexed").show(false)



    /** 7e STage **/

    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf","days_campaign", "hours_prepa", "goal", "country_indexed","currency_indexed"))
      .setOutputCol("features")

    val df_features = assembler.transform(df_currency)

    df_features.select("features").show(5)


    df_features.write.mode(SaveMode.Overwrite).parquet("/home/vaio/Documents/INF729/SPARK/TP_ParisTech_2017_2018_starter/data/pre_pipeline_trainingset")



    /** 8e Stage **/
    val lr= new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7,0.3))
      .setTol(1.0e-6)
      .setMaxIter(100)

    val lr_model= lr.fit(df_features)

    val df_final= lr_model.transform(df_features)
    df_final.select("raw_predictions").show(false)



    /** PIPELINE **/
    val pipeline = new Pipeline()
        .setStages(Array(tokenizer​,wordremover,countvectorizer,idf))

    val pipeline_model = pipeline.fit(parquetFileDF)

    val df_pipeline= pipeline_model.transform(parquetFileDF)
    df_pipeline.show(2)

    /** TRAINING AND GRID-SEARCH **/

  }
}
