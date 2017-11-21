package com.sparkProject.Tests

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.IntegerType


object Preprocessor_test {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",   //traite les données en 12 partitions
      "spark.sql.shuffle.partitions" -> "12"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()


    import spark.implicits._

    /*******************************************************************************
      *
      *       TP 2-3
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    /** 0 - PRE CLEAN DES  DONNEES **/
    // val df1= spark.read.text("/home/vaio/Documents/MDI729/TP_ParisTech_2017_2018_starter/train.csv")

    // val df2 = df1.withColumn("replaced",functions.regexp_replace($"value","\"{2,}"," "))

    //df2.select("replaced").write.text("/home/vaio/Documents/MDI729/TP_ParisTech_2017_2018_starter/cleaned_train.csv")
    // df2.show()


    /** 1 - CHARGEME DES DONNEES **/
/**
    val df = spark.read
              .format("csv")
              .option("header", "true") //reading the headers
              .option("mode", "DROPMALFORMED")
              .load("/home/vaio/Documents/MDI729/TP_ParisTech_2017_2018_starter/train.csv")
**/
    //df.take(5).foreach(println)
    //. ​ /build_and_submit.sh​ ​ Preprocessor
   // df.describe()



    val df3 = spark.read
      .format("csv")
      .option("header", "true") //reading the headers
      .option("nullValue","false")
      .option("mode", "DROPMALFORMED")
      .load("/home/vaio/Documents/MDI729/TP_ParisTech_2017_2018_starter/train.csv")


    //a) Afficher le nombre de lignes et le nombre de colonnes dans le dataFrame.
    println(df3.count)

    //b) Afficher le nb colonnes dans le dataFrame.
    println(df3.columns.length )

    //c) Afficher le dataFrame sous forme de table.
    df3.show()

    //d) Afficher le schéma du dataFrame (nom des colonnes et le type des données contenues dans chacune d’elles).
    df3.printSchema()

    //e) Assigner le type “Int” aux colonnes qui vous semblent contenir des entiers.
    df3.columns.foreach(println) //afficher le noms des columns

    // test sur une colonne
    val df4 = df3.withColumn("currency", $"currency".cast((IntegerType)))
    df4.printSchema()
    // les map ne fonctionne que sur les RDD
    //df4 = df3.rdd map(t => (t._1, t._2.asInstanceOf[Double], t._3.asInstanceOf[], ...))



    /** 2 - CLEANING **/



    //a ) Afficher une description statistique des colonnes de type Int (avec .describe().show )
  }

}
