import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.functions._
import org.apache.spark.sql.functions.col

import scala.collection.mutable.ArrayBuffer

object App {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder.master("local").appName("OutliersDetection").getOrCreate()

    println("")
    println("Define the input data")
    println("=====================")
    println("")
    val data = spark.read.format("csv").option("inferSchema", "true").load("src/main/resources/data.csv")


    // count average of both columns
    val c0_avg = data.select(avg("_c0")).first().get(0).asInstanceOf[Double]
    val c1_avg = data.select(avg("_c1")).first().get(0).asInstanceOf[Double]


    println("")
    println("Create feature vectors and generate a K-Means model")
    println("===================================================")
    println("")

    spark.udf.register("toVector", (a: Object, b: Object) => {
      if (a==null || b == null) {
        Vectors.dense(c0_avg, c1_avg)
      } else {
        Vectors.dense(a.asInstanceOf[Double], b.asInstanceOf[Double])
      }
    })

    val features = data.selectExpr("toVector(_c0, _c1) as feature")
    val kmeans = new KMeans().setK(3).setFeaturesCol("feature").setPredictionCol("prediction")
    val model = kmeans.fit(features)


    println("")
    println("Calculate cluster centers")
    println("=========================")
    println("")
    val predicted = model.transform(features)

    //predicted.show(300)

    val min_distance = 3000000.0


    spark.udf.register("calcDistance", (feature: org.apache.spark.ml.linalg.Vector, prediction: Int) => {
      val current_cluster = model.clusterCenters.apply(prediction) // Find in which cluster the current point was predicted
      Vectors.sqdist(feature, current_cluster)
    })

    val distances = predicted.selectExpr("calcDistance(feature, prediction) as distance")
    println(distances.filter(col("distance") > min_distance).count())


  }

}
