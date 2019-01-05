import breeze.linalg._
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.{StandardScaler}
import org.apache.spark.ml.linalg.{Vectors, Vector, Matrix}
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.stat.Correlation

/*
import breeze.linalg.{DenseVector, inv}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.ml.linalg.{Vectors}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.functions.col
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.mllib.util.MLUtils
*/

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
    val kmeans = new KMeans().setK(5).setFeaturesCol("feature").setPredictionCol("prediction")
    val model = kmeans.fit(features)


    println("")
    println("Calculate cluster centers")
    println("=========================")
    println("")
    val predicted = model.transform(features)

    //predicted.show(300)


    /*

    // ============= METHOD A


    spark.udf.register("calcDistance", (feature: org.apache.spark.ml.linalg.Vector, prediction: Int) => {
      val current_cluster = model.clusterCenters.apply(prediction) // Find in which cluster the current point was predicted
      Math.sqrt(Vectors.sqdist(feature, current_cluster))
    })

    val distances = predicted.selectExpr("feature", "prediction", "calcDistance(feature, prediction) as distance")

    val cluster_averages = distances.groupBy("prediction").avg("distance").sort("prediction").select("avg(distance)").rdd.map(r => r(0)).collect()
    val cluster_max = distances.groupBy("prediction").max("distance").sort("prediction").select("max(distance)").rdd.map(r => r(0)).collect()
    val cluster_min = distances.groupBy("prediction").min("distance").sort("prediction").select("min(distance)").rdd.map(r => r(0)).collect()



    spark.udf.register("calcClusterThreshold", (prediction: Int) => {
      val current_cluster_average = cluster_averages.apply(prediction).asInstanceOf[Double]
      val current_cluster_max = cluster_max.apply(prediction).asInstanceOf[Double]
      val current_cluster_min = cluster_min.apply(prediction).asInstanceOf[Double]

      current_cluster_average + (current_cluster_max + current_cluster_min)/2
    })


    val outliers = distances.selectExpr("feature", "prediction", "distance", "calcClusterThreshold(prediction) as cluster_threshold").filter(col("distance") > col("cluster_threshold"))
    outliers.show()
    println(outliers.count())

    /*

    val min_distance = 3000000.0


    spark.udf.register("calcDistance", (feature: org.apache.spark.ml.linalg.Vector, prediction: Int) => {
      val current_cluster = model.clusterCenters.apply(prediction) // Find in which cluster the current point was predicted
      Vectors.sqdist(feature, current_cluster)
    })

    val distances = predicted.selectExpr("calcDistance(feature, prediction) as distance")
    println(distances.filter(col("distance") > min_distance).count())
    */

    */

    // =============== METHOD B


    var clustersArray = new Array[DataFrame](5)

    clustersArray(0) = predicted.filter(col("prediction") === 0)
    clustersArray(1) = predicted.filter(col("prediction") === 1)
    clustersArray(2) = predicted.filter(col("prediction") === 2)
    clustersArray(3) = predicted.filter(col("prediction") === 3)
    clustersArray(4) = predicted.filter(col("prediction") === 4)


    def calcMahalanobis(df: DataFrame, inputCol: String): DataFrame = {
      val Row(coeff1: Matrix) = Correlation.corr(df, inputCol).head

      val invCovariance = inv(new breeze.linalg.DenseMatrix(2, 2, coeff1.toArray))

      val mahalanobis = udf[Double, Vector] { v =>
        val vB = DenseVector(v.toArray)
        vB.t * invCovariance * vB
      }

      df.withColumn("mahalanobis", mahalanobis(df(inputCol)))
    }

    def stddev(xs: scala.collection.immutable.List[Double], avg: Double): Double = xs match {
      case Nil => 0.0
      case ys => math.sqrt((0.0 /: ys) {
        (a,e) => a + math.pow(e - avg, 2.0)
      } / xs.size)
    }

    var i=0

    for (cluster <- clustersArray){



      // Standardize the df2: This is important step in calculating mahalanobis distance.
      // When normalized to zero mean and unit standard deviation then correlation matrix is equal to covariance matrix

      val standardScalar = new StandardScaler().setInputCol("feature").setOutputCol("scaledFeat").setWithMean(true).setWithStd(true)
      val scalarModel = standardScalar.fit(cluster.select("feature"))
      val clusterScaled = scalarModel.transform(cluster).select("feature", "scaledFeat")


      val clusterWithMahalanobis = calcMahalanobis(clusterScaled, "scaledFeat")

      //clusterWithMahalanobis.sort(col("mahalanobis").desc).show(10000)

      val mahalanobis_avg = clusterWithMahalanobis.select(avg("mahalanobis")).first().get(0).asInstanceOf[Double]
      val mahalanobis_stddev = clusterWithMahalanobis.agg(stddev_pop(col("mahalanobis"))).first().get(0).asInstanceOf[Double]


      val isOutlier = udf[Boolean, Double] { mahalanobis =>
        mahalanobis > mahalanobis_avg + 2*mahalanobis_stddev
      }

      val clusterWithIsOutlier = clusterWithMahalanobis.withColumn("is_outlier", isOutlier(clusterWithMahalanobis("mahalanobis")))


      println("Average: " + mahalanobis_avg)
      println("Std dev: " + mahalanobis_stddev)


      spark.udf.register("getX", (vector: Vector) => {
        vector.apply(0)
      })
      spark.udf.register("getY", (vector: Vector) => {
        vector.apply(1)
      })

      val output = clusterWithIsOutlier.selectExpr("getX(feature) as X", "getY(feature) as Y", "is_outlier")
      output.write.format("com.databricks.spark.csv").save("cluster"+i+".csv")

      i+=1

    }



  }

}
