import breeze.linalg.{DenseVector, inv}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.linalg.{Matrix, Vector, Vectors}
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions._

object MethodB {

  def main(args: Array[String]): Unit = {


    val spark = SparkSession.builder.master("local").appName("OutliersDetection").getOrCreate()

    // read input csv
    val data = spark.read.format("csv").option("inferSchema", "true").load("src/main/resources/data.csv")


    // count average of both columns (x, y) for use on deficient inputs
    val c0_avg = data.select(avg("_c0")).first().get(0).asInstanceOf[Double]
    val c1_avg = data.select(avg("_c1")).first().get(0).asInstanceOf[Double]

    // convert the first two columns of input to a vector
    // replace deficient columns with the average of the column
    spark.udf.register("toVector", (a: Object, b: Object) => {
      if (a == null || b == null) {
        Vectors.dense(c0_avg, c1_avg)
      } else {
        Vectors.dense(a.asInstanceOf[Double], b.asInstanceOf[Double])
      }
    })
    val features = data.selectExpr("toVector(_c0, _c1) as feature")


    // Standardize features with average = 0 and standard deviation = 1
    val standardScalar = new StandardScaler().setInputCol("feature").setOutputCol("scaled_features").setWithMean(true).setWithStd(true)
    val scalarModel = standardScalar.fit(features.select("feature"))
    val dataScaled = scalarModel.transform(features).select("feature", "scaled_features")


    // Perform kmeans on input data with k=5
    val numOfClusters = 5
    val kmeans = new KMeans().setK(numOfClusters).setFeaturesCol("scaled_features").setPredictionCol("cluster_id")
    val model = kmeans.fit(dataScaled)
    val predicted = model.transform(dataScaled)


    // Function that calculates the mahalanobis distance between points of a cluster
    def calcMahalanobis(cluster: DataFrame, inputCol: String): DataFrame = {
      val Row(coeff1: Matrix) = Correlation.corr(cluster, inputCol).head
      val invCovariance = inv(new breeze.linalg.DenseMatrix(2, 2, coeff1.toArray))
      val mahalanobis = udf[Double, Vector] { v =>
        val vB = DenseVector(v.toArray)
        vB.t * invCovariance * vB
      }

      // add mahalanobis column
      cluster.withColumn("mahalanobis", mahalanobis(cluster(inputCol)))
    }


    // getX and getY functions for creating the output CSV
    spark.udf.register("getX", (feature: Vector) => {
      feature.apply(0)
    })
    spark.udf.register("getY", (feature: Vector) => {
      feature.apply(1)
    })


    // loop through every cluster, to calculate its mahalanobis distances and create the output csv
    var i = 0
    var outlier_counter = 0
    for (cluster_id <- 0 until numOfClusters) { // cluster_id = 0,1,2,3,4

      // find the rows of points that belong to the current cluster and add the mahalanobis column
      val cluster = predicted.filter(col("cluster_id") === cluster_id)
      val clusterWithMahalanobis = calcMahalanobis(cluster, "scaled_features")


      // Calculate the average and the standard deviation of the current cluster
      val mahalanobis_avg = clusterWithMahalanobis.select(avg("mahalanobis")).first().get(0).asInstanceOf[Double]
      val mahalanobis_stddev = clusterWithMahalanobis.agg(stddev_pop(col("mahalanobis"))).first().get(0).asInstanceOf[Double]


      // Decide whether a point is outlier depending on if the mahalanobis
      // distance of the point is bigger then the threshold (avg + 2*s)
      val isOutlier = udf[Boolean, Double] { mahalanobis =>
        mahalanobis > mahalanobis_avg + 2 * mahalanobis_stddev // return true if distance is bigger than threshold, false otherwise
      }

      // Add the is_outlier column
      val clusterWithIsOutlier = clusterWithMahalanobis.withColumn("is_outlier", isOutlier(clusterWithMahalanobis("mahalanobis")))


      // Create output csv with columns: X, Y, cluster_id, is_outlier
      val output = clusterWithIsOutlier.selectExpr("getX(feature) as X", "getY(feature) as Y", "cluster_id", "is_outlier")
      output.write.format("com.databricks.spark.csv").save("cluster" + i + ".csv")


      outlier_counter += output.filter(col("is_outlier") === true).count().asInstanceOf[Int]
      i += 1

    }

    println("Number of outliers: " + outlier_counter)


  }

}