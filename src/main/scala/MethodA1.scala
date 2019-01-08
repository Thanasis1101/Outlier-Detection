import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}

object MethodA1 {

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


    // Calculate euclidean distance between every point and the center of the cluster where the point belongs
    spark.udf.register("calcDistance", (feature: Vector, cluster_id: Int) => {
      val current_cluster_center = model.clusterCenters.apply(cluster_id) // Find the center of the cluster the current point was predicted
      Math.sqrt(Vectors.sqdist(feature, current_cluster_center))
    })
    val distances = predicted.selectExpr("feature", "cluster_id", "scaled_features", "calcDistance(scaled_features, cluster_id) as distance")

    // Calculate the average and the standard deviation for every cluster's distances
    val cluster_averages = distances.groupBy("cluster_id").avg("distance").sort("cluster_id").select("avg(distance)").rdd.map(r => r(0)).collect()
    val cluster_stddev = distances.groupBy("cluster_id").agg(stddev_pop(col("distance"))).sort("cluster_id").select("stddev_pop(distance)").rdd.map(r => r(0)).collect()


    /*
    println(model.clusterCenters.apply(0))
    println(cluster_averages.apply(0).asInstanceOf[Double] + 2 * cluster_stddev.apply(0).asInstanceOf[Double])
    println(model.clusterCenters.apply(1))
    println(cluster_averages.apply(1).asInstanceOf[Double] + 2 * cluster_stddev.apply(1).asInstanceOf[Double])
    println(model.clusterCenters.apply(2))
    println(cluster_averages.apply(2).asInstanceOf[Double] + 2 * cluster_stddev.apply(2).asInstanceOf[Double])
    println(model.clusterCenters.apply(3))
    println(cluster_averages.apply(3).asInstanceOf[Double] + 2 * cluster_stddev.apply(3).asInstanceOf[Double])
    println(model.clusterCenters.apply(4))
    println(cluster_averages.apply(4).asInstanceOf[Double] + 2 * cluster_stddev.apply(4).asInstanceOf[Double])
    */


    // Decide whether a point is outlier depending on if the euclidean distance
    // of the point from its cluster center is bigger then the threshold (avg + 2*s)
    spark.udf.register("isOutlier", (distance: Double, cluster_id: Int) => {
      val current_cluster_average = cluster_averages.apply(cluster_id).asInstanceOf[Double]
      val current_cluster_stddev = cluster_stddev.apply(cluster_id).asInstanceOf[Double]
      val threshold = current_cluster_average + 2 * current_cluster_stddev


      distance > threshold // return true if distance is bigger than threshold, false otherwise
    })


    // Create output csv with columns: X, Y, cluster_id, is_outlier
    spark.udf.register("getX", (feature: Vector) => {
      feature.apply(0)
    })
    spark.udf.register("getY", (feature: Vector) => {
      feature.apply(1)
    })
    val output = distances.selectExpr("getX(feature) as X", "getY(feature) as Y", "cluster_id", "isOutlier(distance, cluster_id) as is_outlier")
    output.write.format("com.databricks.spark.csv").save("results.csv")


    // print how many outliers per cluster were found
    //output.filter(col("is_outlier") === true).groupBy("cluster_id").count().show()

    // print how many outliers were found
    val outlier_counter = output.filter(col("is_outlier") === true).count().asInstanceOf[Int]
    println("Number of outliers: " + outlier_counter)


  }

}