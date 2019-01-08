import breeze.linalg.{DenseVector, inv}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.linalg.{Matrix, Vector, Vectors}
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql.{Column, DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions._

object MethodΒ {

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
    val dataScaled = scalarModel.transform(features).select("scaled_features", "feature")


    val dataRDD = dataScaled.rdd.map(r => r(0)).collect() // convert scaled_features to RDD for use in isOutlier udf

    // Decide if a point is outlier by counting the near points that are in a
    // specific radius (R) , and comparing this number with a threshold (neighbors_threshold)
    spark.udf.register("isOutlier", (feature: Vector) => {

      val R = 0.5
      val neighbors_threshold = 5500

      var numOfNeighborsInR = 0

      dataRDD.foreach(feature2 =>{
        // check for every point if it is in radius of current point (feature)
        if (Math.sqrt(Vectors.sqdist(feature, feature2.asInstanceOf[Vector])) < R){
          numOfNeighborsInR += 1
        }

      })

      numOfNeighborsInR < neighbors_threshold // return true if there are not enough neighbors

    })




    // Create output csv with columns: X, Y, cluster_id, is_outlier
    spark.udf.register("getX", (feature: Vector) => {
      feature.apply(0)
    })
    spark.udf.register("getY", (feature: Vector) => {
      feature.apply(1)
    })
    val output = dataScaled.selectExpr("getX(feature) as X", "getY(feature) as Y", "isOutlier(scaled_features) as is_outlier")
    output.write.format("com.databricks.spark.csv").save("resultsB.csv")


    // print how many outliers were found
    val outlier_counter = output.filter(col("is_outlier") === true).count().asInstanceOf[Int]
    println("Number of outliers: " + outlier_counter)




  }

}