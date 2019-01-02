import org.apache.spark.sql.SparkSession


object App {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder.master("local").appName("OutliersDetection").getOrCreate()
    val df = spark.read.format("csv").option("header", "false").load("src/main/resources/data.csv")

  }

}
