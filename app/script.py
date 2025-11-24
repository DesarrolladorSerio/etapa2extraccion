from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, month
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder.appName("EntrenamientoCluster").getOrCreate()

print("\n=== 🚀 ENTRENANDO MODELO EN CLUSTER ===\n")

# 1. Leer dataset real sin header
df = spark.read.csv("pp-complete.csv", header=False, inferSchema=True)

# 2. Asignar nombres correctos

df = df.select(
    col("_c0").alias("ID"),
    col("_c1").alias("Price"),
    col("_c2").alias("Date"),
    col("_c3").alias("Postcode"),
    col("_c4").alias("PropertyType"),
    col("_c5").alias("NewBuild"),
    col("_c6").alias("Duration"),
    col("_c7").alias("PAON"),
    col("_c8").alias("SAON"),
    col("_c9").alias("Street"),
    col("_c10").alias("TownCity"),
    col("_c11").alias("District"),
    col("_c12").alias("County"),
    col("_c13").alias("PPDCategory"),
    col("_c14").alias("RecordStatus")
)

# Columnas útiles para el modelo
df = df.select("Price", "Date", "PropertyType", "NewBuild", "Duration", "County")

# Limpiar y convertir tipos
df = df.withColumn("Price", col("Price").cast("int"))
df = df.withColumn("Date", col("Date").cast("date"))

df = df.dropna(subset=["Price", "PropertyType", "County", "Date"])

# Crear Year y Month
df = df.withColumn("Year", year(col("Date")))
df = df.withColumn("Month", month(col("Date")))

print("Dataset final:", df.count(), "filas")

# --------------------------
# 3. Indexers (y guardarlos)
# --------------------------
indexer_PT = StringIndexer(inputCol="PropertyType", outputCol="PropertyTypeIndex").fit(df)
indexer_PT.write().overwrite().save("modelos/indexer_PT")

indexer_NB = StringIndexer(inputCol="NewBuild", outputCol="NewBuildIndex").fit(df)
indexer_NB.write().overwrite().save("modelos/indexer_NB")

indexer_DU = StringIndexer(inputCol="Duration", outputCol="DurationIndex").fit(df)
indexer_DU.write().overwrite().save("modelos/indexer_DU")

indexer_CO = StringIndexer(inputCol="County", outputCol="CountyIndex").fit(df)
indexer_CO.write().overwrite().save("modelos/indexer_CO")

df2 = indexer_PT.transform(df)
df2 = indexer_NB.transform(df2)
df2 = indexer_DU.transform(df2)
df2 = indexer_CO.transform(df2)

assembler = VectorAssembler(
    inputCols=["PropertyTypeIndex", "NewBuildIndex", "DurationIndex", "CountyIndex", "Year", "Month"],
    outputCol="features"
)

df2 = assembler.transform(df2).select("features", "Price")

# --------------------------
# 4. Train/Test 80–20
# --------------------------
train, test = df2.randomSplit([0.8, 0.2], seed=42)

print("\nEntrenando modelo LinearRegression...\n")

lr = LinearRegression(featuresCol="features", labelCol="Price")
modelo = lr.fit(train)

# --------------------------
# 5. Evaluación
# --------------------------
pred = modelo.transform(test)

eval_rmse = RegressionEvaluator(labelCol="Price", predictionCol="prediction", metricName="rmse")
rmse = eval_rmse.evaluate(pred)

eval_r2 = RegressionEvaluator(labelCol="Price", predictionCol="prediction", metricName="r2")
r2 = eval_r2.evaluate(pred)

print(f"📈 RMSE: {rmse:,.2f}")
print(f"📊 R²: {r2:.4f}")

# --------------------------
# 6. Guardar el modelo final
# --------------------------
modelo.write().overwrite().save("modelos/modelo_lineal_precio")

print("\n✅ Modelo guardado en: modelos/modelo_lineal_precio\n")

spark.stop()
print("\n=== ENTRENAMIENTO FINALIZADO ===\n")
