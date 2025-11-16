from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, month, avg, count, when
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# ==============================================================
# 🧠 ETAPA 3 – Preparación del experimento
# ==============================================================
# Objetivo: diseñar y ejecutar un experimento que extraiga conocimiento
# relevante del dataset: tendencias y un modelo de regresión simple
# para predecir precios de vivienda según atributos estructurales.
# ==============================================================

# 1️⃣ Inicializar Spark
spark = SparkSession.builder.appName("Etapa3_Experimento").getOrCreate()
print("\n=== 🚀 INICIO EXPERIMENTO ETAPA 3 ===\n")

# 2️⃣ Cargar dataset (extracción)
df = spark.read.csv("/data/pp-sample-10k.csv", header=True, inferSchema=True)
print(f"✅ Dataset cargado: {df.count()} filas, {len(df.columns)} columnas")

# 3️⃣ Limpieza básica (procesamiento)
# Mantener variables relevantes para predicción
df = df.select("Price", "Date", "PropertyType", "NewBuild", "Duration", "TownCity", "County")

# Eliminar filas sin datos críticos
df = df.dropna(subset=["Price", "PropertyType", "County", "Date"])

# Convertir fecha a año y mes
df = df.withColumn("Year", year(col("Date"))).withColumn("Month", month(col("Date")))

# Mostrar esquema final
df.printSchema()

# 4️⃣ Análisis exploratorio avanzado
print("\n=== 📊 Análisis exploratorio ===")
df.groupBy("Year").agg(
    count("*").alias("Num_transacciones"),
    avg("Price").alias("Precio_promedio")
).orderBy("Year").show(10)

df.groupBy("County").agg(
    avg("Price").alias("Precio_promedio"),
    count("*").alias("Ventas")
).orderBy(col("Ventas").desc()).show(10)

# 5️⃣ Codificación de variables categóricas
print("\n=== 🔧 Preparando variables para modelo ===")
indexers = {
    "PropertyType": StringIndexer(inputCol="PropertyType", outputCol="PropertyTypeIndex"),
    "NewBuild": StringIndexer(inputCol="NewBuild", outputCol="NewBuildIndex"),
    "Duration": StringIndexer(inputCol="Duration", outputCol="DurationIndex"),
    "County": StringIndexer(inputCol="County", outputCol="CountyIndex")
}

for key, indexer in indexers.items():
    df = indexer.fit(df).transform(df)

# 6️⃣ Definición del conjunto de características
assembler = VectorAssembler(
    inputCols=["PropertyTypeIndex", "NewBuildIndex", "DurationIndex", "CountyIndex", "Year", "Month"],
    outputCol="features"
)
data = assembler.transform(df).select("features", "Price")

# 7️⃣ División de datos en entrenamiento y prueba
train, test = data.randomSplit([0.8, 0.2], seed=42)

# 8️⃣ Entrenamiento del modelo
print("\n=== 🤖 Entrenando modelo de regresión lineal ===")
lr = LinearRegression(featuresCol="features", labelCol="Price")
modelo = lr.fit(train)

# 9️⃣ Evaluación
predicciones = modelo.transform(test)
evaluador = RegressionEvaluator(labelCol="Price", predictionCol="prediction", metricName="rmse")
rmse = evaluador.evaluate(predicciones)
r2 = modelo.summary.r2

print(f"\n📈 RMSE (error cuadrático medio): {rmse:.2f}")
print(f"📊 R² (coeficiente de determinación): {r2:.4f}")

# 10️⃣ Guardar resultados
predicciones.select("Price", "prediction").limit(20).show()
modelo.write().overwrite().save("/data/modelo_lineal_precio")
print("\n✅ Modelo guardado en /data/modelo_lineal_precio")

# 11️⃣ Cierre
spark.stop()
print("\n=== ✅ EXPERIMENTO FINALIZADO CORRECTAMENTE ===")
