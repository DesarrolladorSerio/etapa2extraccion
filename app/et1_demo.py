from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count

# --- Crear sesión Spark conectada al clúster ---
spark = SparkSession.builder \
    .appName("Etapa2_Exploracion") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

print("\n=== 🔹 ETAPA 2: Exploración de datos y herramientas ===\n")

# --- 1. EXTRACCIÓN ---
df = spark.read.option("header", True).csv("/data/pp-sample-10k.csv", inferSchema=True)
print(f"Filas: {df.count()} | Columnas: {len(df.columns)}")
df.printSchema()

# --- 2. PROCESAMIENTO ---
# Limpiar nulos y mantener columnas relevantes
df_clean = df.dropna(subset=["Price", "PropertyType", "County"])

# Convertir precios a millones (solo para visualizar más limpio)
df_clean = df_clean.withColumn("Price_M", col("Price") / 1_000_000)

# --- 3. ANÁLISIS ---
# Promedio de precio por tipo de propiedad y condado
stats = df_clean.groupBy("County", "PropertyType").agg(
    count("*").alias("Cantidad_ventas"),
    avg("Price_M").alias("Precio_promedio_M")
).orderBy(col("Precio_promedio_M").desc())

stats.show(100, truncate=False)

print("\n✅ Proceso completado correctamente.\n")

spark.stop()
