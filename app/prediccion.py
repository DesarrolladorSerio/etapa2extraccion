                         
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.ml.feature import StringIndexerModel, VectorAssembler
from pyspark.ml.regression import LinearRegressionModel
import sys

# =====================================================
# Parámetros externos desde consola
# =====================================================
if len(sys.argv) != 7:
    print("Uso: spark-submit prediccion.py <year> <month> <type> <newbuild> <duration> <city>")
    sys.exit(1)

year = int(sys.argv[1])
month = int(sys.argv[2])
ptype = sys.argv[3].upper()
newb = sys.argv[4].upper()
duration = sys.argv[5].upper()
city = sys.argv[6].upper()

# =====================================================
# Iniciar Spark
# =====================================================
spark = SparkSession.builder.appName("PrediccionPrecioCluster").getOrCreate()

print("\n=== 🧮 MODO PREDICCIÓN (Cluster) ===\n")

# =====================================================
# Cargar indexers guardados
# =====================================================
index_PT = StringIndexerModel.load("modelos/indexer_PT")
index_NB = StringIndexerModel.load("modelos/indexer_NB")
index_DU = StringIndexerModel.load("modelos/indexer_DU")
index_CO = StringIndexerModel.load("modelos/indexer_CO")

# =====================================================
# Crear DataFrame con los parámetros ingresados
# =====================================================

df = spark.createDataFrame(
    [(ptype, newb, duration, city, year, month)],
    ["PropertyType", "NewBuild", "Duration", "County", "Year", "Month"]
)

# =====================================================
# Aplicar transformaciones
# =====================================================
df = index_PT.transform(df)
df = index_NB.transform(df)
df = index_DU.transform(df)
df = index_CO.transform(df)

assembler = VectorAssembler(
    inputCols=["PropertyTypeIndex", "NewBuildIndex", "DurationIndex", "CountyIndex", "Year", "Month"],
    outputCol="features"
)

df = assembler.transform(df)

# =====================================================
# Cargar modelo entrenado
# =====================================================
modelo = LinearRegressionModel.load("modelos/modelo_lineal_precio")

# =====================================================
# Hacer predicción
# =====================================================
pred = modelo.transform(df).collect()[0]["prediction"]

print("\n===== 💰 RESULTADO DE PREDICCIÓN =====")
print(f"Precio estimado: £{pred:,.0f}")
print("======================================\n")

spark.stop()