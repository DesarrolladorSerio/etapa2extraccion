# 🧠 Etapa 2 – Exploración de Datos y Herramientas
**Proyecto:** Predicción de precios de viviendas con Apache Spark  
**Curso:** Extracción y Gestión de Datos Masivos 2025-2  
**Integrantes:** *[Agregar nombres del grupo]*  

---

## 📘 Descripción General
Este proyecto corresponde a la **Etapa 2**, cuyo objetivo es **demostrar que los datos elegidos son apropiados** para el análisis, verificando su **calidad, estructura y volumen**, además de **probar las herramientas** que se usarán en etapas posteriores (Spark MLlib en la Etapa 3).  

Se utiliza un **clúster distribuido de Apache Spark 4.0.1** con **3 nodos (1 master y 2 workers)** desplegado con **Docker Compose**.  
El flujo implementa las tres fases principales de un proceso ETL:  
1. **Extracción** del dataset.  
2. **Transformación y limpieza** de datos.  
3. **Análisis exploratorio** mediante agregaciones y estadísticas.  

---

## ⚙️ Requisitos previos
Asegúrate de tener instalado:
- **Docker Desktop** y **Docker Compose**
- **PowerShell** o **Terminal Bash**
- Archivo del proyecto con la siguiente estructura:

```
etapa2/
├── docker-compose.yml
├── app/
│   └── et1_demo.py
└── data/
    └── pp-sample-10k.csv
```

---

## 🧩 Arquitectura del Clúster

| Servicio | Rol | Imagen | Puertos expuestos |
|-----------|-----|--------|-------------------|
| spark-master | Nodo principal (control y monitoreo) | apache/spark:4.0.1 | 9090:8080, 7077:7077 |
| spark-worker-1 | Nodo de ejecución | apache/spark:4.0.1 | — |
| spark-worker-2 | Nodo de ejecución | apache/spark:4.0.1 | — |

📡 Red interna: `sparknet`  
📁 Carpetas compartidas:
- `/app` → scripts de análisis  
- `/data` → datasets CSV  

---

## 🚀 Ejecución paso a paso

### 1️⃣ Levantar el clúster
Desde la carpeta principal del proyecto:
```bash
docker compose up -d
```

Verifica los contenedores:
```bash
docker ps
```
Deberías ver tres contenedores activos:  
`spark-master`, `spark-worker-1`, `spark-worker-2`

Interfaz Web del master:  
👉 [http://localhost:9090](http://localhost:9090)

---

### 2️⃣ Ejecutar el script de análisis
Ejecuta el flujo ETL dentro del contenedor master:

```bash
docker exec -it spark-master bash -c "/opt/spark/bin/spark-submit --master spark://spark-master:7077 /app/et1_demo.py"
```

Esto realiza:
- **Extracción:** lectura paralela del CSV `/data/pp-sample-10k.csv`.  
- **Transformación:** limpieza de filas nulas y conversión de tipos.  
- **Análisis:** cálculo de cantidad de ventas y precios promedio por tipo y región.  

El resultado aparece en consola y puede verse reflejado en la interfaz de Spark UI.

---

### 3️⃣ Verificar la ejecución
Durante la ejecución se observarán:
- *Logs* de tareas distribuidas (workers procesando en paralelo).  
- Información del esquema y recuento de filas/columnas:
  ```
  Filas: 10000 | Columnas: 9
  root
   |-- Price: integer ...
   |-- County: string ...
  ```
- Tabla resumen:
  ```
  +------------------+------------+---------------+-----------------+
  |County            |PropertyType|Cantidad_ventas|Precio_promedio_M|
  +------------------+------------+---------------+-----------------+
  ```

---

### 4️⃣ Detener el clúster
Al finalizar, puedes detener todos los contenedores:
```bash
docker compose down
```

---

## 📊 Resultados esperados

- Dataset correctamente cargado y procesado en Spark.  
- Estadísticas y promedios distribuidos calculados sin errores.  
- Funcionamiento confirmado del clúster (2 workers conectados al master).  
- Prueba de concepto de análisis exploratorio con datos reales.  

---

## 🧠 Conclusiones
- Los datos inmobiliarios del *Land Registry UK* presentan una estructura limpia, coherente y escalable.  
- Spark 4.0.1 demostró manejar el dataset de forma eficiente en modo distribuido.  
- El flujo de **extracción, limpieza y análisis** sienta las bases para las etapas siguientes (modelado predictivo con MLlib).  
