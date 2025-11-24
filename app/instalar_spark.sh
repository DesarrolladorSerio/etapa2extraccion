#!/bin/bash

# ===========================================
# Instalador automático de Apache Spark
# Para clústeres MASTER + WORKERS (Proxmox)
# Autor: ChatGPT
# ===========================================

SPARK_VERSION="4.0.1"
HADOOP_VERSION="3"
SPARK_DIR="/opt/spark"

echo "==========================================="
echo " 🚀 Instalador de Apache Spark $SPARK_VERSION"
echo "==========================================="

# -------------------------------------------
# 1. Actualizar sistema
# -------------------------------------------
echo "[1/8] Actualizando sistema..."
apt update && apt upgrade -y

# -------------------------------------------
# 2. Instalar dependencias
# -------------------------------------------
echo "[2/8] Instalando dependencias..."
apt install -y wget curl tar ssh default-jdk scala python3 python3-pip

# -------------------------------------------
# 3. Descargar Spark
# -------------------------------------------
echo "[3/8] Descargando Apache Spark..."
cd /opt
wget https://archive.apache.org/dist/spark/spark-$SPARK_VERSION/spark-$SPARK_VERSION-bin-hadoop$HADOOP_VERSION.tgz

echo "[4/8] Descomprimiendo..."
tar -xvf spark-$SPARK_VERSION-bin-hadoop$HADOOP_VERSION.tgz
rm spark-$SPARK_VERSION-bin-hadoop$HADOOP_VERSION.tgz
mv spark-$SPARK_VERSION-bin-hadoop$HADOOP_VERSION spark

# -------------------------------------------
# 4. Variables de entorno
# -------------------------------------------
echo "[5/8] Configurando variables de entorno..."

cat <<EOF >/etc/profile.d/spark.sh
export SPARK_HOME=$SPARK_DIR
export PATH=\$PATH:\$SPARK_HOME/bin:\$SPARK_HOME/sbin
export PYSPARK_PYTHON=/usr/bin/python3
EOF

source /etc/profile.d/spark.sh

# -------------------------------------------
# 5. Archivos de configuración
# -------------------------------------------
echo "[6/8] Generando archivos spark-env.sh y workers..."

cd $SPARK_DIR/conf
cp spark-env.sh.template spark-env.sh
cp workers.template workers

echo ""
echo "============================================="
echo " Selecciona el tipo de nodo:"
echo " 1) MASTER"
echo " 2) WORKER"
echo "============================================="
read -p "Opción: " NODE_TYPE

read -p "Ingresa la IP del MASTER: " MASTER_IP

echo "" >> spark-env.sh
echo "SPARK_MASTER_HOST=$MASTER_IP" >> spark-env.sh
echo "SPARK_MASTER_PORT=7077" >> spark-env.sh

if [ "$NODE_TYPE" == "1" ]; then
    echo "[MASTER] Configurando memoria y cores..."
    echo "SPARK_WORKER_MEMORY=4g" >> spark-env.sh
    echo "SPARK_WORKER_CORES=2" >> spark-env.sh

    echo ""
    echo "¿Cuántos workers deseas registrar?"
    read -p "Cantidad: " WORKER_COUNT

    > workers
    for i in $(seq 1 $WORKER_COUNT); do
        read -p "IP del worker $i: " WIP
        echo $WIP >> workers
    done

    echo ""
    echo "=============================================="
    echo " Iniciando MASTER..."
    echo "=============================================="
    start-master.sh

    echo "Master iniciado en:"
    echo "👉 http://$MASTER_IP:8080"

elif [ "$NODE_TYPE" == "2" ]; then
    echo "[WORKER] Fijando memoria y cores..."
    echo "SPARK_WORKER_MEMORY=4g" >> spark-env.sh
    echo "SPARK_WORKER_CORES=2" >> spark-env.sh

    echo ""
    echo "=============================================="
    echo " Registrando WORKER con el master..."
    echo "=============================================="
    start-worker.sh spark://$MASTER_IP:7077
    echo "Worker registrado en el master."
else
    echo "❌ Opción inválida. Abortando instalación."
    exit 1
fi

# -------------------------------------------
# 8. Verificación final
# -------------------------------------------
echo ""
echo "[8/8] Verificando instalación..."
spark-submit --version

echo ""
echo "=============================================="
echo " ✔ Instalación completada correctamente"
echo "=============================================="
echo " Para iniciar manualmente:"
echo "   start-master.sh"
echo "   start-worker.sh spark://$MASTER_IP:7077"
echo ""
echo " Para ver el UI:"
echo "   http://$MASTER_IP:8080"
echo "=============================================="