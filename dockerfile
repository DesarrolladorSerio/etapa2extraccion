FROM apache/spark:4.0.1

USER root

# Instalar dependencias necesarias para MLlib
RUN apt-get update && apt-get install -y python3-pip && \
    pip3 install --no-cache-dir numpy pandas pyarrow scipy

USER spark
