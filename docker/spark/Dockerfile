# syntax=docker/dockerfile:1
# Spark Structured Streaming job (streaming/spark_job.py), the alert bridge (streaming/bridge.py) and the drift
# monitor (monitoring/monitor.py): one image for the Python data services.
# Spark runs in local mode (one container, 2 cores): enough for a few cameras, and the same job runs unchanged on a
# Spark cluster (spark-submit --master spark://... or k8s://...).
FROM apache/spark:3.5.3-scala2.12-java17-python3-ubuntu

USER root
# Kafka source/sink for Structured Streaming (not part of the Spark distribution), pinned to Spark 3.5.3
RUN cd /opt/spark/jars && python3 - <<'PY'
import urllib.request
base = "https://repo1.maven.org/maven2/"
for path in ["org/apache/spark/spark-sql-kafka-0-10_2.12/3.5.3/spark-sql-kafka-0-10_2.12-3.5.3.jar",
             "org/apache/spark/spark-token-provider-kafka-0-10_2.12/3.5.3/spark-token-provider-kafka-0-10_2.12-3.5.3.jar",
             "org/apache/kafka/kafka-clients/3.4.1/kafka-clients-3.4.1.jar",
             "org/apache/commons/commons-pool2/2.11.1/commons-pool2-2.11.1.jar"]:
    urllib.request.urlretrieve(base + path, path.rsplit("/", 1)[1])
PY

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app PYSPARK_PYTHON=python3
WORKDIR /app
COPY streaming/requirements.txt streaming/requirements.txt
RUN pip3 install -r streaming/requirements.txt

# only the light parts of the project: feature code, tracks, model client (no PyTorch / OpenCV / YOLO)
COPY rover_ai/ rover_ai/
COPY mlops/__init__.py mlops/lake.py mlops/
COPY streaming/ streaming/
COPY monitoring/ monitoring/
RUN mkdir -p /checkpoints /home/spark && chown -R spark /checkpoints /app /home/spark
USER spark
ENV HOME=/home/spark

# 4040 Spark UI + Spark metrics, 9101 pipeline metrics, 9200 drift monitor (same image, other command)
EXPOSE 4040 9101 9200
CMD ["/opt/spark/bin/spark-submit", "--master", "local[2]", "--driver-memory", "768m", \
     "--conf", "spark.ui.port=4040", "/app/streaming/spark_job.py"]
