# Airflow (scheduler + webserver, LocalExecutor) for the retraining pipeline (airflow/dags/).
# The DAG's steps run the project's own commands (dvc repro, training, registry) in /opt/trainer, a separate
# virtualenv with the training packages, on the project folder mounted at /opt/rover.
FROM apache/airflow:2.10.3-python3.11

USER root
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory '*'
COPY docker/airflow/trainer-requirements.txt /tmp/trainer-requirements.txt
RUN python -m venv /opt/trainer \
    && /opt/trainer/bin/pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
       -r /tmp/trainer-requirements.txt \
    && chown -R airflow: /opt/trainer
USER airflow
RUN pip install --no-cache-dir "pytest>=8"          # tests/test_dag.py runs here (Airflow itself is in this env)

# DVC remote as seen from inside Docker (the project's .dvc/config points at localhost:9000 for the laptop);
# credentials come from AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY at run time
RUN /opt/trainer/bin/dvc remote add --global lake-docker s3://dvc \
    && /opt/trainer/bin/dvc remote modify --global lake-docker endpointurl http://minio:9000
ENV PATH="/home/airflow/.local/bin:${PATH}" TRAINER=/opt/trainer/bin
