# MLflow tracking server + model registry (PostgreSQL metadata, MinIO artifacts)
FROM python:3.11-slim
RUN pip install --no-cache-dir mlflow==2.17.2 "sqlalchemy>=2.0,<2.1" psycopg2-binary==2.9.10 boto3==1.35.54
EXPOSE 5000
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
