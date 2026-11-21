-- Runs once, when the PostgreSQL volume is first created.
-- "rover" (POSTGRES_DB) holds the application data; these hold the MLOps tools' metadata.
CREATE DATABASE mlflow;
CREATE DATABASE airflow;
