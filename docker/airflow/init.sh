#!/bin/bash
# One-shot (service airflow-init): create / upgrade Airflow's database and the admin user.
set -e
airflow db migrate
airflow users create --role Admin --username "${AIRFLOW_ADMIN_USER}" --password "${AIRFLOW_ADMIN_PASSWORD}" \
  --firstname Rover --lastname Admin --email admin@rover.local \
  || echo "admin user already exists"
