# Model serving API (FastAPI + ONNX Runtime). Loads the registry's @champion over MLflow's REST API, so the
# image needs neither PyTorch nor the MLflow client (~0.5 GB instead of several GB).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_CACHE=/cache

WORKDIR /app
COPY serving/requirements.txt serving/requirements.txt
RUN pip install -r serving/requirements.txt

COPY serving/ serving/

RUN useradd --create-home --uid 1000 app && mkdir -p /cache /models && chown -R app /app /cache
USER app

EXPOSE 8000
# healthy = every configured model is loaded (a container without a model must not get traffic)
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/ready')"
# one worker: ONNX Runtime uses several threads itself (ORT_THREADS), and the model is loaded once
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
