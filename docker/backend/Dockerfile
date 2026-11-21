# Application backend (FastAPI). Small image: no ML libraries, the backend does no inference.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt

COPY config.yaml ./
COPY rover_ai/__init__.py rover_ai/settings.py rover_ai/
COPY backend/ backend/

RUN useradd --create-home --uid 1000 app && mkdir -p alerts && chown -R app /app
USER app

EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=5s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/ready')"
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
