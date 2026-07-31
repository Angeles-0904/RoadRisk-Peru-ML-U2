FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python -m src.roadrisk.train && python -m src.roadrisk.train_clustering
# Render inyecta el puerto real via $PORT; usamos ${PORT:-8000} como fallback local.
# exec: uvicorn queda como PID 1 y recibe las senales (SIGTERM) directamente.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

