FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python -m src.roadrisk.train && python -m src.roadrisk.train_clustering
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

