# Advanced-RAG-System — one image, four commands (api / ingestion / outbox / reaper).
# docker-compose.yml picks the command per service.
FROM python:3.12-slim

WORKDIR /srv

# lxml needs libxml2/libxslt at runtime; build tools only for the install layer
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libxml2-dev libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY workers ./workers
COPY scripts ./scripts
COPY evaluation ./evaluation
COPY migrations ./migrations
COPY frontend ./frontend

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
