FROM python:3.10-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY vendor ./vendor

ENV HOST=0.0.0.0
ENV PORT=8765
EXPOSE 8765

CMD ["python", "app/server.py"]
