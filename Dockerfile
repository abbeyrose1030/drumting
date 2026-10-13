FROM python:3.10-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# GitHub clone leaves vendor/DrumScript empty; fetch the submodule explicitly.
COPY .gitmodules ./
RUN git clone --depth 1 https://github.com/abbeyrose1030/DrumScriptAbbey.git vendor/DrumScript

COPY app ./app

ENV HOST=0.0.0.0
EXPOSE 10000

CMD ["python", "app/server.py"]
