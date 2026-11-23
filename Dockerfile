FROM python:3.12-slim-bookworm

# Java is needed because Member 1's scripts use Spark
RUN apt-get update && apt-get install -y --no-install-recommends \
        default-jre-headless procps wget unzip \
    && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/default-java

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

# data/ is mounted at run time, not baked into the image
CMD ["python", "src/birank_rerank.py", "--help"]
