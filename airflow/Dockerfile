FROM apache/airflow:3.4.0
USER root
RUN apt-get update && apt-get install -y --no-install-recommends default-jre-headless procps \
    && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/default-java
USER airflow
COPY requirements.txt /tmp/requirements.txt
RUN python -m venv /opt/airflow/venv && /opt/airflow/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt
