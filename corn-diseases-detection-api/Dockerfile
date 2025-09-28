FROM python:3.10-slim

# Evitar bytecode y hacer logs "flusheados"
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Sys deps mínimos
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# App
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY models ./models
COPY run.sh ./run.sh
RUN chmod +x run.sh

EXPOSE 8000
CMD ["./run.sh"]
FROM python:3.10-slim

# Evitar bytecode y hacer logs "flusheados"
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Sys deps mínimos
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# App
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY models ./models
COPY run.sh ./run.sh
RUN chmod +x run.sh

EXPOSE 8000
CMD ["./run.sh"]