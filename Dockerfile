# ================================================================================
# Dockerfile Multi-Stage para Detección de Enfermedades del Maíz
# ================================================================================
# Este Dockerfile implementa un build multi-stage para optimizar el tamaño
# de la imagen final y mejorar la seguridad.
#
# Uso:
#   docker build -t corn-diseases-detection .
#   docker run -p 8000:8000 corn-diseases-detection
# ================================================================================

# ================================================================================
# STAGE 1: Builder - Instalación de dependencias
# ================================================================================
FROM python:3.10-slim as builder

LABEL maintainer="Corn Diseases Detection Team"
LABEL description="Image classification for corn leaf diseases using Transfer Learning"

# Variables de entorno para Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Instalar dependencias del sistema necesarias para compilación
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar solo archivos de dependencias primero (para aprovechar cache de Docker)
COPY requirements.txt .
COPY pyproject.toml .

# Instalar dependencias de Python en un directorio virtual
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Actualizar pip e instalar dependencias
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# ================================================================================
# STAGE 2: Runtime - Imagen final optimizada
# ================================================================================
FROM python:3.10-slim

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    # TensorFlow optimizations
    TF_CPP_MIN_LOG_LEVEL=2 \
    # Configuración de la aplicación
    APP_HOME=/app \
    # Usuario no-root para seguridad
    USER=appuser \
    UID=1000 \
    GID=1000

# Instalar solo dependencias runtime necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario no-root para ejecutar la aplicación
RUN groupadd -g ${GID} ${USER} && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USER}

# Copiar entorno virtual desde el builder
COPY --from=builder /opt/venv /opt/venv

# Establecer directorio de trabajo
WORKDIR ${APP_HOME}

# Copiar código de la aplicación
COPY --chown=${USER}:${USER} . .

# Crear directorios necesarios con permisos correctos
RUN mkdir -p data models/mlruns models/exported models/tuner_checkpoints logs && \
    chown -R ${USER}:${USER} data models logs

# Cambiar a usuario no-root
USER ${USER}

# Verificar instalación (healthcheck durante build)
RUN python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')" && \
    python -c "import src.core.config; print('Config module OK')"

# Exponer puerto para API (si se implementa FastAPI/Flask)
EXPOSE 8000

# Healthcheck para contenedor
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Punto de entrada por defecto
# Para entrenamiento: docker run corn-diseases-detection train
# Para inferencia: docker run corn-diseases-detection infer
ENTRYPOINT ["python", "-m"]
CMD ["src.pipelines.train"]

# ================================================================================
# METADATA
# ================================================================================
LABEL org.opencontainers.image.title="Corn Diseases Detection"
LABEL org.opencontainers.image.description="Deep Learning model for corn leaf disease classification"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.vendor="Corn Diseases Detection Team"
