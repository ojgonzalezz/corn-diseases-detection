# Makefile para el proyecto Corn Diseases Detection
# Facilita tareas comunes de desarrollo, testing y CI/CD

.PHONY: help install install-dev setup test test-cov lint format clean pre-commit verify-setup train
.PHONY: docker-build docker-train docker-mlflow docker-notebook docker-clean docker-shell

# Variables
PYTHON := python3
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
BLACK := $(PYTHON) -m black
ISORT := $(PYTHON) -m isort
FLAKE8 := $(PYTHON) -m flake8
MYPY := $(PYTHON) -m mypy
DOCKER_COMPOSE := docker-compose

help:  ## Mostrar este mensaje de ayuda
	@echo "Comandos disponibles:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Instalar dependencias básicas
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✓ Dependencias instaladas"

install-dev:  ## Instalar dependencias de desarrollo
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	@echo "✓ Dependencias de desarrollo instaladas"

setup:  ## Configuración inicial completa del proyecto
	@echo "=== Configuración inicial del proyecto ==="
	$(MAKE) install-dev
	@if [ ! -f src/core/.env ]; then \
		cp src/core/.env_example src/core/.env; \
		echo "✓ Archivo .env creado desde .env_example"; \
	else \
		echo "ℹ Archivo .env ya existe"; \
	fi
	$(PIP) install pre-commit
	pre-commit install
	@echo "✓ Pre-commit hooks instalados"
	$(PYTHON) verify_setup.py
	@echo "✓ Configuración completada"

verify-setup:  ## Verificar que el proyecto está correctamente configurado
	$(PYTHON) verify_setup.py

test:  ## Ejecutar tests
	$(PYTEST) tests/ -v

test-cov:  ## Ejecutar tests con reporte de cobertura
	$(PYTEST) tests/ -v --cov=src --cov-report=term-missing --cov-report=html
	@echo "✓ Reporte HTML generado en: htmlcov/index.html"

test-fast:  ## Ejecutar solo tests rápidos (excluir tests lentos)
	$(PYTEST) tests/ -v -m "not slow"

lint:  ## Verificar calidad de código (flake8)
	$(FLAKE8) src/ tests/ --max-line-length=120 --extend-ignore=E203,W503

format-check:  ## Verificar formato de código sin modificar
	$(BLACK) --check src/ tests/
	$(ISORT) --check-only src/ tests/

format:  ## Formatear código automáticamente (black + isort)
	$(BLACK) src/ tests/
	$(ISORT) src/ tests/
	@echo "✓ Código formateado"

type-check:  ## Verificar tipos con mypy
	$(MYPY) src/ --ignore-missing-imports

pre-commit:  ## Ejecutar pre-commit hooks manualmente
	pre-commit run --all-files

pre-commit-update:  ## Actualizar versiones de pre-commit hooks
	pre-commit autoupdate

clean:  ## Limpiar archivos temporales y cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage.xml
	rm -rf build/ dist/
	@echo "✓ Archivos temporales eliminados"

clean-data:  ## Limpiar datos procesados (WARNING: elimina data/processed)
	@echo "⚠️  ADVERTENCIA: Esto eliminará data/processed/"
	@read -p "¿Continuar? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf data/processed/; \
		echo "✓ Datos procesados eliminados"; \
	fi

clean-models:  ## Limpiar modelos y experimentos MLflow
	@echo "⚠️  ADVERTENCIA: Esto eliminará models/ y mlruns/"
	@read -p "¿Continuar? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf models/ mlruns/; \
		echo "✓ Modelos y experimentos eliminados"; \
	fi

train:  ## Entrenar modelo con configuración por defecto
	$(PYTHON) -m src.pipelines.train

train-vgg16:  ## Entrenar con VGG16
	$(PYTHON) -c "from src.pipelines.train import train; train(backbone_name='VGG16', balanced='oversample')"

train-resnet50:  ## Entrenar con ResNet50
	$(PYTHON) -c "from src.pipelines.train import train; train(backbone_name='ResNet50', balanced='oversample')"

mlflow-ui:  ## Abrir interfaz de MLflow
	mlflow ui

jupyter:  ## Iniciar Jupyter Lab
	jupyter lab

check-gpu:  ## Verificar disponibilidad de GPU
	$(PYTHON) -c "from src.utils.utils import check_cuda_availability; check_cuda_availability()"

ci:  ## Ejecutar todas las validaciones de CI localmente
	@echo "=== Ejecutando validaciones de CI ==="
	$(MAKE) format-check
	$(MAKE) lint
	$(MAKE) test-cov
	@echo "✓ Todas las validaciones pasaron"

all:  ## Ejecutar setup completo + tests
	$(MAKE) setup
	$(MAKE) test

# ================================================================================
# Docker Commands
# ================================================================================

docker-build:  ## Construir imagen Docker
	$(DOCKER_COMPOSE) build
	@echo "✓ Imagen Docker construida"

docker-train:  ## Entrenar modelo usando Docker
	$(DOCKER_COMPOSE) --profile training up

docker-train-bg:  ## Entrenar modelo en background
	$(DOCKER_COMPOSE) --profile training up -d
	@echo "✓ Entrenamiento iniciado en background"
	@echo "Ver logs: make docker-logs-training"

docker-preprocess:  ## Preprocesar datos usando Docker
	$(DOCKER_COMPOSE) --profile preprocessing up

docker-mlflow:  ## Iniciar MLflow UI usando Docker
	$(DOCKER_COMPOSE) --profile mlflow up -d
	@echo "✓ MLflow UI disponible en: http://localhost:5000"

docker-notebook:  ## Iniciar Jupyter Lab usando Docker
	$(DOCKER_COMPOSE) --profile notebook up -d
	@echo "✓ Jupyter Lab disponible en: http://localhost:8888"

docker-api:  ## Iniciar API de inferencia usando Docker
	$(DOCKER_COMPOSE) --profile api up -d
	@echo "✓ API disponible en: http://localhost:8000"
	@echo "Documentación: http://localhost:8000/docs"

docker-logs-training:  ## Ver logs del entrenamiento
	$(DOCKER_COMPOSE) logs -f training

docker-logs-mlflow:  ## Ver logs de MLflow
	$(DOCKER_COMPOSE) logs -f mlflow

docker-shell:  ## Abrir shell interactiva en el contenedor
	$(DOCKER_COMPOSE) run --rm training bash

docker-test:  ## Ejecutar tests en Docker
	$(DOCKER_COMPOSE) run --rm training pytest tests/ -v

docker-stop:  ## Detener todos los contenedores
	$(DOCKER_COMPOSE) down
	@echo "✓ Contenedores detenidos"

docker-clean:  ## Limpiar contenedores y volúmenes (⚠️ elimina datos)
	@echo "⚠️  ADVERTENCIA: Esto eliminará contenedores, redes y volúmenes"
	@read -p "¿Continuar? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		$(DOCKER_COMPOSE) down -v; \
		echo "✓ Contenedores y volúmenes eliminados"; \
	fi

docker-clean-all:  ## Limpiar todo (contenedores, volúmenes, imágenes)
	@echo "⚠️  ADVERTENCIA: Esto eliminará TODO (contenedores, volúmenes, imágenes)"
	@read -p "¿Continuar? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		$(DOCKER_COMPOSE) down -v --rmi all; \
		echo "✓ Todo eliminado"; \
	fi

docker-rebuild:  ## Reconstruir imagen desde cero
	$(DOCKER_COMPOSE) build --no-cache
	@echo "✓ Imagen reconstruida desde cero"

.DEFAULT_GOAL := help
