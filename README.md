#  Detección de Enfermedades del Maíz con Transfer Learning

Sistema de Deep Learning para clasificación de enfermedades en hojas de maíz utilizando Transfer Learning con VGG16/ResNet50, completamente containerizado con Docker.

---

##  Resumen del Proyecto

Pipeline robusto de Deep Learning para diagnóstico automático de enfermedades comunes en hojas de maíz. El proyecto utiliza Transfer Learning con arquitecturas preentrenadas (VGG16/ResNet50), optimización de hiperparámetros con Keras Tuner, y seguimiento de experimentos con MLflow.

**Características Principales:**
-  **100% Containerizado** - Solo necesitas Docker
-  Transfer Learning con VGG16/ResNet50
-  Optimización con Keras Tuner
-  Tracking de experimentos con MLflow
-  API REST con FastAPI
-  Suite completa de tests automatizados
-  Gestión de configuración con Pydantic

---

##  Clases de Enfermedades

El modelo clasifica 4 categorías:

1. **Blight** (Tizón)
2. **Common_Rust** (Roya Común)
3. **Gray_Leaf_Spot** (Mancha Gris)
4. **Healthy** (Saludable)

---

##  Estructura del Proyecto

```
corn-diseases-detection/
 data/                       # Dataset (ignorado por git)
    train/                  # 3,856 imágenes (balanceado)
    val/                    # 716 imágenes (estratificado)
    test/                   # 722 imágenes (estratificado)

 src/                        # Código fuente
    adapters/               # Cargadores de datos
    api/                    # API REST (FastAPI)
    builders/               # Constructores de modelos
    core/                   # Configuración central
    pipelines/              # Pipelines ML (train, infer, preprocess)
    utils/                  # Utilidades

 tests/                      # Suite de tests (10 archivos)

 experimentation/            # Scripts EDA y notebooks

 experiments/                #  Experimentos edge computing
    edge_models/            # Entrenamiento arquitecturas livianas
        train_edge_model.py
        train_all_models.py
        compare_models.py
        select_best_model.py
        README.md
        best_edge_model.json  # Salida: mejor modelo seleccionado

 models/                     # Modelos entrenados (ignorado por git)
    exported/               # Modelos finales (.keras)
    mlruns/                 # Tracking MLflow
    tuner_checkpoints/      # Keras Tuner

 docker-compose.yml          # Orquestación de servicios
 Dockerfile                  # Imagen multi-stage optimizada
 requirements.txt            # Dependencias Python
 README.md                   # Este archivo
```

---

##  Inicio Rápido

### Opción 1: Google Colab (Recomendado - GPU Gratis) 🚀

**Ideal si tienes problemas de memoria en Docker o no tienes GPU local**

1. Sube tus datos a Google Drive en: `MyDrive/corn-diseases-data/`
2. Abre el notebook: `colab_edge_models_training.ipynb` en [Google Colab](https://colab.research.google.com)
3. Runtime > Change runtime type > GPU (T4)
4. Ejecuta todas las celdas

📖 **Guía completa:** Ver `COLAB_SETUP.md`

⏱️ **Tiempo:** 2-3 horas con GPU T4 (vs 20-30 horas en CPU)

---

### Opción 2: Docker Local

#### Requisitos

- **Docker Desktop** instalado ([Descargar](https://www.docker.com/products/docker-desktop))
- **Git** para clonar el repositorio
- **Datos** en `data/train`, `data/val`, `data/test`
- **Memoria:** Mínimo 8GB RAM disponible para Docker

---

##  **NUEVO: Experimentos Edge Computing**

### Entrenamiento de Arquitecturas Livianas

El proyecto incluye un sistema completo para evaluar **4 arquitecturas** optimizadas para edge computing:

**Arquitecturas evaluadas:**
- **MobileNetV3Large** - Balance óptimo tamaño/precisión (~5.4M params, ~21MB)
- **EfficientNet-Lite B2** - Máxima eficiencia (~10.1M params, ~42MB)
- **MobileViT** - Vision Transformer móvil (~6.4M params, ~25MB) + fine-tuning
- **PMVT** - Específico para plantas (~6M params, ~24MB) + fine-tuning

### Ejecutar Experimentos Completos

```bash
# Entrenar TODAS las arquitecturas edge automáticamente
docker-compose --profile edge-experiments up
```

Esto ejecuta:
1.  Entrenamiento de 4 arquitecturas livianas seleccionadas
2.  Comparación automática de resultados
3.  Selección del mejor modelo
4.  Generación de `best_edge_model.json`

**Criterios de selección:**
- Precisión global ≥ 85%
- Recall por clase ≥ 0.80
- Mejor balance precisión/tamaño

**Salida:** `experiments/edge_models/best_edge_model.json`

### Ver Resultados

```bash
# MLflow UI para ver todos los experimentos
docker-compose --profile mlflow up -d
open http://localhost:5000

# Buscar experimento: "edge_models_comparison"
```

 **Documentación completa:** `experiments/edge_models/README.md`

### 1. Clonar Repositorio

```bash
git clone https://github.com/ojgonzalezz/corn-diseases-detection.git
cd corn-diseases-detection
```

### 2. Construir Imagen Docker

```bash
# Construir imagen (incluye TensorFlow, Keras, MLflow)
docker-compose build
```

Esto instalará automáticamente:
- TensorFlow 2.20.0
- Keras 3.11.3
- MLflow 3.3.2
- FastAPI + todas las dependencias

### 3. Entrenar Modelo

```bash
# Entrenar modelo con tus datos
docker-compose --profile training up

# O en background
docker-compose --profile training up -d

# Ver logs
docker-compose logs -f training
```

El modelo se guardará en `models/exported/best_VGG16.keras`

### 4. Iniciar API de Inferencia

```bash
# Iniciar API
docker-compose --profile api up -d

# Acceder a documentación
open http://localhost:8000/docs
```

### 5. Ver Experimentos en MLflow

```bash
# Iniciar MLflow UI
docker-compose --profile mlflow up -d

# Acceder a dashboard
open http://localhost:5000
```

---

##  Servicios Docker Disponibles

| Servicio | Profile | Puerto | Comando | Descripción |
|----------|---------|--------|---------|-------------|
| **training** | `training` | - | `docker-compose --profile training up` | Entrenamiento estándar |
| **edge-experiments** | `edge-experiments` | - | `docker-compose --profile edge-experiments up` |  Entrenar modelos edge |
| **api** | `api` | 8000 | `docker-compose --profile api up -d` | API REST predicciones |
| **mlflow** | `mlflow` | 5000 | `docker-compose --profile mlflow up -d` | UI experimentos |
| **notebook** | `notebook` | 8888 | `docker-compose --profile notebook up -d` | Jupyter Lab |
| **preprocessing** | `preprocessing` | - | `docker-compose --profile preprocessing up` | Preprocesar datos |

---

##  Uso de la API

### Endpoints Disponibles

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Información del Modelo:**
```bash
curl http://localhost:8000/info
```

**Predicción Individual:**
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@ruta/a/imagen.jpg"
```

**Predicción por Lotes:**
```bash
curl -X POST http://localhost:8000/batch-predict \
  -F "files=@imagen1.jpg" \
  -F "files=@imagen2.jpg" \
  -F "files=@imagen3.jpg"
```

### Ejemplo Python

```python
import requests

# Predicción
with open('hoja_maiz.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f}
    )

result = response.json()
print(f"Predicción: {result['prediction']['predicted_label']}")
print(f"Confianza: {result['prediction']['confidence']:.2%}")
```

---

##  Configuración

### Variables de Entorno

El proyecto usa un archivo `.env` para configuración. Todas las variables tienen valores por defecto en `src/core/.env_example`.

**Variables Principales:**

| Variable | Valor por Defecto | Descripción |
|----------|-------------------|-------------|
| `IMAGE_SIZE` | `(224, 224)` | Dimensiones de entrada |
| `NUM_CLASSES` | `4` | Número de clases |
| `BATCH_SIZE` | `32` | Tamaño del batch |
| `MAX_EPOCHS` | `20` | Épocas máximas |
| `BACKBONE` | `VGG16` | Arquitectura base |
| `BALANCE_STRATEGY` | `oversample` | Estrategia de balanceo |

Para personalizar, edita `src/core/.env`

---

##  Testing

### Ejecutar Tests en Docker

```bash
# Todos los tests
docker-compose run --rm training pytest tests/ -v

# Tests específicos
docker-compose run --rm training pytest tests/test_train.py -v

# Tests sin módulos lentos
docker-compose run --rm training pytest tests/ -m "not slow" -v
```

### Cobertura de Tests

El proyecto incluye **10 archivos de tests** con **~90% de cobertura**:
- `test_train.py` - Pipeline de entrenamiento
- `test_infer.py` - Pipeline de inferencia
- `test_preprocess.py` - Preprocesamiento
- `test_augmentation.py` - Augmentación de datos
- `test_config.py` - Sistema de configuración
- `test_builders.py` - Constructores de modelos
- `test_data_loader.py` - Carga de datos
- `test_logger.py` - Sistema de logging
- `test_paths.py` - Gestión de rutas
- `test_api.py` - Endpoints de API

---

##  Comandos Docker Útiles

### Gestión de Contenedores

```bash
# Ver contenedores corriendo
docker-compose ps

# Ver logs
docker-compose logs -f [servicio]

# Detener todos los servicios
docker-compose down

# Detener y eliminar volúmenes ( elimina datos)
docker-compose down -v

# Reconstruir imagen desde cero
docker-compose build --no-cache
```

### Comandos Únicos

```bash
# Ejecutar cualquier comando en el contenedor
docker-compose run --rm training python -m src.pipelines.train

# Shell interactiva
docker-compose run --rm training bash

# Verificar versión de TensorFlow
docker-compose run --rm training python -c "import tensorflow as tf; print(tf.__version__)"
```

---

##  Personalización del Entrenamiento

### Entrenar con Diferente Backbone

Edita `src/core/.env`:
```bash
BACKBONE=ResNet50
```

Luego ejecuta:
```bash
docker-compose --profile training up
```

### Ajustar Hiperparámetros

Edita `src/core/.env`:
```bash
BATCH_SIZE=64
MAX_EPOCHS=50
MAX_TRIALS=20
BALANCE_STRATEGY=downsample
```

---

##  Estructura de Datos

### Formato Esperado

El proyecto soporta dos estructuras:

**Opción 1: Datos Ya Divididos (Recomendado)**
```
data/
 train/
    Blight/
    Common_Rust/
    Gray_Leaf_Spot/
    Healthy/
 val/
    ...
 test/
     ...
```

**Opción 2: Datos Raw para Preprocesar**
```
data/
 raw/
     data_1/
        [clases]/
     data_2/
         [clases]/
```

Si tienes datos raw, ejecuta:
```bash
docker-compose --profile preprocessing up
```

---

##  Características Avanzadas

### De-augmentación Inteligente

El sistema detecta y filtra imágenes duplicadas usando embeddings de ResNet50:

```bash
# Configurar umbral en src/core/.env
IM_SIM_THRESHOLD=0.95  # 0.0 a 1.0 (más alto = más estricto)
```

### Balanceo de Clases

Tres estrategias disponibles:

1. **Oversample** (por defecto) - Aumenta clases minoritarias
2. **Downsample** - Reduce clases mayoritarias
3. **None** - Sin balanceo

```bash
# En src/core/.env
BALANCE_STRATEGY=oversample
```

### Tracking con MLflow

Todos los experimentos se registran automáticamente:
- Hiperparámetros
- Métricas (accuracy, loss)
- Modelos entrenados
- Configuración completa

```bash
# Ver experimentos
docker-compose --profile mlflow up -d
open http://localhost:5000
```

---

##  Versionado de Modelos

Los modelos se guardan automáticamente con:

```
models/exported/
 VGG16_20251002_143022_acc0.9745.keras    # Con timestamp + accuracy
 VGG16_20251002_143022_metadata.json      # Metadatos de entrenamiento
 best_VGG16.keras                         # Último mejor modelo
```

Los metadatos incluyen:
- Timestamp
- Accuracy y loss en test
- Hiperparámetros utilizados
- Configuración completa

---

##  Troubleshooting Docker

### Error: "Cannot connect to Docker daemon"
```bash
# Iniciar Docker Desktop en Mac
# Verificar que está corriendo en la barra de menú
```

### Error: "Port already in use"
```bash
# Cambiar puerto en docker-compose.yml
ports:
  - "8001:8000"  # API en puerto 8001
```

### Limpiar Sistema Docker
```bash
# Limpiar contenedores detenidos
docker system prune

# Ver uso de espacio
docker system df

# Eliminar todo ( cuidado)
docker system prune -a
```

### Problemas de Memoria
```bash
# Editar docker-compose.yml para limitar recursos
services:
  training:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
```

---

##  Workflow Típico

### 1. Preparar Datos
```bash
# Si tienes datos raw
docker-compose --profile preprocessing up

# Verifica estructura en data/train, data/val, data/test
```

### 2. Entrenar Modelo
```bash
# Entrenar
docker-compose --profile training up

# El mejor modelo se guarda automáticamente
```

### 3. Evaluar Modelo
```bash
# Evaluar modelo guardado
docker-compose --profile evaluation up
```

### 4. Desplegar API
```bash
# Iniciar API
docker-compose --profile api up -d

# Probar
curl http://localhost:8000/health
```

### 5. Hacer Predicciones
```bash
# Via API
curl -X POST http://localhost:8000/predict \
  -F "file=@imagen_hoja.jpg"

# Verifica en:
open http://localhost:8000/docs
```

---

##  Arquitectura del Sistema

### Pipeline de Datos
1. **Carga** - `src/adapters/data_loader.py`
2. **Preprocesamiento** - `src/pipelines/preprocess.py`
   - Detección de duplicados por embeddings
   - División estratificada
   - Balanceo de clases (oversample/downsample)
3. **Augmentación** - `src/utils/data_augmentator.py`

### Pipeline de Entrenamiento
1. **Construcción** - `src/builders/builders.py`
   - Carga backbone (VGG16/ResNet50)
   - Ensambla cabeza de clasificación
2. **Tuning** - Keras Tuner con Hyperband
3. **Tracking** - MLflow registra todo
4. **Guardado** - Versionado automático

### Pipeline de Inferencia
1. **Carga** - `src/pipelines/infer.py`
2. **API** - `src/api/main.py`
   - Endpoint `/predict`
   - Endpoint `/batch-predict`
3. **Respuesta** - JSON con probabilidades

---

##  Sistema de Configuración

### Gestión Centralizada con Pydantic

```python
from src.core.config import config

# Acceso type-safe
image_size = config.data.image_size        # (224, 224)
batch_size = config.training.batch_size    # 32
backbone = config.training.backbone        # 'VGG16'
```

Validación automática de:
- Tipos de datos
- Rangos de valores
- Consistencia entre variables

---

##  Testing Automatizado

**Cobertura:** ~90%  
**Tests:** 10 archivos, 3,000+ líneas

```bash
# Ejecutar todos los tests
docker-compose run --rm training pytest tests/ -v

# Con detalles
docker-compose run --rm training pytest tests/ -vv

# Solo tests rápidos
docker-compose run --rm training pytest tests/ -m "not slow"
```

---

##  Seguridad

-  Contenedores corren con usuario no-root
-  Variables sensibles en `.env` (no commiteado)
-  Multi-stage build minimiza superficie de ataque
-  Dependencias con versiones fijas

---

##  Volúmenes Docker

Los datos y modelos persisten entre reinicios:

```yaml
volumes:
  - ./data:/app/data           # Datos
  - ./models:/app/models       # Modelos entrenados
  - ./src:/app/src             # Código (solo lectura)
```

**Nota:** Los modelos entrenados se guardan en tu máquina local en `./models/`

---

##  Configuración Avanzada

### GPU Support (NVIDIA)

Descomentar en `docker-compose.yml`:

```yaml
training:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

Requiere: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Limitar Recursos

```yaml
training:
  deploy:
    resources:
      limits:
        cpus: '4.0'
        memory: 8G
```

---

##  Despliegue en Producción

### Cloud Run (Google Cloud)

```bash
# Build y deploy
gcloud builds submit --tag gcr.io/PROJECT-ID/corn-api
gcloud run deploy corn-api \
  --image gcr.io/PROJECT-ID/corn-api \
  --platform managed \
  --region us-central1 \
  --memory 4Gi
```

### AWS ECS

```bash
# Push a ECR
aws ecr get-login-password | docker login --username AWS --password-stdin ECR_URL
docker tag corn-diseases-detection:latest ECR_URL/corn-api:latest
docker push ECR_URL/corn-api:latest

# Deploy en ECS (usar consola o Terraform)
```

### Heroku

```bash
heroku create corn-diseases-detection
heroku stack:set container
git push heroku main
```

---

##  Uso Programático

### Pipeline de Inferencia

```python
from src.pipelines.infer import predict

# Cargar imagen
with open('hoja_maiz.jpg', 'rb') as f:
    image_bytes = f.read()

# Predecir
result = predict(image_bytes)

print(f"Enfermedad: {result['predicted_label']}")
print(f"Confianza: {result['confidence']:.2%}")
print(f"Probabilidades: {result['all_probabilities']}")
```

### Pipeline de Entrenamiento

```python
from src.pipelines.train import train

# Entrenar con parámetros personalizados
tuner, (X_test, y_test) = train(
    backbone_name='ResNet50',
    split_ratios=(0.7, 0.15, 0.15),
    balanced='oversample'
)
```

---

##  Desarrollo

### Ejecutar Tests

```bash
docker-compose run --rm training pytest tests/ -v
```

### Acceder al Contenedor

```bash
# Shell interactiva
docker-compose run --rm training bash

# Explorar estructura
ls -la /app/src
```

### Jupyter para Experimentación

```bash
# Iniciar Jupyter Lab
docker-compose --profile notebook up -d

# Acceder
open http://localhost:8888

# Notebooks en: experimentation/notebooks/
```

---

##  Monitoreo

### Logs de Contenedores

```bash
# Logs en tiempo real
docker-compose logs -f training

# Últimas 100 líneas
docker-compose logs --tail=100 api

# Todos los servicios
docker-compose logs -f
```

### Health Checks

```bash
# API
curl http://localhost:8000/health

# MLflow
curl http://localhost:5000/health

# En scripts
docker-compose ps
```

---

##  Documentación Adicional

- **API Docs:** http://localhost:8000/docs (Swagger)
- **API ReDoc:** http://localhost:8000/redoc
- **MLflow UI:** http://localhost:5000
- **OpenAPI Schema:** http://localhost:8000/openapi.json

---

##  Contribuir

1. Fork el repositorio
2. Crea una rama (`git checkout -b feature/nueva-funcionalidad`)
3. Haz tus cambios
4. Ejecuta tests: `docker-compose run --rm training pytest tests/`
5. Commit (`git commit -m 'feat: nueva funcionalidad'`)
6. Push (`git push origin feature/nueva-funcionalidad`)
7. Abre un Pull Request

---

##  Licencia

Este proyecto está bajo la licencia MIT.

---

##  Soporte

- **Issues:** [GitHub Issues](https://github.com/ojgonzalezz/corn-diseases-detection/issues)
- **Repository:** [ojgonzalezz/corn-diseases-detection](https://github.com/ojgonzalezz/corn-diseases-detection)

---

** Desarrollado con Transfer Learning y Docker para máxima reproducibilidad**
