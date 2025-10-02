-----

# Detección de Enfermedades del Maíz con Transfer Learning

-----

## Resumen del Proyecto

Este proyecto implementa un pipeline de Deep Learning robusto para la clasificación de enfermedades comunes en hojas de maíz. El objetivo es diagnosticar automáticamente la salud de las plantas utilizando técnicas de Transfer Learning basadas en la arquitectura VGG16, optimizando la cabeza de clasificación mediante Keras Tuner y rastreando todos los experimentos con MLflow.

El proyecto se destaca por su rigurosa estrategia de preprocesamiento, que aborda activamente el sesgo por duplicación (data leakage) y el desbalance de clases en un dataset bimodal compuesto por dos fuentes de datos distintas.

**Características Principales:**
- Transfer Learning con VGG16/ResNet50
- De-augmentación inteligente usando embeddings de ResNet50
- Balanceo avanzado de clases (oversample/downsample)
- Seguimiento de experimentos con MLflow
- Optimización de hiperparámetros con Keras Tuner
- Gestión de configuración validada con Pydantic
- Sistema de logging profesional con colores
- Suite completa de tests con pytest
- Manejo robusto de excepciones con sugerencias de recuperación

-----

## Problema y Contexto

Las enfermedades del maíz, como la roya común, el tizón foliar y la mancha gris, representan una amenaza crítica para la seguridad alimentaria. El diagnóstico tradicional mediante inspección visual es un proceso lento, subjetivo y dependiente de la pericia del observador. Este proyecto busca validar la viabilidad de un sistema de diagnóstico automatizado mediante Inteligencia Artificial para superar estas limitaciones.

-----

## Objetivo y Tipos de Datos

### Objetivo Principal

Desarrollar un modelo de clasificación de imágenes altamente preciso y generalizable, capaz de diferenciar entre las siguientes cuatro categorías de salud de las hojas de maíz:

1. **Blight**
2. **Common_Rust**
3. **Gray_Leaf_Spot**
4. **Healthy**

### Tipos de Datos

El dataset está compuesto por imágenes RGB de hojas de maíz, recopiladas de dos fuentes distintas que se manejan por separado para controlar el Data Augmentation:

| Fuente | Descripción | Consideración |
| :--- | :--- | :--- |
| **data_1** | Dataset limpio, sin aumentación sintética | no-augmentation |
| **data_2** | Dataset con aumentación sintética aplicada | augmented |

-----

## Estructura Detallada del Repositorio

El proyecto sigue una estructura modular y escalable para separar el código de producción (src), los datos (data), y la experimentación (experimentation).

```
corn-diseases-detection/
├── data/                     # Directorio de dataset (ignorado por git)
│   ├── train/                # Conjunto de entrenamiento (3,856 imágenes - balanceado)
│   │   ├── Blight/           # 964 imágenes
│   │   ├── Common_Rust/      # 964 imágenes
│   │   ├── Gray_Leaf_Spot/   # 964 imágenes
│   │   └── Healthy/          # 964 imágenes
│   ├── val/                  # Conjunto de validación (716 imágenes - estratificado)
│   │   ├── Blight/           # 171 imágenes
│   │   ├── Common_Rust/      # 195 imágenes
│   │   ├── Gray_Leaf_Spot/   # 176 imágenes
│   │   └── Healthy/          # 174 imágenes
│   └── test/                 # Conjunto de prueba (722 imágenes - estratificado)
│       ├── Blight/           # 173 imágenes
│       ├── Common_Rust/      # 197 imágenes
│       ├── Gray_Leaf_Spot/   # 177 imágenes
│       └── Healthy/          # 175 imágenes
│
├── experimentation/          # Scripts de EDA y exploración
│   ├── eda/                  # Validación y análisis de datos
│   │   ├── analyze_feature.py
│   │   ├── explore_distribution.py
│   │   ├── validate_dataset.py
│   │   └── view_samples.py
│   └── notebooks/
│   ├── 01_eda_exploracion.ipynb
│   ├── 02_modelado_basico.ipynb
│   └── 03_transfer_learning.ipynb
│
├── models/                   # Artefactos de modelos (ignorado por git)
│   ├── mlruns/               # Seguimiento de experimentos con MLflow
│   ├── tuner_checkpoints/    # Búsqueda de hiperparámetros con Keras Tuner
│   └── exported/             # Modelos finales entrenados (best_VGG16.keras, etc.)
│
├── src/                      # Código fuente de producción
│   ├── adapters/
│   │   └── data_loader.py    # Abstracción: Carga de datos de múltiples fuentes
│   ├── builders/
│   │   ├── base_models.py    # Definición de backbones preentrenados (VGG16)
│   │   └── builders.py       # Ensamblaje de la cabeza de clasificación para Keras Tuner
│   ├── core/                 # Configuración y utilidades de entorno
│   │   ├── load_env.py       # Carga de variables de entorno (.env)
│   │   ├── config.py         # Gestión de configuración con Pydantic (validación)
│   │   └── path_finder.py    # Detección de la ruta raíz del proyecto
│   ├── pipelines/            # Scripts del ciclo de Machine Learning
│   │   ├── data_pipeline.py  # Generación de DataGenerators para Keras
│   │   ├── evaluate_finetuned.py
│   │   ├── train.py          # Orquestación del entrenamiento con Keras Tuner/MLflow
│   │   ├── preprocess.py     # Script principal de filtrado, unificación y balanceo
│   │   └── infer.py          # Lógica de inferencia para la API (clasificación)
│   └── utils/                # Funciones de ayuda
│       ├── aug_detectors.py    # Detección y filtrado de aumentaciones por Embedding
│       ├── data_augmentator.py # Transformaciones espaciales para Oversampling
│       ├── image_modifier.py   # Transformaciones de calidad (brillo, contraste, ruido)
│       ├── utils.py            # Utilidades misceláneas (flatten_data, split, etc.)
│       ├── paths.py            # Manejo centralizado de rutas del proyecto
│       ├── logger.py           # Sistema de logging profesional con colores
│       └── exceptions.py       # Excepciones personalizadas con sugerencias
│
├── tests/                    # Suite de pruebas con pytest
│   ├── __init__.py
│   ├── conftest.py           # Fixtures compartidas
│   ├── test_preprocess.py    # Tests de preprocesamiento y split
│   └── test_augmentation.py  # Tests de augmentación de imágenes
│
├── requirements.txt
├── pyproject.toml            # Configuración del proyecto (PEP 518)
├── README.md
└── .gitignore
```

### Nota sobre el Flujo de Datos

**Estructura Actual:** El proyecto utiliza datos pre-divididos en los directorios `data/train/`, `data/val/`, `data/test/`. Esta es la estructura de trabajo real.

**Justificación:** Los datos ya han sido preprocesados, balanceados y divididos usando el pipeline de preprocesamiento en `src/pipelines/preprocess.py`. La división es:
- **Entrenamiento:** 70% (3,856 imágenes - perfectamente balanceado en 4 clases)
- **Validación:** 15% (716 imágenes - estratificado)
- **Prueba:** 15% (722 imágenes - estratificado)

### Descripción Profesional de Módulos y Scripts

| Carpeta/Script | Descripción |
| :--- | :--- |
| **`data/train/`** | Dataset de entrenamiento con clases balanceadas (964 imágenes por clase) |
| **`data/val/`** | Dataset de validación con división estratificada para evaluación del modelo |
| **`data/test/`** | Dataset de prueba para evaluación final del modelo |
| **`experimentation/`** | Scripts de EDA y notebooks de Jupyter para exploración |
| **`models/exported/`** | Modelos finales entrenados listos para inferencia (ej. `best_VGG16.keras`) |
| **`src/adapters/data_loader.py`** | Componente de abstracción de datos. Encargado de cargar las imágenes desde el disco a memoria (PIL.Image) desde múltiples fuentes (data_1, data_2). |
| **`src/builders/builders.py`** | Factoría de modelos. Define la arquitectura de la cabeza de clasificación y ensambla el modelo completo (VGG16 + cabeza), listo para la búsqueda de hiperparámetros con Keras Tuner. |
| **`src/core/load_env.py`** | Utilidad de configuración. Carga y parsea de forma segura todas las variables de entorno (rutas, ratios, tamaños de imagen) desde el archivo `.env`. |
| **`src/pipelines/train.py`** | Script de orquestación central. Ejecuta la búsqueda de Keras Tuner, aplica Early Stopping y registra todos los resultados en MLflow. |
| **`src/pipelines/preprocess.py`** | Script principal del pipeline de datos. Dirige el filtrado, la unificación, la división estratificada y el balanceo (submuestreo/sobremuestreo) de los datos. |
| **`src/pipelines/infer.py`** | Pipeline de inferencia optimizado para el servidor. Carga el modelo final y contiene la función `predict()` utilizada por la API para clasificar imágenes. |
| **`src/utils/aug_detectors.py`** | Implementa la lógica de De-Augmentación; contiene las funciones para la generación de embeddings y el cálculo de la similitud del coseno para detectar duplicados. |
| **`src/utils/data_augmentator.py`** | Define las funciones para las transformaciones espaciales complejas utilizadas durante el oversampling controlado. |
| **`src/utils/image_modifier.py`** | Contiene funciones de bajo nivel para las transformaciones de calidad de imagen (ej., ruido, brillo, contraste) utilizadas en el Data Augmentation. |

---

## Inicio Rápido

### Instalación

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### Entrenamiento

```bash
# Entrenar modelo con configuración por defecto (VGG16, datos balanceados)
python -m src.pipelines.train

# Entrenar con un backbone específico
python -c "from src.pipelines.train import train; train(backbone_name='ResNet50')"
```

### Inferencia

```python
from src.pipelines.infer import predict

# Cargar imagen como bytes
with open('ruta/a/hoja_maiz.jpg', 'rb') as f:
    image_bytes = f.read()

# Obtener predicción
result = predict(image_bytes)
print(f"Predicción: {result['predicted_label']}")
print(f"Confianza: {result['confidence']:.2%}")
```

### Configuración

Editar `src/core/.env` para personalizar:
- `IMAGE_SIZE`: Dimensiones de la imagen de entrada
- `NUM_CLASSES`: Número de clases de enfermedades
- `BATCH_SIZE`: Tamaño del lote de entrenamiento
- `MAX_TRIALS`: Iteraciones de búsqueda de Keras Tuner

---

## Estrategia de Pruebas

El proyecto cuenta con una suite completa de pruebas usando pytest:

### Ejecutar Tests

```bash
# Ejecutar todos los tests
pytest

# Ejecutar con cobertura
pytest --cov=src --cov-report=html

# Ejecutar solo tests específicos
pytest tests/test_preprocess.py
pytest tests/test_augmentation.py

# Ejecutar tests con verbosidad
pytest -v
```

### Tests Implementados

**Tests de Preprocesamiento (`test_preprocess.py`):**
- Validación de ratios de división estratificada
- Verificación de proporciones correctas en train/val/test
- Manejo de categorías vacías
- Validación de la función `flatten_data`
- Verificación de dimensiones y tipos de datos

**Tests de Augmentación (`test_augmentation.py`):**
- Preservación de dimensiones tras transformaciones
- Validación de cada método de `ImageAugmentor`
- Reproducibilidad con semillas fijas
- Verificación de valores de píxeles en rango válido
- Consistencia de etiquetas tras augmentación

**Scripts de EDA (validación adicional):**
```bash
# Validar integridad del dataset
python experimentation/eda/validate_dataset.py

# Verificar distribuciones de clases
python experimentation/eda/explore_distribution.py

# Inspección visual de muestras
python experimentation/eda/view_samples.py
```

---

## Versionado de Modelos

**Versionado Automático:** El pipeline de entrenamiento guarda automáticamente los modelos con información de versión:

**Convención de Nombres de Archivo:**
```
models/exported/
├── VGG16_20250102_143022_acc0.9745.keras    # Con timestamp + precisión
├── VGG16_20250102_143022_metadata.json      # Configuración de entrenamiento
└── best_VGG16.keras                         # Último mejor modelo (para inferencia)
```

**Los Metadatos Incluyen:**
- Timestamp
- Precisión y pérdida en prueba
- Hiperparámetros utilizados
- Ratios de división de datos
- Estrategia de balanceo
- Tamaño de imagen y número de clases

**Registro de Modelos:** Todas las ejecuciones de entrenamiento se rastrean en MLflow en `models/mlruns/` para comparación de experimentos.

---

## Características Avanzadas

### Gestión de Configuración con Pydantic

El proyecto utiliza Pydantic para validación robusta de configuración:

```python
from src.core.config import config

# Acceso type-safe a la configuración
image_size = config.data.image_size  # (224, 224)
batch_size = config.training.batch_size  # 32
class_names = config.data.class_names  # ['Blight', ...]

# Validación automática de tipos y valores
# Si IMAGE_SIZE en .env es inválido, se lanza una excepción clara
```

**Beneficios:**
- Validación automática de tipos
- Valores por defecto razonables
- Errores claros cuando la configuración es inválida
- Autocompletado en IDEs

### Sistema de Logging Profesional

```python
from src.utils.logger import get_logger, log_section, log_dict

logger = get_logger(__name__)

logger.info("Iniciando entrenamiento...")
logger.warning("GPU no disponible, usando CPU")
logger.error("Error al cargar datos")

# Logging estructurado
log_section(logger, "Configuración de Entrenamiento")
log_dict(logger, {'lr': 0.001, 'epochs': 10}, "Hiperparámetros")
```

**Características:**
- Colores en terminal para mejor legibilidad
- Formato consistente en toda la aplicación
- Opción de guardar logs en archivos
- Niveles de logging configurables

### Manejo Robusto de Excepciones

Todas las excepciones incluyen sugerencias de recuperación:

```python
from src.utils.exceptions import NoModelToLoadError, DatasetNotFoundError

try:
    model = load_model('model.keras')
except NoModelToLoadError as e:
    print(e)
    # Imprime:
    # "No se encontró el modelo en: model.keras
    #
    # Sugerencia: Opciones:
    #   1. Entrenar un modelo ejecutando: python -m src.pipelines.train
    #   2. Verificar la ruta del modelo en la configuración
    #   ..."
```

**Excepciones Disponibles:**
- `DatasetNotFoundError`, `EmptyDatasetError`, `InvalidImageError`
- `NoModelToLoadError`, `ModelLoadError`, `InvalidBackboneError`
- `MissingConfigError`, `InvalidConfigError`, `NoLabelsError`
- `GPUNotAvailableError`, `InsufficientMemoryError`
- `TrainingDivergenceError`, `NoImprovementError`

### Manejo Centralizado de Rutas

```python
from src.utils.paths import paths

# Acceso limpio a rutas del proyecto
data_dir = paths.data_raw
models_dir = paths.models_exported
mlruns = paths.mlruns

# Crear directorios si no existen
paths.ensure_dir(paths.models_exported)

# Rutas relativas para logging
print(f"Guardando en: {paths.relative_to_root(model_path)}")
```

---

## Instalación y Desarrollo

### Instalación Básica

```bash
# Clonar repositorio
git clone https://github.com/ojgonzalezz/corn-diseases-detection.git
cd corn-diseases-detection

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -e .
```

### Instalación para Desarrollo

```bash
# Instalar con dependencias de desarrollo
pip install -e ".[dev]"

# Instalar todas las dependencias opcionales
pip install -e ".[all]"
```

### Configuración de pre-commit (opcional)

```bash
pip install pre-commit
pre-commit install
```

---

## Licencia

Este proyecto está bajo la licencia MIT.

---

## Contacto

Para preguntas o colaboraciones, por favor contactar a los mantenedores del proyecto.
