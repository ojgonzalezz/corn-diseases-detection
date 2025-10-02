
-----

# 🌽 Detección de Enfermedades del Maíz con Transfer Learning

## 🌟 Resumen del Proyecto

Este proyecto implementa un *pipeline* de *Deep Learning* robusto para la **clasificación de enfermedades comunes en hojas de maíz**. El objetivo es diagnosticar automáticamente la salud de las plantas utilizando técnicas de *Transfer Learning* basadas en la arquitectura VGG16, optimizando la cabeza de clasificación mediante **Keras Tuner** y rastreando todos los experimentos con **MLflow**.

El proyecto se destaca por su rigurosa estrategia de preprocesamiento, que aborda activamente el **sesgo por duplicación (data leakage)** y el **desbalance de clases** en un *dataset* bimodal compuesto por dos fuentes de datos distintas.

-----

📜 Problema y Contexto
Las enfermedades del maíz, como la roya común, el tizón foliar y la mancha gris, representan una amenaza crítica para la seguridad alimentaria. El diagnóstico tradicional mediante inspección visual es un proceso lento, subjetivo y dependiente de la pericia del observador. Este proyecto busca validar la viabilidad de un sistema de diagnóstico automatizado mediante Inteligencia Artificial para superar estas limitaciones.

-----

## 🎯 Objetivo y Tipos de Datos

### Objetivo Principal

Desarrollar un modelo de clasificación de imágenes altamente preciso y generalizable, capaz de diferenciar entre las siguientes cuatro categorías de salud de las hojas de maíz:

1.  **Blight**
2.  **Common\_Rust**
3.  **Gray\_Leaf\_Spot**
4.  **Healthy**

### Tipos de Datos

El *dataset* está compuesto por imágenes RGB de hojas de maíz, recopiladas de dos fuentes distintas que se manejan por separado para controlar el *Data Augmentation*:

| Fuente | Descripción | Consideración |
| :--- | :--- | :--- |
| **`data_1`** | Dataset primario (ej., Kaggle). | **No-Augmentation** (Se considera limpio y original). |
| **`data_2`** | Dataset secundario (ej., Roboflow). | **Augmented** (Contiene *Data Augmentation* preaplicado, introduciendo riesgo de sesgo). |

-----

## 🛠️ Retos de Datos y Pipeline de Preprocesamiento

El mayor desafío del proyecto fue mitigar el **sesgo por duplicación** presente en la fuente `data_2`, lo que habría inflado artificialmente las métricas de prueba. Para garantizar la integridad del modelo, se estableció un *pipeline* de tres fases:

### 1\. Desafío: Sesgo por Duplicación (*Data Leakage*)

El sesgo por duplicación ocurre cuando una misma imagen o una copia casi idéntica (resultado de un *Data Augmentation* simple) se encuentra tanto en el conjunto de entrenamiento como en el de prueba, lo que conduce al **sobreajuste** (el modelo memoriza en lugar de generalizar).

### Estrategia de De-Augmentación por Embedding

Para contrarrestar esto, se aplicó una estrategia innovadora de filtrado antes de la división de datos:

  * **Generación de Embeddings:** Se generaron vectores de características (*embeddings*) para todas las imágenes en la fuente `"augmented"` (`data_2`) utilizando un modelo preentrenado (ej., la base VGG16).
  * **Similitud del Coseno ($\cos(\theta)$):** Se calculó la similitud del coseno entre todos los pares de *embeddings* dentro de cada categoría. Las imágenes con una similitud superior a un umbral alto ($\tau \ge 0.95$) fueron identificadas como duplicados artificiales.
  * **Filtrado:** El *pipeline* eliminó una de las imágenes del par similar, asegurando que solo **una versión representativa** de cada imagen original se conservara en el conjunto de datos final.

### 2\. División Estratificada y Balanceo de Clases

Una vez filtrados los duplicados, las imágenes de `data_1` y las filtradas de `data_2` se unificaron y se procesaron para la fase de entrenamiento:

  * **Unificación y División Estratificada:** El conjunto de datos unificado se dividió en `Train`, `Validation` y `Test` utilizando la función **`stratified_split_dataset`**. Esta división garantiza que la proporción de cada clase de enfermedad sea la misma en los tres conjuntos, lo cual es fundamental para una evaluación imparcial.
  * **Balanceo (Solo en el set de Train):** Para corregir el desbalance, solo se modifica el conjunto de entrenamiento, evitando la contaminación de los sets de validación y prueba.
      * **Modo Downsampling:** Reduce el número de imágenes de todas las clases al tamaño de la clase minoritaria.
      * **Modo Oversampling:** Expande las clases minoritarias hasta un tamaño objetivo (clase mayoritaria + N extra) utilizando una estrategia de **aumento en cascada** (doble transformación de calidad seguida de una transformación espacial) para generar variaciones más robustas.

-----

## 🧠 Metodología General de Entrenamiento (Métricas y Optimización)

El proyecto emplea una metodología rigurosa para el entrenamiento y el seguimiento de los experimentos:

1.  **Transfer Learning:** Se utiliza el modelo **VGG16** preentrenado en ImageNet como *backbone*, congelando sus capas convolucionales para aprovechar las características visuales aprendidas.
2.  **Búsqueda de Hiperparámetros (Keras Tuner):** Se utiliza la técnica **Hyperband** de Keras Tuner para encontrar la arquitectura óptima para la cabeza de clasificación (las capas densas) que se añade al *backbone* de VGG16.
3.  **Seguimiento de Experimentos (MLflow):** Todos los *trials* generados por Keras Tuner son registrados como *runs* individuales en MLflow. Esto incluye:
      * Registro de hiperparámetros por *trial*.
      * Registro de métricas por época (`loss`, `accuracy`, `val_loss`, `val_accuracy`).
      * Guardado del modelo final (`best_model.h5`) y las métricas de rendimiento en el conjunto de prueba.

-----

## 📂 Estructura Detallada del Repositorio

El proyecto sigue una estructura modular y escalable para separar el código de producción (`src`), los datos (`data`), y la experimentación (`notebooks`).

```
corn-diseases-detection/
mi_proyecto_maiz_dl/
├── data/
│   ├── raw/                  # Datos originales (no modificados)
│   │   ├── train/            # Dataset de entrenamiento original
│   │   └── validation/       # Dataset de validación original
│   └── processed/            # Datos limpios y listos para el ciclo ML
│       ├── data_1            # Fuente 1 (limpia/no-augmentada)
│       ├── data_2            # Fuente 2 (des-aumentada/filtrada)
│       └── split             # Conjuntos finales para el modelo
│           ├── train/        # Conjunto de Entrenamiento (balanceado)
│           ├── val/          # Conjunto de Validación (estratificado)
│           └── test/         # Conjunto de Prueba (estratificado)
│
├── notebooks/                # Espacio para experimentación y EDA
│   ├── 01_eda_exploracion.ipynb 
│   ├── 02_modelado_basico.ipynb
│   └── 03_transfer_learning.ipynb
│
├── models/                   # Artefactos de modelos
│   ├── checkpoints/          # Puntos de control intermedios (MLflow)
│   └── exported/             # Versiones finales para inferencia/producción
│
├── src/                      # Código fuente de producción
│   ├── adapters/
│   │   └── data_loader.py    # Abstracción: Carga de datos de múltiples fuentes
│   ├── builders/
│   │   ├── base_models.py    # Definición de backbones preentrenados (VGG16)
│   │   └── builders.py       # Ensamblaje de la cabeza de clasificación para Keras Tuner
│   ├── core/                 # Configuración y utilidades de entorno
│   │   ├── load_env.py       # Carga de variables de entorno (.env)
│   │   └── path_finder.py    # Detección de la ruta raíz del proyecto
│   ├── pipelines/            # Scripts del ciclo de Machine Learning
│   │   ├── data_pipeline.py  # Generación de DataGenerators para Keras
│   │   ├── evaluate_finetuned.py
│   │   ├── train.py          # Orquestación del entrenamiento con Keras Tuner/MLflow
│   │   ├── preproces.py      # Script principal de filtrado, unificación y balanceo (Down/Oversampling)
│   │   └── infer.py          # Lógica de inferencia para la API (clasificación)
│   └── utils/                # Funciones de ayuda
│       ├── aug_detectors.py    # Detección y filtrado de aumentaciones por Embedding
│       ├── data_augmentator.py # Transformaciones espaciales para Oversampling
│       ├── image_modifier.py   # Transformaciones de calidad (brillo, contraste, ruido)
│       └── utils.py          # Utilidades misceláneas
│
├── reports/                  # Documentación y resultados
├── requirements.txt
├── README.md
└── .gitignore
```

### Descripción Profesional de Módulos y Scripts

| Carpeta/Script | Descripción |
| :--- | :--- |
| **`data/raw/`** | Contiene los datasets originales. Estos archivos nunca deben ser modificados por el código. |
| **`data/processed/`** | Almacena los datasets limpios, filtrados y listos para el consumo del modelo. Contiene los subdirectorios `data_1`, `data_2` (filtrados) y `split` (conjuntos finales). |
| **`notebooks/`** | Entorno para la experimentación, EDA, y prototipado inicial. |
| **`models/exported/`** | Directorio para la versión final del modelo (ej., `final_model.h5`) que está lista para el despliegue. |
| **`src/adapters/data_loader.py`** | Componente de abstracción de datos. Encargado de cargar las imágenes desde el disco a memoria (`PIL.Image`) desde múltiples fuentes (`data_1`, `data_2`). |
| **`src/builders/builders.py`** | Factoría de modelos. Define la arquitectura de la cabeza de clasificación y ensambla el modelo completo (VGG16 + cabeza), listo para la búsqueda de hiperparámetros con Keras Tuner. |
| **`src/core/load_env.py`** | Utilidad de configuración. Carga y parsea de forma segura todas las variables de entorno (rutas, ratios, tamaños de imagen) desde el archivo `.env`. |
| **`src/pipelines/train.py`** | Script de orquestación central. Ejecuta la búsqueda de Keras Tuner, aplica *Early Stopping* y registra todos los resultados en MLflow. |
| **`src/pipelines/preprocess.py`** | Script principal del *pipeline* de datos. Dirige el filtrado, la unificación, la división estratificada y el balanceo (submuestreo/sobremuestreo) de los datos. |
| **`src/pipelines/infer.py`** | Pipeline de inferencia optimizado para el servidor. Carga el modelo final y contiene la función **`predict()`** utilizada por la API para clasificar imágenes. |
| **`src/utils/aug_detectors.py`** | Implementa la lógica de **De-Augmentación**; contiene las funciones para la generación de *embeddings* y el cálculo de la similitud del coseno para detectar duplicados. |
| **`src/utils/data_augmentator.py`** | Define las funciones para las transformaciones espaciales complejas utilizadas durante el *oversampling* controlado. |
| **`src/utils/image_modifier.py`** | Contiene funciones de bajo nivel para las transformaciones de *calidad* de imagen (ej., ruido, brillo, contraste) utilizadas en el *Data Augmentation*. |
