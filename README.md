# 🌽 Solución de visión por computadora para la detección de enfermedades en maíz

> Un proyecto de visión por computadora para la clasificación automática de enfermedades comunes en hojas de maíz, diseñado para ofrecer un diagnóstico rápido y preciso a los agricultores.

Este repositorio contiene todo el código, análisis y datos asociados al desarrollo de un modelo de Deep Learning capaz de identificar si una planta de maíz está sana o si padece una de tres enfermedades comunes: Roya Común, Tizón Foliar o Mancha Gris.

[![Licencia: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow: 2.x](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![FastAPI: Backend](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com/)

---

## 🚀 Demo en Vivo

Prueba el modelo en tiempo real. Sube o arrastra una imagen de una hoja de maíz y obtén un diagnóstico instantáneo con un historial de tus predicciones.

**[➡️ Acceder a la Aplicación Web Desplegada](https://felipepflorezo.github.io/corn-diseases-detection/)**

---

### Tabla de Contenidos
1. [Descripción del Proyecto](#-descripción-del-proyecto)
2. [Stack Tecnológico](#-stack-tecnológico)
3. [Metodología y Arquitectura del Modelo](#-metodología-y-arquitectura-del-modelo)
4. [Resultados Finales](#-resultados-finales)
5. [Arquitectura de Despliegue](#-arquitectura-de-despliegue)
6. [Dataset Utilizado](#-dataset-utilizado)
7. [Estructura del Repositorio](#-estructura-del-repositorio)
8. [Cómo Empezar Localmente](#-cómo-empezar-localmente)
9. [Contribuciones](#-contribuciones)
10. [Equipo de Trabajo](#-equipo-de-trabajo)

---

## 📜 Descripción del Proyecto

### El Problema
El maíz es un pilar de la seguridad alimentaria global, pero sus cultivos enfrentan amenazas constantes por enfermedades que reducen drásticamente el rendimiento. El método tradicional de diagnóstico es la inspección visual, un proceso lento, subjetivo y que requiere un alto nivel de experticia, lo que impide tomar acciones rápidas y efectivas para frenar la propagación.

### La Solución
Este proyecto resuelve el problema mediante una **solución de Inteligencia Artificial** que automatiza el diagnóstico. Se desarrolló un modelo de Deep Learning que analiza imágenes de hojas de maíz para identificar con alta precisión si una planta está sana o si padece una de tres enfermedades comunes: **Roya Común (Common Rust)**, **Tizón Foliar (Blight)** o **Mancha Gris (Gray Leaf Spot)**. El objetivo es empoderar a los agricultores con una herramienta de diagnóstico instantánea, objetiva y accesible.

---

## 🛠️ Stack Tecnológico

| Área                     | Tecnologías Utilizadas                                                                                                                                                             |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Análisis y Modelado** | Python, TensorFlow (Keras), Scikit-learn, Jupyter Notebooks                                                                                                                        |
| **Procesamiento de Datos** | Pandas, NumPy, Matplotlib, Seaborn, Pillow                                                                                                                                         |
| **Optimización y Backend** | **ONNX** (con `tf2onnx` y `onnxruntime`), **FastAPI**, Uvicorn, Requests                                                                                                                |
| **Despliegue y MLOps** | Git, **GitHub** (Código Fuente y Pages), **Hugging Face** (Hub para el modelo, Spaces para la API), **Docker** |

---

## 📊 Dataset utilizado

El modelo fue entrenado utilizando un conjunto de datos consolidado a partir de dos fuentes públicas para asegurar un volumen y una diversidad adecuados.

1.  **Fuente principal (Kaggle):** [Corn or Maize Leaf Disease Dataset](https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset)
2.  **Fuente de aumento (Roboflow):** [Corn Diseases Dataset](https://universe.roboflow.com/corn-disease-7/corn-diseases-oxojk)
3. **Dataset aumentado:** https://drive.google.com/drive/folders/16dK4pekmruoguRkIFG9lgdztTWkzBbUo?usp=sharing 

Inicialmente, el dataset de Kaggle presentaba un desbalance de clases. Para mitigarlo, se incorporaron imágenes de la fuente de Roboflow, específicamente en la clase con menor representación (*Gray Leaf Spot*), resultando en un conjunto de datos final y balanceado, ideal para el entrenamiento de un modelo robusto.

---

### Distribución inicial de clases
Inicialmente teniamos: 
  * **Roya Común (Common Rust):** 1,306 imágenes (27.3%)
  * **Mancha Gris (Gray Leaf Spot):** 1,171 imágenes (24.5%)
  * **Sana (Healthy):** 1,162 imágenes (24.3%)
  * **Tizón (Blight):** 1,146 imágenes (23.9%)

---

## ⚙️ Metodología y Arquitectura del Modelo

El proyecto siguió un flujo de trabajo iterativo y completo de Machine Learning:

1.  **Análisis Exploratorio de Datos (EDA):** Se analizaron los datasets, revelando un **desbalance de clases** significativo y una alta similitud visual entre las lesiones de *Blight* y *Gray Leaf Spot*, anticipando un desafío de clasificación.

2.  **Preprocesamiento y Balanceo:** Se aplicó **submuestreo (undersampling)** para crear un dataset perfectamente balanceado de 4,580 imágenes (1,145 por clase). Posteriormente, se dividió de forma estratificada en conjuntos de entrenamiento (70%), validación (15%) y prueba (15%). Se construyó un pipeline de datos para aplicar **aumento de datos en tiempo real** (rotaciones, zoom, etc.) al conjunto de entrenamiento.

3.  **Modelado y Entrenamiento (Iteración 1):**
    * Se implementó una arquitectura de **Transfer Learning** utilizando **VGG16** pre-entrenado en ImageNet como base.
    * Se "congeló" la base y se entrenaron capas de clasificación personalizadas, alcanzando una precisión inicial de **91.37%**.

4.  **Optimización (Iteración 2 - Ajuste Fino):**
    * Para mejorar el rendimiento, se aplicó **Ajuste Fino (Fine-Tuning)**. Se "descongelaron" las últimas 4 capas de VGG16 y se re-entrenó el modelo con una tasa de aprendizaje muy baja (`1e-5`).
    * Este proceso permitió que el modelo ajustara sus detectores de características a las sutilezas de las enfermedades del maíz.

5.  **Preparación para Despliegue (Conversión a ONNX):**
    * Para asegurar un despliegue eficiente y evitar problemas de memoria, el modelo final `.keras` fue convertido al formato **ONNX**. Esto redujo drásticamente el consumo de RAM y aceleró las predicciones en el servidor.

---

## 📈 Resultados Finales

La evaluación final se realizó sobre el conjunto de prueba utilizando el modelo optimizado tras el ajuste fino, confirmando la efectividad de la estrategia.

* **Exactitud Final (Accuracy):** **92.92%**
* **Pérdida (Loss):** **0.1989**

### Matriz de Confusión Final
La matriz confirma la alta efectividad del modelo. La diagonal principal (150, 164, 160, 169) muestra el número de predicciones correctas. Se observa que la confusión principal entre `Gray_Leaf_Spot` y `Blight` se redujo significativamente después del ajuste fino.

![Matriz de Confusión del Modelo Final](Figure_3.png)

### Reporte de Clasificación Final

| Clase          | Precision | Recall | F1-Score |
| :------------- | :-------: | :----: | :------: |
| Blight         |   0.90    |  0.87  |   0.88   |
| Common_Rust    |   0.98    |  0.95  |   0.96   |
| Gray_Leaf_Spot |   0.85    |  0.92  |   0.88   |
| Healthy        |   0.99    |  0.98  |   0.99   |

---

## ☁️ Arquitectura de Despliegue

La aplicación utiliza una arquitectura moderna y desacoplada:

* **Modelo (`.onnx`):** El artefacto entrenado se aloja en **Hugging Face Hub**.
* **Backend (API):** Una API construida con **FastAPI** se ejecuta dentro de un contenedor **Docker** en **Hugging Face Spaces**. Al iniciarse, la API descarga el modelo desde el Hub y expone un endpoint `/predict`.
* **Frontend:** La interfaz de usuario es una página estática (`index.html` con JavaScript) alojada en **GitHub Pages**, que se comunica con la API para ofrecer una experiencia interactiva.

---

## 📊 Dataset Utilizado

El modelo fue entrenado utilizando datos de dos fuentes públicas, posteriormente balanceados y procesados.
1.  **Fuente Principal (Kaggle):** [Corn or Maize Leaf Disease Dataset](https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset)
2.  **Fuente de Aumento (Roboflow):** [Corn Diseases Dataset](https://universe.roboflow.com/corn-disease-7/corn-diseases-oxojk)
3.  **Dataset Aumentado:** Un tercer dataset fue considerado y se puede encontrar en este [Google Drive](https://drive.google.com/drive/folders/16dK4pekmruoguRkIFG9lgdztTWkzBbUo?usp=sharing).

---

## 📁 Estructura del Repositorio

```
.
├── src/                      # Contiene todo el código fuente de Python
│   ├── api.py                # Lógica del backend con FastAPI
│   ├── model.py              # Arquitectura del modelo VGG16
│   ├── train.py              # Script para el entrenamiento inicial
│   ├── fine_tune.py          # Script para el ajuste fino
│   ├── evaluate.py           # Script para evaluar los modelos
│   ├── data_pipeline.py      # Generadores de datos con aumento
│   └── convert_to_onnx.py    # Script para optimizar el modelo
├── preprocessing/            # Scripts para la preparación inicial de datos
│   └── preprocess.py         # Balanceo y división del dataset
├── models/                   # (Local) Modelos generados - Ignorado por .gitignore
├── data/                     # (Local) Datasets - Ignorado por .gitignore
├── index.html                # Interfaz de usuario (Frontend)
├── requirements.txt          # Dependencias de Python
└── README.md                 # Este archivo
```

-----

## 🚀 Cómo Empezar Localmente

1.  **Clona el repositorio:**
    ```sh
    git clone https://github.com/ojgonzalezz/corn-diseases-detection.git
    ```
2.  **Navega al directorio:**
    ```sh
    cd corn-diseases-detection
    ```
3.  **Crea un entorno virtual e instala las dependencias:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```
4.  **Replicar el Proceso:** Para generar los resultados, ejecuta los scripts de la carpeta `src/` en orden: `train.py`, `fine_tune.py`, y `evaluate.py`.

-----

## 🤝 Contribuciones

Este repositorio es público para consulta. Las contribuciones al código son gestionadas de manera controlada para garantizar la integridad del proyecto.

  * El trabajo se organiza en **ramas individuales** por colaborador.
  * Todos los cambios deben ser integrados a la rama principal a través de **Pull Requests (PRs)**.
  * Cada PR debe ser **revisado y aprobado** por al menos otro miembro del equipo.

-----

## 🧑‍💻 Equipo de Trabajo

  * **Oscar Gonzalez:** Recolección y gestión de datos.
  * **Luis Macea:** Desarrollo del prototipo y gestión del repositorio GitHub.
  * **Felipe Florez:** Exploración y descripción de datos, gestión del repositorio DVC.
  * **Nicolas Castillo:** Exploración y descripción de datos, gestión del repositorio DVC.