# 🌽 Detector de Enfermedades en Maíz con IA: Un Proyecto Completo

> Un proyecto de Visión por Computadora de extremo a extremo que abarca desde el análisis de datos hasta el despliegue de un modelo de Deep Learning optimizado en una aplicación web interactiva.

[![Licencia: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow: 2.x](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![FastAPI: Backend](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com/)

---

## 🚀 Demo en Vivo

Prueba el modelo en tiempo real. Sube o arrastra una imagen de una hoja de maíz y obtén un diagnóstico instantáneo con un historial de tus predicciones.

**[➡️ Acceder a la Aplicación Web Desplegada](https://felipepflorezo.github.io/corn-diseases-detection/)** *(Nota: Reemplaza con la URL final de tu GitHub Pages)*

![Demostración de la aplicación web](https://i.imgur.com/225956.png) 

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