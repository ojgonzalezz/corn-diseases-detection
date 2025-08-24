# Solución de visión por computadora para la detección de enfermedades en maíz

> Un proyecto de visión por computadora para la clasificación automática de enfermedades comunes en hojas de maíz, diseñado para ofrecer un diagnóstico rápido y preciso a los agricultores.

Este repositorio contiene todo el código, análisis y datos asociados al desarrollo de un modelo de Deep Learning capaz de identificar si una planta de maíz está sana o si padece una de tres enfermedades comunes: Roya Común, Tizón Foliar o Mancha Gris.

## 📜 Descripción del proyecto

### El problema

El maíz es un pilar de la seguridad alimentaria global, pero sus cultivos enfrentan amenazas constantes por enfermedades que reducen drásticamente el rendimiento. El método tradicional de diagnóstico es la inspección visual, un proceso lento, subjetivo y que requiere un alto nivel de experticia, lo que impide tomar acciones rápidas y efectivas para frenar la propagación.

### La solución

Este proyecto desarrolla una solución basada en **Inteligencia Artificial** que automatiza el diagnóstico. Mediante un modelo de visión por computadora, la herramienta analiza una imagen de una hoja de maíz y determina con alta precisión si está sana o clasifica la enfermedad específica que la afecta.

El objetivo es empoderar a los agricultores con un diagnóstico **instantáneo, objetivo y accesible**, permitiendo una gestión proactiva de la salud del cultivo para minimizar pérdidas y optimizar el uso de recursos.

## 📊 Dataset utilizado

El modelo fue entrenado utilizando un conjunto de datos consolidado a partir de dos fuentes públicas para asegurar un volumen y una diversidad adecuados.

1.  **Fuente principal (Kaggle):** [Corn or Maize Leaf Disease Dataset](https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset)
2.  **Fuente de aumento (Roboflow):** [Corn Diseases Dataset](https://universe.roboflow.com/corn-disease-7/corn-diseases-oxojk)
3. **Dataset aumentado:** https://drive.google.com/drive/folders/16dK4pekmruoguRkIFG9lgdztTWkzBbUo?usp=sharing 

Inicialmente, el dataset de Kaggle presentaba un desbalance de clases. Para mitigarlo, se incorporaron imágenes de la fuente de Roboflow, específicamente en la clase con menor representación (*Gray Leaf Spot*), resultando en un conjunto de datos final y balanceado, ideal para el entrenamiento de un modelo robusto.

### Distribución final de clases

  * **Roya Común (Common Rust):** 1,306 imágenes (27.3%)
  * **Mancha Gris (Gray Leaf Spot):** 1,171 imágenes (24.5%)
  * **Sana (Healthy):** 1,162 imágenes (24.3%)
  * **Tizón (Blight):** 1,146 imágenes (23.9%)

## ⚙️ Metodología

El desarrollo del proyecto siguió los siguientes pasos clave:

1.  **Análisis Exploratorio de Datos (EDA):** Se realizó un análisis exhaustivo de las imágenes para entender sus características. Se validó la integridad de los datos, se cuantificó el desbalance de clases y se analizaron las propiedades visuales (dimensiones y perfiles de color), confirmando que el color es un rasgo altamente discriminatorio.
2.  **Balanceo de clases:** Se aplicó una estrategia de aumento de datos, incorporando imágenes externas para balancear la distribución de clases y evitar sesgos en el modelo.
3.  **Preprocesamiento de imágenes:** Se definieron y aplicaron transformaciones necesarias, como el redimensionamiento a un tamaño estándar (ej. 224x224 px) y la normalización de los valores de los píxeles.
4.  **Entrenamiento del modelo:** Se desarrolló y entrenó un modelo de clasificación de imágenes para distinguir entre las cuatro categorías.
5.  **Prototipo:** Se diseñó una maqueta funcional que permite a un usuario cargar una imagen y recibir la predicción del modelo con las probabilidades asociadas a cada clase.

## 🚀 Prototipo

El prototipo es una interfaz simple donde el usuario puede:

1.  Cargar una imagen de una hoja de maíz desde su dispositivo.
2.  Hacer clic en el botón "Predecir".
3.  Recibir como resultado la clasificación (ej. "Roya Común") junto con el porcentaje de confianza de la predicción para cada una de las cuatro clases.

## 🛠️ Cómo empezar

Para clonar y ejecutar este proyecto localmente, sigue estos pasos:

1.  **Clona el repositorio:**
    ```sh
    git clone https://github.com/ojgonzalezz/corn-diseases-detection.git
    ```
2.  **Navega al directorio del proyecto:**
    ```sh
    cd corn-diseases-detection
    ```
3.  **Instala las dependencias (se recomienda usar un entorno virtual):**
    ```sh
    pip install -r requirements.txt
    ```
4.  **Explora los notebooks y scripts** en las carpetas correspondientes para replicar los análisis y el entrenamiento.

## 🤝 Contribuciones

Este repositorio es público para consulta. Las contribuciones al código son gestionadas de manera controlada para garantizar la integridad del proyecto. Solo tienen acceso losss colaboradores del proyeeeeecto.

  * El trabajo se organiza en **ramas individuales** por colaborador.
  * Todos los cambios deben ser integrados a la rama principal (`main` o `develop`) a través de **Pull Requests (PRs)**.
  * Cada PR debe ser **revisado y aprobado** por al menos un otro miembro del equipo antes de ser fusionado.

## 🧑‍💻 Equipo de trabajo

  * **Oscar Gonzalez:** Recolección y gestión de datos.
  * **Luis Macea:** Desarrollo del prototipo y gestión del repositorio GitHub.
  * **Felipe Florez:** Exploración y descripción de datos, gestión del repositorio DVC.
  * **Nicolas Castillo:** Exploración y descripción de datos, gestión del repositorio DVC.

*La redacción de la problemática y la pregunta de negocio fue un esfuerzo conjunto de todo el equipo.*
