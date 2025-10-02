🌽 Análisis Exploratorio de Datos
Detección de Enfermedades en Cultivos de Maíz
Este documento presenta los hallazgos de la primera fase del proyecto, centrada en el análisis y la comprensión de los datos iniciales.

📜 Problema y Contexto
Las enfermedades del maíz, como la roya común, el tizón foliar y la mancha gris, representan una amenaza crítica para la seguridad alimentaria. El diagnóstico tradicional mediante inspección visual es un proceso lento, subjetivo y dependiente de la pericia del observador. Este proyecto busca validar la viabilidad de un sistema de diagnóstico automatizado mediante Inteligencia Artificial para superar estas limitaciones.

📊 Dataset Inicial
Para el análisis, se utilizó el dataset público "Corn or Maize Leaf Disease Dataset" de Kaggle, una compilación de imágenes de las fuentes PlantVillage y PlantDoc.

Total de Imágenes: 4,188

Formato: JPEG (.jpg)

Distribución de Clases (Inicial):

Roya Común (Common Rust): 1,306 imágenes (31.2%)

Sana (Healthy): 1,162 imágenes (27.7%)

Tizón (Blight): 1,146 imágenes (27.4%)

Mancha Gris (Gray Leaf Spot): 574 imágenes (13.7%)

Observación Clave: El dataset inicial presenta un notable desbalance, con la clase "Mancha Gris" significativamente subrepresentada. Este hallazgo es fundamental para las siguientes etapas del proyecto.

🔬 Hallazgos del Análisis Exploratorio de Datos (EDA)
Validación e Integridad de Datos
Se realizó una validación estructural del dataset para confirmar la cantidad de clases, el número de imágenes y la integridad de los archivos. Se encontró y corrigió una inconsistencia de formato (un archivo .jpeg en lugar de .jpg) en la clase "Blight", asegurando la homogeneidad del conjunto de datos.

Análisis Cualitativo Visual
La inspección de muestras aleatorias reveló una buena calidad de imagen general (nitidez y enfoque). Se destacó una alta variabilidad en iluminación, escala y ángulos de captura, lo cual es beneficioso para entrenar un modelo más robusto y generalizable.

Desafío Principal Identificado: Se observó una alta similitud morfológica entre las lesiones en etapas avanzadas de "Mancha Gris" y "Tizón", lo que anticipa el principal reto de clasificación para el modelo de IA.

Análisis Cuantitativo de Características Físicas
Dimensiones: Se confirmó una considerable variabilidad en el tamaño (alto y ancho) de las imágenes, lo que fundamenta la necesidad de un paso de redimensionamiento estándar antes de alimentar el modelo.

Distribución de Color: El análisis de histogramas de color, particularmente en el canal verde, demostró ser un rasgo altamente discriminatorio. Las hojas sanas ("Healthy") mostraron un perfil de color verde único y vibrante, claramente distinto al de las hojas enfermas. Esto valida el potencial del color como una característica potente para la clasificación automática y justifica la necesidad de normalizar los valores de los píxeles.

# Estructura del repositorio

mi_proyecto_maiz_dl/
│
├── data/
│   ├── raw/                  # Datos originales, sin modificar (puedes enlazar a ellos)
│   │   ├── train/
│   │   │   ├── Healthy/
│   │   │   └── Blight/
│   │   │   └── ...
│   │   └── validation/
│   │       ├── Healthy/
│   │       └── Blight/
│   │       └── ...
│   │
│   ├── processed/            # Datos limpios y listos para el entrenamiento
│   │   ├── images_resized/
│   │   └── train_labels.csv
│   │
│   └── external/             # Conjuntos de datos de terceros
│
├── notebooks/
│   ├── 01_eda_exploracion.ipynb    # Notebooks para el EDA y visualización de datos
│   ├── 02_modelado_basico.ipynb    # Experimentación con modelos iniciales
│   └── 03_transfer_learning.ipynb  # Pruebas con técnicas más avanzadas
│
├── models/
│   ├── checkpoints/          # Puntos de control (checkpoints) durante el entrenamiento
│   │   ├── best_model.h5
│   │   └── epoch_10.h5
│   │
│   └── exported/             # Versiones finales de modelos para producción/uso
│       ├── final_model.h5
│       └── tflite_model.tflite
│
├── src/                      # Código fuente de producción
│   ├── __init__.py           # Hace que el directorio sea un paquete Python
│   │
│   ├── data_pipeline.py      # Script para carga, preprocesamiento y aumento de datos
│   ├── model.py              # Definición de la arquitectura del modelo
│   ├── train.py              # Script principal para el entrenamiento del modelo
│   └── predict.py            # Script para realizar predicciones
│
├── utils/
│   ├── __init__.py
│   ├── helpers.py            # Funciones de ayuda (por ejemplo, para graficar)
│   └── metrics.py            # Funciones para calcular métricas personalizadas
│
├── reports/
│   ├── figures/              # Gráficos generados
│   │   ├── class_distribution.png
│   │   └── image_dimensions.png
│   │
│   └── report.pdf            # Un informe final con los hallazgos
│
├── requirements.txt          # Lista de librerías y dependencias
├── README.md                 # Descripción del proyecto, cómo instalar y usar
└── .gitignore                # Archivos a ignorar por Git (ej: datos grandes, checkpoints)


¡Claro\! Aquí está la información que puedes usar para la sección de instalación de tu archivo `README.md`. He organizado la información de manera clara, agregando comentarios para cada paso y resaltando las especificaciones técnicas de tu equipo.

-----

## 💻 Requisitos y Configuración del Entorno

### Especificaciones Técnicas

El proyecto ha sido desarrollado y probado en la siguiente configuración de hardware:

  * **Tarjeta Gráfica (GPU):** NVIDIA RTX 4060
  * **Versión Máxima de CUDA Compatible:** 12.5

### Instalación de CUDA y cuDNN

Para replicar el entorno de desarrollo, es necesario instalar las versiones compatibles de CUDA y cuDNN.

1.  **Verificar la Compatibilidad:** Antes de comenzar, ejecuta el siguiente comando en tu terminal para confirmar que el controlador de tu tarjeta gráfica soporta la versión de CUDA que vas a instalar. La versión de CUDA mostrada debe ser mayor o igual a la que se desea instalar.

    ```bash
    nvidia-smi
    ```

      * **Nota:** Aunque tu tarjeta es compatible con CUDA 12.5, el proyecto utiliza **CUDA 12.4** para mantener la compatibilidad con las librerías de PyTorch.

2.  **Instalación de cuDNN:** Una vez que el entorno de Conda esté activo, instala el kit de desarrollo de cuDNN.

    ```bash
    conda install nvidia::cudnn cuda-version=12.4
    ```

      * **Comentario:** Este comando instala la biblioteca de redes neuronales profundas (cuDNN), que es crucial para acelerar las operaciones de redes neuronales en la GPU.

3.  **Instalación de PyTorch:** Utiliza `pip` para instalar las librerías principales de PyTorch, especificando la versión de CUDA.

    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    ```

      * **Comentario:** Este comando descarga las versiones de `torch`, `torchvision` y `torchaudio` compiladas para la versión de CUDA 12.4, asegurando que el soporte de la GPU esté habilitado.

-----



NOTAS:
para levantar el aplicativo en local, dirifase a:
corn-diseases-detection-api

ejecute:

uvicorn main:app --reload