# Detección de Enfermedades del Maíz

Sistema de deep learning para clasificación de enfermedades en hojas de maíz usando arquitecturas ligeras optimizadas para edge computing.

## Inicio Rápido

1. Ver [docs/README.md](docs/README.md) para instrucciones completas
2. Ejecutar [notebooks/colab_edge_models_training.ipynb](notebooks/colab_edge_models_training.ipynb) en Google Colab con GPU

## Estructura del Proyecto

```
corn-diseases-detection/
├── data/                     # Dataset (divisiones train/val/test)
├── src/                      # Código fuente
│   ├── adapters/            # Carga de datos
│   ├── builders/            # Constructores de modelos edge
│   ├── core/                # Configuración central
│   ├── pipelines/           # Pipelines ML
│   └── utils/               # Utilidades
├── EDA/                     # Análisis exploratorio de datos
│   ├── eda/                # Scripts de análisis EDA
│   └── notebooks/          # Notebooks de exploración
├── experiments/             # Experimentos edge computing
│   └── edge_models/        # Entrenamiento arquitecturas ligeras
├── mobilenetv3_inference/   # Pipeline MobileNetV3 completo
│   ├── config.yaml         # Configuración específica
│   ├── convert_to_tflite.py # Conversión a TensorFlow Lite
│   ├── inference.py        # Pipeline de inferencia
│   ├── run_pipeline.py     # Ejecución automática
│   ├── validate_model.py   # Validación y métricas
│   └── README.md          # Documentación específica
├── models/                 # Modelos y experimentos
│   ├── exported/          # Modelos exportados
│   └── mlruns/            # Experimentos MLflow
├── tests/                  # Suite de pruebas (10 archivos)
├── docs/                   # Documentación
│   ├── README.md          # Documentación completa
│   └── LICENSE            # Información de licencia
├── notebooks/             # Notebooks Jupyter
│   └── colab_edge_models_training.ipynb  # Entrenamiento en Colab
├── requirements.txt       # Dependencias Python
├── .gitignore            # Reglas git ignore
└── README.md             # Este archivo
```
