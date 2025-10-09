"""
Configuración común para el entrenamiento de modelos
"""

import os
from pathlib import Path

# Rutas
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data_processed'
MODELS_DIR = BASE_DIR / 'entrenamiento_modelos' / 'models'
LOGS_DIR = BASE_DIR / 'entrenamiento_modelos' / 'logs'
MLFLOW_DIR = BASE_DIR / 'entrenamiento_modelos' / 'mlruns'

# Crear directorios
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
MLFLOW_DIR.mkdir(parents=True, exist_ok=True)

# Clases
CLASSES = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
NUM_CLASSES = len(CLASSES)

# Hiperparámetros comunes
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5

# División del dataset
TRAIN_SPLIT = 0.70  # 70% entrenamiento
VAL_SPLIT = 0.15    # 15% validación
TEST_SPLIT = 0.15   # 15% prueba

# Semilla para reproducibilidad
RANDOM_SEED = 42

# GPU Configuration
GPU_MEMORY_LIMIT = 10240  # Limite de memoria GPU en MB (ajustar segun disponibilidad)

# MLflow
MLFLOW_TRACKING_URI = f"file:///{MLFLOW_DIR}"
MLFLOW_EXPERIMENT_NAME = "corn_disease_classification"

# Data Augmentation
DATA_AUGMENTATION = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'horizontal_flip': True,
    'vertical_flip': True,
    'zoom_range': 0.2,
    'fill_mode': 'nearest'
}
