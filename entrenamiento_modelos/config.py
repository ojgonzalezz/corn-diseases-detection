"""
Configuración común para el entrenamiento de modelos
Compatible con Google Colab (con Drive) y entornos locales
"""

import os
from pathlib import Path

# Detectar si estamos en Google Colab
try:
    import google.colab
    IN_COLAB = True

    # Montar Google Drive automáticamente
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)

    # Rutas en Google Drive
    DRIVE_BASE = Path('/content/drive/MyDrive/corn-diseases-detection')
    BASE_DIR = Path('/content/corn-diseases-detection')

    # Dataset desde Drive
    DATA_DIR = DRIVE_BASE / 'data_processed'

    # Salidas en Drive (persistentes)
    MODELS_DIR = DRIVE_BASE / 'models'
    LOGS_DIR = DRIVE_BASE / 'logs'
    MLFLOW_DIR = DRIVE_BASE / 'mlruns'

    print("=" * 60)
    print("CONFIGURACIÓN GOOGLE COLAB + DRIVE")
    print("=" * 60)
    print(f"Dataset (entrada): {DATA_DIR}")
    print(f"Modelos (salida): {MODELS_DIR}")
    print(f"Logs (salida): {LOGS_DIR}")
    print("=" * 60)

except ImportError:
    IN_COLAB = False
    BASE_DIR = Path(__file__).parent.parent

    # Rutas locales
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
# En Google Colab, la GPU se gestiona automáticamente
# En entornos locales, puedes ajustar este valor según tu GPU
GPU_MEMORY_LIMIT = None  # None = usar toda la memoria disponible

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
